/*
 * postprocess_solution.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include "buoyant_fluid_solver.h"
#include "postprocessor.h"

namespace BuoyantFluid {

template<int dim>
double BuoyantFluidSolver<dim>::compute_kinetic_energy() const
{
    const QGauss<dim> quadrature_formula(parameters.velocity_degree + 1);

    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> stokes_fe_values(mapping,
                                   navier_stokes_fe,
                                   quadrature_formula,
                                   update_values|update_JxW_values);

    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double local_kinetic_energy = 0;

    for (auto cell: navier_stokes_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            stokes_fe_values.reinit(cell);

            stokes_fe_values[velocities].get_function_values(navier_stokes_solution,
                    velocity_values);

            for (unsigned int q=0; q<n_q_points; ++q)
                local_kinetic_energy += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
        }

    AssertIsFinite(local_kinetic_energy);
    Assert(local_kinetic_energy >= 0, ExcLowerRangeType<double>(local_kinetic_energy, 0));

    return Utilities::MPI::sum(local_kinetic_energy, mpi_communicator);
}

template<int dim>
std::pair<double, double> BuoyantFluidSolver<dim>::compute_rms_values() const
{
    const QGauss<dim> quadrature_formula(parameters.velocity_degree + 1);

    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> stokes_fe_values(mapping,
                                   navier_stokes_fe,
                                   quadrature_formula,
                                   update_values|update_JxW_values);

    FEValues<dim> temperature_fe_values(mapping,
                                        temperature_fe,
                                        quadrature_formula,
                                        update_values);

    std::vector<double>         temperature_values(n_q_points);
    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double local_sum_velocity = 0;
    double local_sum_temperature = 0;
    double local_volume = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = navier_stokes_dof_handler.begin_active(),
    temperature_cell = temperature_dof_handler.begin_active(),
    endc = navier_stokes_dof_handler.end();

    for (; cell != endc; ++cell, ++temperature_cell)
    if (cell->is_locally_owned())
    {
        stokes_fe_values.reinit(cell);
        temperature_fe_values.reinit(temperature_cell);

        temperature_fe_values.get_function_values(temperature_solution,
                                                  temperature_values);
        stokes_fe_values[velocities].get_function_values(navier_stokes_solution,
                                                         velocity_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            local_sum_velocity += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
            local_sum_temperature += temperature_values[q] * temperature_values[q] * stokes_fe_values.JxW(q);
            local_volume += stokes_fe_values.JxW(q);
        }
    }

    AssertIsFinite(local_sum_velocity);
    Assert(local_sum_velocity >= 0, ExcLowerRangeType<double>(local_sum_velocity, 0));
    AssertIsFinite(local_sum_temperature);
    Assert(local_sum_temperature >= 0, ExcLowerRangeType<double>(local_sum_temperature, 0));
    Assert(local_volume >= 0, ExcLowerRangeType<double>(local_volume, 0));

    const double local_sums[3]  = { local_sum_velocity, local_sum_temperature, local_volume};
    double global_sums[3];

    Utilities::MPI::sum(local_sums, mpi_communicator, global_sums);

    const double rms_velocity = global_sums[0] / global_sums[2];
    const double rms_temperature = global_sums[1] / global_sums[2];

    return std::pair<double,double>(std::sqrt(rms_velocity), std::sqrt(rms_temperature));
}

template <int dim>
double BuoyantFluidSolver<dim>::compute_cfl_number() const
{
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping,
                            navier_stokes_fe,
                            quadrature_formula,
                            update_values);

    std::vector<Tensor<1,dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double max_cfl = 0;

    for (auto cell : navier_stokes_dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        fe_values.reinit (cell);
        fe_values[velocities].get_function_values(navier_stokes_solution,
                                                  velocity_values);
        double max_cell_velocity = 0;
        for (unsigned int q=0; q<n_q_points; ++q)
            max_cell_velocity = std::max(max_cell_velocity,
                                         velocity_values[q].norm());
        max_cfl = std::max(max_cfl,
                           max_cell_velocity / cell->diameter());
    }
    const double local_cfl = max_cfl * timestep;

    const double global_cfl
    = Utilities::MPI::max(local_cfl, mpi_communicator);

    return global_cfl;
}

template<int dim>
void BuoyantFluidSolver<dim>::output_results(const bool initial_condition) const
{
    if (parameters.verbose)
        pcout << "Output results..." << std::endl;

    // create joint finite element
    const FE_DGQ<dim>   aux_fe(0);
    const FESystem<dim> joint_fe(navier_stokes_fe, 1,
                                 temperature_fe, 1,
                                 aux_fe, 1);

    // create joint dof handler
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    DoFHandler<dim>     aux_dof_handler(triangulation);
    aux_dof_handler.distribute_dofs(aux_fe);

    Assert(joint_dof_handler.n_dofs() ==
           navier_stokes_dof_handler.n_dofs() +
           temperature_dof_handler.n_dofs() +
           aux_dof_handler.n_dofs(),
           ExcInternalError());

    // create joint solution
    LA::Vector  distributed_joint_solution;
    distributed_joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                                      mpi_communicator);
    {
        std::vector<types::global_dof_index> local_joint_dof_indices(joint_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_stokes_dof_indices(navier_stokes_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_temperature_dof_indices(temperature_fe.dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
        joint_cell       = joint_dof_handler.begin_active(),
        joint_endc       = joint_dof_handler.end(),
        stokes_cell      = navier_stokes_dof_handler.begin_active(),
        temperature_cell = temperature_dof_handler.begin_active();
        for (; joint_cell!=joint_endc; ++joint_cell, ++stokes_cell, ++temperature_cell)
        if (joint_cell->is_locally_owned())
        {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            stokes_cell->get_dof_indices(local_stokes_dof_indices);
            temperature_cell->get_dof_indices(local_temperature_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    Assert (joint_fe.system_to_base_index(i).second < local_stokes_dof_indices.size(),
                            ExcInternalError());
                    distributed_joint_solution(local_joint_dof_indices[i])
                    = navier_stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                    Assert (joint_fe.system_to_base_index(i).first.first == 1,
                            ExcInternalError());
                    Assert (joint_fe.system_to_base_index(i).second < local_temperature_dof_indices.size(),
                            ExcInternalError());
                    distributed_joint_solution(local_joint_dof_indices[i])
                    = temperature_solution(local_temperature_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else
                    distributed_joint_solution(local_joint_dof_indices[i])
                    = (double)joint_cell->material_id();
        }
    }
    distributed_joint_solution.compress(VectorOperation::insert);

    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                            locally_relevant_joint_dofs);

    LA::Vector  joint_solution;
    joint_solution.reinit(locally_relevant_joint_dofs,
                          mpi_communicator);
    joint_solution = distributed_joint_solution;


    // create post processor
    PostProcessor<dim>   postprocessor(Utilities::MPI::this_mpi_process(mpi_communicator));

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    data_out.add_data_vector(joint_solution, postprocessor);
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  (initial_condition ? "initial":
                                   Utilities::int_to_string (timestep_number, 5)) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
            filenames.push_back(std::string("solution-") +
                                (initial_condition ? "initial":
                                Utilities::int_to_string (timestep_number, 5)) +
                                "." +
                                Utilities::int_to_string(i, 4) +
                                ".vtu");
        const std::string
        pvtu_master_filename = ("solution-" +
                                (initial_condition ? "initial":
                                Utilities::int_to_string (timestep_number, 5)) +
                                ".pvtu");
        std::ofstream pvtu_master(pvtu_master_filename.c_str());
        data_out.write_pvtu_record(pvtu_master, filenames);
        const std::string
        visit_master_filename = ("solution-" +
                                 (initial_condition ? "initial":
                                 Utilities::int_to_string (timestep_number, 5)) +
                                 ".visit");
        std::ofstream visit_master(visit_master_filename.c_str());
        DataOutBase::write_visit_record(visit_master, filenames);
    }
}
}  // namespace BuoyantFluid

// explicit instantiation
template std::pair<double,double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_rms_values() const;
template std::pair<double,double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_rms_values() const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_kinetic_energy() const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_kinetic_energy() const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_cfl_number() const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_cfl_number() const;

template void
BuoyantFluid::BuoyantFluidSolver<2>::output_results
(const bool) const;
template void
BuoyantFluid::BuoyantFluidSolver<3>::output_results
(const bool) const;
