/*
 * postprocess_solution.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

// Sadly this include file is not found
// #include <boost/filesystem.hpp>

#include <adsolic/buoyant_fluid_solver.h>
#include <adsolic/postprocessor.h>

namespace BuoyantFluid {

template<int dim>
std::vector<double> BuoyantFluidSolver<dim>::compute_global_averages() const
{
    const QGauss<dim> velocity_quadrature(parameters.velocity_degree + 1);

    const QGauss<dim> temperature_quadrature(parameters.temperature_degree + 1);

    const unsigned int n_velocity_q_points = velocity_quadrature.size();
    const unsigned int n_temperature_q_points = temperature_quadrature.size();

    FEValues<dim> stokes_fe_values(mapping,
                                   navier_stokes_fe,
                                   velocity_quadrature,
                                   update_values|
                                   update_JxW_values);

    FEValues<dim> temperature_fe_values(mapping,
                                        temperature_fe,
                                        temperature_quadrature,
                                        update_values|
                                        update_JxW_values);

    std::vector<double>         temperature_values(n_temperature_q_points);
    std::vector<Tensor<1,dim>>  velocity_values(n_velocity_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double local_sum_velocity_sqrd = 0;
    double local_sum_temperature = 0;
    double local_navier_stokes_volume = 0;
    double local_temperature_volume = 0;

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

        for (unsigned int q=0; q<n_velocity_q_points; ++q)
        {
            local_sum_velocity_sqrd += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
            local_navier_stokes_volume += stokes_fe_values.JxW(q);
        }
        for (unsigned int q=0; q<n_temperature_q_points; ++q)
        {
            local_sum_temperature += temperature_values[q] * temperature_values[q] * temperature_fe_values.JxW(q);
            local_temperature_volume += temperature_fe_values.JxW(q);
        }
    }

    AssertIsFinite(local_sum_velocity_sqrd);
    AssertIsFinite(local_sum_temperature);
    AssertIsFinite(local_navier_stokes_volume);
    AssertIsFinite(local_temperature_volume);

    Assert(local_sum_velocity_sqrd >= 0, ExcLowerRangeType<double>(local_sum_velocity_sqrd, 0));
    Assert(local_sum_temperature >= 0, ExcLowerRangeType<double>(local_sum_temperature, 0));
    Assert(local_navier_stokes_volume >= 0, ExcLowerRangeType<double>(local_navier_stokes_volume, 0));
    Assert(local_temperature_volume >= 0, ExcLowerRangeType<double>(local_temperature_volume, 0));

    const double local_sums[4]  = { local_sum_velocity_sqrd,
                                    local_sum_temperature,
                                    local_navier_stokes_volume,
                                    local_temperature_volume};
    double global_sums[4];

    Utilities::MPI::sum(local_sums, mpi_communicator, global_sums);

    const double rms_velocity = std::sqrt(global_sums[0] / global_sums[2]);
    const double rms_kinetic_energy = 0.5 * global_sums[0] / global_sums[2];
    const double rms_temperature = std::sqrt(global_sums[1] / global_sums[3]);

    return std::vector<double>{rms_velocity, rms_kinetic_energy, rms_temperature};
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

    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    const double largest_viscosity = std::max(equation_coefficients[1],
                                              equation_coefficients[3]);

    double max_cfl = 0;
    for (auto cell: navier_stokes_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit (cell);
            fe_values[velocities].get_function_values(navier_stokes_solution,
                                                      velocity_values);

            double  max_cell_velocity = 0;
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                max_cell_velocity = std::max(max_cell_velocity,
                                             velocity_values[q].norm());
            }
            max_cfl = std::max(max_cfl,
                               max_cell_velocity / (cell->diameter() * std::sqrt(dim)));
            max_cfl = std::max(max_cfl,
                               largest_viscosity / (cell->diameter() * cell->diameter() * dim));
        }

    const double max_polynomial_degree
    = double(std::max(parameters.temperature_degree,
                      parameters.velocity_degree));

    const double local_cfl = max_cfl * timestep / max_polynomial_degree;

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
    const FESystem<dim> joint_fe(navier_stokes_fe, 1,
                                 temperature_fe, 1);

    // create joint dof handler
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    Assert(joint_dof_handler.n_dofs() ==
           navier_stokes_dof_handler.n_dofs() +
           temperature_dof_handler.n_dofs(),
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

    /*
     *
    // create results directory
    {
        // verify that the output directory actually exists. if it doesn't, create
        // it on processor zero
        bool success;

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            using namespace boost::filesystem;

            const path  results_path("results");

            if (!is_directory(results_path))
            {
                pcout << "----------------------------------------------------"
                      << std::endl
                      << "The output directory appears not to exist\n"
                      << "and will be created for you.\n"
                      << "----------------------------------------------------"
                      << std::endl;

                success = create_directory(results_path);

            }
            else
            {
                success = true;
            }
            // Broadcast error code
            MPI_Bcast(&success, 1, MPIU_BOOL, 0, mpi_communicator);
            AssertThrow(success == false,
                        ExcMessage(std::string("Can't create the output directory.")));
        }
        else
        {
            // Wait to receive error code, and throw ExcInternalError if directory
            // creation has failed
            MPI_Bcast (&success, 1, MPIU_BOOL, 0, mpi_communicator);
            if (success == false)
                throw ExcInternalError();
        }
    }
     *
     */

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
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_global_averages() const;
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_global_averages() const;

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
