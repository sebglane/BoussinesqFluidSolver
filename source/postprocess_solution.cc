/*
 * postprocess_solution.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include "buoyant_fluid_solver.h"
#include "postprocessor.h"

namespace BuoyantFluid {

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

    double rms_velocity = 0;
    double rms_temperature = 0;
    double volume = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = navier_stokes_dof_handler.begin_active(),
    temperature_cell = temperature_dof_handler.begin_active(),
    endc = navier_stokes_dof_handler.end();

    for (; cell != endc; ++cell, ++temperature_cell)
    {
        stokes_fe_values.reinit(cell);
        temperature_fe_values.reinit(temperature_cell);

        temperature_fe_values.get_function_values(temperature_solution,
                                                  temperature_values);
        stokes_fe_values[velocities].get_function_values(navier_stokes_solution,
                                                         velocity_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            rms_velocity += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
            rms_temperature += temperature_values[q] * temperature_values[q] * stokes_fe_values.JxW(q);
            volume += stokes_fe_values.JxW(q);
        }
    }

    rms_velocity /= volume;
    AssertIsFinite(rms_velocity);
    Assert(rms_velocity >= 0, ExcLowerRangeType<double>(rms_velocity, 0));

    rms_temperature /= volume;
    AssertIsFinite(rms_temperature);
    Assert(rms_temperature >= 0, ExcLowerRangeType<double>(rms_temperature, 0));

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
    return max_cfl * timestep;
}

template<int dim>
void BuoyantFluidSolver<dim>::output_results(const bool initial_condition) const
{
    if (parameters.verbose)
        std::cout << "   Output results..." << std::endl;

    // create joint finite element
    const FESystem<dim> joint_fe(navier_stokes_fe, 1,
                                 temperature_fe, 1);

    // create joint dof handler
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    Assert(joint_dof_handler.n_dofs() ==
           navier_stokes_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
           ExcInternalError());

    // create joint solution
    Vector<double>      joint_solution;
    joint_solution.reinit(joint_dof_handler.n_dofs());

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
        {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            stokes_cell->get_dof_indices(local_stokes_dof_indices);
            temperature_cell->get_dof_indices(local_temperature_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    Assert (joint_fe.system_to_base_index(i).second < local_stokes_dof_indices.size(),
                            ExcInternalError());
                    joint_solution(local_joint_dof_indices[i])
                    = navier_stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else
                {
                    Assert (joint_fe.system_to_base_index(i).first.first == 1,
                            ExcInternalError());
                    Assert (joint_fe.system_to_base_index(i).second < local_temperature_dof_indices.size(),
                            ExcInternalError());
                    joint_solution(local_joint_dof_indices[i])
                    = temperature_solution(local_temperature_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
        }
    }

    // create post processor
    PostProcessor<dim>   postprocessor;

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    data_out.add_data_vector(joint_solution, postprocessor);
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  (initial_condition ?
                                  "initial":
                                  Utilities::int_to_string(timestep_number, 5)) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}
}  // namespace BuoyantFluid

// explicit instantiation
template std::pair<double,double> BuoyantFluid::BuoyantFluidSolver<2>::compute_rms_values() const;
template std::pair<double,double> BuoyantFluid::BuoyantFluidSolver<3>::compute_rms_values() const;

template double BuoyantFluid::BuoyantFluidSolver<2>::compute_cfl_number() const;
template double BuoyantFluid::BuoyantFluidSolver<3>::compute_cfl_number() const;

template void BuoyantFluid::BuoyantFluidSolver<2>::output_results() const;
template void BuoyantFluid::BuoyantFluidSolver<3>::output_results() const;
