/*
 * setup.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"
#include "grid_factory.h"

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::setup_dofs()
{
    std::cout << "Setup dofs..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    // temperature part
    temperature_dof_handler.distribute_dofs(temperature_fe);

    DoFRenumbering::boost::king_ordering(temperature_dof_handler);

    // temperature constraints
    {
        temperature_constraints.clear();

        DoFTools::make_hanging_node_constraints(
                temperature_dof_handler,
                temperature_constraints);

        const Functions::ConstantFunction<dim> icb_temperature(0.5);
        const Functions::ConstantFunction<dim> cmb_temperature(-0.5);

        const std::map<typename types::boundary_id, const Function<dim>*>
        temperature_boundary_values = {{GridFactory::BoundaryIds::ICB, &icb_temperature},
                                       {GridFactory::BoundaryIds::CMB, &cmb_temperature}};

        VectorTools::interpolate_boundary_values(temperature_dof_handler,
                                                 temperature_boundary_values,
                                                 temperature_constraints);

        temperature_constraints.close();
    }
    // temperature matrix and vector setup
    const unsigned int n_dofs_temperature
    = temperature_dof_handler.n_dofs();

    setup_temperature_matrix(n_dofs_temperature);

    temperature_solution.reinit(n_dofs_temperature);
    old_temperature_solution.reinit(n_dofs_temperature);
    old_old_temperature_solution.reinit(n_dofs_temperature);
    temperature_rhs.reinit(n_dofs_temperature);

    // velocity part
    velocity_dof_handler.distribute_dofs(velocity_fe);

    DoFRenumbering::boost::king_ordering(velocity_dof_handler);
    DoFRenumbering::component_wise(velocity_dof_handler);
    // velocity constraints
    {
        velocity_constraints.clear();

        DoFTools::make_hanging_node_constraints(velocity_dof_handler,
                                                velocity_constraints);

        const Functions::ZeroFunction<dim> zero_function(dim);

        const std::map<typename types::boundary_id, const Function<dim>*>
        velocity_boundary_values = {{GridFactory::BoundaryIds::ICB, &zero_function},
                                    {GridFactory::BoundaryIds::CMB, &zero_function}};

        const FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(velocity_dof_handler,
                                                 velocity_boundary_values,
                                                 velocity_constraints);

        velocity_constraints.close();
    }

    // velocity matrix setup
    const types::global_dof_index   n_dofs_velocity
    = velocity_dof_handler.n_dofs();
    setup_velocity_system(n_dofs_velocity);

    // reinit velocity vectors
    velocity_solution.reinit(n_dofs_velocity);
    old_velocity_solution.reinit(n_dofs_velocity);
    old_old_velocity_solution.reinit(n_dofs_velocity);
    velocity_rhs.reinit(n_dofs_velocity);

    // pressure part
    pressure_dof_handler.distribute_dofs(pressure_fe);

    DoFRenumbering::boost::king_ordering(pressure_dof_handler);
    // pressure constraints
    {
        pressure_constraints.clear();

        DoFTools::make_hanging_node_constraints(pressure_dof_handler,
                                                pressure_constraints);

        // find pressure boundary dofs
        std::vector<bool> boundary_dofs(pressure_dof_handler.n_dofs(),
                                        false);
        DoFTools::extract_boundary_dofs(pressure_dof_handler,
                                        ComponentMask(),
                                        boundary_dofs);

        // find first unconstrained pressure boundary dof
        unsigned int first_boundary_dof = pressure_dof_handler.n_dofs();
        std::vector<bool>::const_iterator
        dof = boundary_dofs.begin(),
        end_dof = boundary_dofs.end();
        for (; dof != end_dof; ++dof)
            if (*dof)
                if (!pressure_constraints.is_constrained(dof - boundary_dofs.begin()))
                {
                    first_boundary_dof = dof - boundary_dofs.begin();
                    break;
                }
        Assert(first_boundary_dof < pressure_dof_handler.n_dofs(),
               ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        std::vector<bool>::const_iterator it = std::find(boundary_dofs.begin(),
                                                         boundary_dofs.end(),
                                                         true);
        Assert(first_boundary_dof >= it - boundary_dofs.begin(),
               ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        // set first pressure boundary dof to zero
        pressure_constraints.add_line(first_boundary_dof);

        pressure_constraints.close();
    }
    // velocity matrix setup
    const types::global_dof_index   n_dofs_pressure
    = pressure_dof_handler.n_dofs();
    setup_pressure_system(n_dofs_pressure);

    // reinit pressure vectors
    pressure_solution.reinit(n_dofs_pressure);
    old_pressure_solution.reinit(n_dofs_pressure);
    pressure_rhs.reinit(n_dofs_pressure);
    phi_solution.reinit(n_dofs_pressure);
    old_phi_solution.reinit(n_dofs_pressure);
    old_old_phi_solution.reinit(n_dofs_pressure);

    // print info message
    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << n_dofs_temperature + n_dofs_velocity + n_dofs_pressure
              << std::endl
              << "   Number of velocity degrees of freedom: "
              << n_dofs_velocity
              << std::endl
              << "   Number of pressure degrees of freedom: "
              << n_dofs_pressure
              << std::endl
              << "   Number of temperature degrees of freedom: "
              << n_dofs_temperature
              << std::endl;
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_temperature_matrix(const types::global_dof_index n_dofs)
{
    preconditioner_temperature.reset();

    temperature_matrix.clear();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();

    DynamicSparsityPattern dsp(n_dofs);

    DoFTools::make_sparsity_pattern(temperature_dof_handler,
                                    dsp,
                                    temperature_constraints);

    temperature_sparsity_pattern.copy_from(dsp);

    temperature_matrix.reinit(temperature_sparsity_pattern);
    temperature_mass_matrix.reinit(temperature_sparsity_pattern);
    temperature_stiffness_matrix.reinit(temperature_sparsity_pattern);

    rebuild_temperature_matrices = true;
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_velocity_system(
        const types::global_dof_index n_dofs)
{
    preconditioner_diffusion.reset();

    velocity_matrix.clear();
    velocity_mass_matrix.clear();
    velocity_laplace_matrix.clear();

    Table<2,DoFTools::Coupling> velocity_coupling(dim, dim);
    for (unsigned int c=0; c<dim; ++c)
        for (unsigned int d=0; d<dim; ++d)
            if (c==d)
                velocity_coupling[c][d] = DoFTools::always;
            else
                velocity_coupling[c][d] = DoFTools::none;

    DynamicSparsityPattern  dsp(n_dofs);

    DoFTools::make_sparsity_pattern(velocity_dof_handler,
                                    velocity_coupling,
                                    dsp,
                                    velocity_constraints);

    velocity_sparsity_pattern.copy_from(dsp);

    velocity_matrix.reinit(velocity_sparsity_pattern);
    velocity_mass_matrix.reinit(velocity_sparsity_pattern);
    velocity_laplace_matrix.reinit(velocity_sparsity_pattern);

    rebuild_velocity_matrices = true;
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_pressure_system(const types::global_dof_index n_dofs)
{
    preconditioner_projection.reset();
    preconditioner_pressure_mass.reset();

    pressure_laplace_matrix.clear();
    pressure_mass_matrix.clear();

    DynamicSparsityPattern  dsp(n_dofs);

    DoFTools::make_sparsity_pattern(pressure_dof_handler,
                                    dsp,
                                    pressure_constraints);

    pressure_sparsity_pattern.copy_from(dsp);

    pressure_laplace_matrix.reinit(pressure_sparsity_pattern);
    pressure_mass_matrix.reinit(pressure_sparsity_pattern);

    rebuild_pressure_matrices = true;
}

}  // namespace BuoyantFluid


// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::setup_dofs();
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_dofs();

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_temperature_matrix(const types::global_dof_index );
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_temperature_matrix(const types::global_dof_index );

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_velocity_system(const types::global_dof_index );
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_velocity_system(const types::global_dof_index );

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_pressure_system(const types::global_dof_index );
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_pressure_system(const types::global_dof_index );
