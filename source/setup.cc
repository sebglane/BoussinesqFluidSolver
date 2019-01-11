/*
 * setup.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::setup_dofs()
{
    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

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
        temperature_boundary_values = {{EquationData::BoundaryIds::ICB, &icb_temperature},
                                       {EquationData::BoundaryIds::CMB, &cmb_temperature}};

        VectorTools::interpolate_boundary_values(
                temperature_dof_handler,
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

    // stokes part
    navier_stokes_dof_handler.distribute_dofs(navier_stokes_fe);

    DoFRenumbering::boost::king_ordering(navier_stokes_dof_handler);

    std::vector<unsigned int> stokes_block_component(dim+1,0);
    stokes_block_component[dim] = 1;
    DoFRenumbering::component_wise(navier_stokes_dof_handler,
                                   stokes_block_component);
    // stokes constraints
    {
        navier_stokes_constraints.clear();

        DoFTools::make_hanging_node_constraints(navier_stokes_dof_handler,
                                                navier_stokes_constraints);

        const Functions::ZeroFunction<dim> zero_function(dim+1);

        const std::map<typename types::boundary_id, const Function<dim>*>
        velocity_boundary_values = {{EquationData::BoundaryIds::ICB, &zero_function},
                                    {EquationData::BoundaryIds::CMB, &zero_function}};

        const FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(
                navier_stokes_dof_handler,
                velocity_boundary_values,
                navier_stokes_constraints,
                navier_stokes_fe.component_mask(velocities));

        // find pressure boundary dofs
        const FEValuesExtractors::Scalar pressure(dim);

        std::vector<bool> boundary_dofs(navier_stokes_dof_handler.n_dofs(),
                                        false);
        DoFTools::extract_boundary_dofs(navier_stokes_dof_handler,
                                        navier_stokes_fe.component_mask(pressure),
                                        boundary_dofs);

        // find first unconstrained pressure boundary dof
        unsigned int first_boundary_dof = navier_stokes_dof_handler.n_dofs();
        std::vector<bool>::const_iterator
        dof = boundary_dofs.begin(),
        end_dof = boundary_dofs.end();
        for (; dof != end_dof; ++dof)
            if (*dof)
                if (!navier_stokes_constraints.is_constrained(dof - boundary_dofs.begin()))
                {
                    first_boundary_dof = dof - boundary_dofs.begin();
                    break;
                }
        Assert(first_boundary_dof < navier_stokes_dof_handler.n_dofs(),
               ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        std::vector<bool>::const_iterator it = std::find(boundary_dofs.begin(),
                                                         boundary_dofs.end(),
                                                         true);
        Assert(first_boundary_dof >= it - boundary_dofs.begin(),
                ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        // set first pressure boundary dof to zero
        navier_stokes_constraints.add_line(first_boundary_dof);

        navier_stokes_constraints.close();
    }

    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(navier_stokes_dof_handler,
                                   dofs_per_block,
                                   stokes_block_component);

    // stokes matrix and vector setup
    setup_navier_stokes_system(dofs_per_block);

    // reinit block vectors
    navier_stokes_solution.reinit(dofs_per_block);
    old_navier_stokes_solution.reinit(dofs_per_block);
    old_old_navier_stokes_solution.reinit(dofs_per_block);
    navier_stokes_rhs.reinit(dofs_per_block);

    // reinit pressure vectors
    phi_pressure.reinit(dofs_per_block[1]);
    old_phi_pressure.reinit(dofs_per_block[1]);
    old_old_phi_pressure.reinit(dofs_per_block[1]);

    // print info message
    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "      Number of degrees of freedom: "
              << navier_stokes_dof_handler.n_dofs() + n_dofs_temperature
              << std::endl
              << "      Number of velocity degrees of freedom: "
              << dofs_per_block[0]
              << std::endl
              << "      Number of pressure degrees of freedom: "
              << dofs_per_block[1]
              << std::endl
              << "      Number of temperature degrees of freedom: "
              << n_dofs_temperature
              << std::endl;

}

template<int dim>
void BuoyantFluidSolver<dim>::setup_temperature_matrix(const types::global_dof_index n_temperature_dofs)
{
    preconditioner_temperature.reset();

    temperature_matrix.clear();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();

    DynamicSparsityPattern dsp(n_temperature_dofs, n_temperature_dofs);

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
void BuoyantFluidSolver<dim>::setup_navier_stokes_system(
        const std::vector<types::global_dof_index> dofs_per_block)
{

    preconditioner_diffusion.reset();

    navier_stokes_matrix.clear();
    velocity_mass_matrix.clear();
    velocity_laplace_matrix.clear();

    Table<2,DoFTools::Coupling> stokes_coupling(dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
        for (unsigned int d=0; d<dim+1; ++d)
            if ((c<dim || d<dim))
                if (parameters.rotation)
                    stokes_coupling[c][d] = DoFTools::always;
                else if (c==d)
                    stokes_coupling[c][d] = DoFTools::always;
            else
                stokes_coupling[c][d] = DoFTools::none;

    BlockDynamicSparsityPattern dsp(dofs_per_block,
                                    dofs_per_block);

    DoFTools::make_sparsity_pattern(
            navier_stokes_dof_handler,
            stokes_coupling,
            dsp,
            navier_stokes_constraints);

    navier_stokes_sparsity_pattern.copy_from(dsp);

    navier_stokes_matrix.reinit(navier_stokes_sparsity_pattern);

    velocity_mass_matrix.reinit(navier_stokes_sparsity_pattern.block(0,0));
    velocity_laplace_matrix.reinit(navier_stokes_sparsity_pattern.block(0,0));

    rebuild_navier_stokes_matrices = true;
}

}  // namespace BuoyantFluid


// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::setup_dofs();
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_dofs();

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_temperature_matrix(const unsigned int );
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_temperature_matrix(const unsigned int );

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_navier_stokes_system(const std::vector<types::global_dof_index> );
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_navier_stokes_system(const std::vector<types::global_dof_index> );
