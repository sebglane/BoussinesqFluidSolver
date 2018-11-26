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

    // stokes part
    stokes_dof_handler.distribute_dofs(stokes_fe);

    DoFRenumbering::boost::king_ordering(stokes_dof_handler);

    std::vector<unsigned int> stokes_block_component(dim+1,0);
    stokes_block_component[dim] = 1;

    DoFRenumbering::component_wise(stokes_dof_handler, stokes_block_component);

    // IO
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(stokes_dof_handler,
                                   dofs_per_block,
                                   stokes_block_component);

    const unsigned int n_temperature_dofs = temperature_dof_handler.n_dofs();

    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "      Number of degrees of freedom: "
              << stokes_dof_handler.n_dofs()
              << std::endl
              << "      Number of velocity degrees of freedom: "
              << dofs_per_block[0]
              << std::endl
              << "      Number of pressure degrees of freedom: "
              << dofs_per_block[1]
              << std::endl
              << "      Number of temperature degrees of freedom: "
              << n_temperature_dofs
              << std::endl;

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
    setup_temperature_matrices(n_temperature_dofs);

    temperature_solution.reinit(n_temperature_dofs);
    old_temperature_solution.reinit(n_temperature_dofs);
    old_old_temperature_solution.reinit(n_temperature_dofs);

    temperature_rhs.reinit(n_temperature_dofs);

    // stokes constraints
    {
        stokes_constraints.clear();

        DoFTools::make_hanging_node_constraints(
                stokes_dof_handler,
                stokes_constraints);

        const Functions::ZeroFunction<dim> zero_function(dim+1);

        const std::map<typename types::boundary_id, const Function<dim>*>
        velocity_boundary_values = {{EquationData::BoundaryIds::ICB, &zero_function},
                                    {EquationData::BoundaryIds::CMB, &zero_function}};

        const FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(
                stokes_dof_handler,
                velocity_boundary_values,
                stokes_constraints,
                stokes_fe.component_mask(velocities));

        stokes_constraints.close();
    }
    // stokes constraints for pressure laplace matrix
    if (parameters.assemble_schur_complement)
    {
        stokes_laplace_constraints.clear();

        stokes_laplace_constraints.merge(stokes_constraints);

        // find pressure boundary dofs
        const FEValuesExtractors::Scalar pressure(dim);

        std::vector<bool> boundary_dofs(stokes_dof_handler.n_dofs(),
                                        false);
        DoFTools::extract_boundary_dofs(stokes_dof_handler,
                                        stokes_fe.component_mask(pressure),
                                        boundary_dofs);

        // find first unconstrained pressure boundary dof
        unsigned int first_boundary_dof = stokes_dof_handler.n_dofs();
        std::vector<bool>::const_iterator
        dof = boundary_dofs.begin(),
        end_dof = boundary_dofs.end();
        for (; dof != end_dof; ++dof)
            if (*dof)
                if (!stokes_laplace_constraints.is_constrained(dof - boundary_dofs.begin()))
                {
                    first_boundary_dof = dof - boundary_dofs.begin();
                    break;
                }
        Assert(first_boundary_dof < stokes_dof_handler.n_dofs(),
               ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        std::vector<bool>::const_iterator it = std::find(boundary_dofs.begin(),
                                                         boundary_dofs.end(),
                                                         true);
        Assert(first_boundary_dof >= it - boundary_dofs.begin(),
                ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        // set first pressure boundary dof to zero
        stokes_laplace_constraints.add_line(first_boundary_dof);

        stokes_laplace_constraints.close();
    }
    else
    {
        stokes_laplace_constraints.clear();
    }
    // stokes matrix and vector setup
    setup_stokes_matrix(dofs_per_block);
    stokes_solution.reinit(dofs_per_block);
    old_stokes_solution.reinit(dofs_per_block);
    old_old_stokes_solution.reinit(dofs_per_block);
    stokes_rhs.reinit(dofs_per_block);
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_temperature_matrices(const types::global_dof_index n_temperature_dofs)
{
    preconditioner_T.reset();

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
void BuoyantFluidSolver<dim>::setup_stokes_matrix(
        const std::vector<types::global_dof_index> dofs_per_block)
{
    preconditioner_A.reset();
    preconditioner_Kp.reset();
    preconditioner_Mp.reset();

    stokes_matrix.clear();
    stokes_laplace_matrix.clear();

    pressure_mass_matrix.clear();
    velocity_mass_matrix.clear();

    {
        TimerOutput::Scope timer_section(computing_timer, "setup stokes matrix");

        Table<2,DoFTools::Coupling> stokes_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d || c<dim || d<dim)
                    stokes_coupling[c][d] = DoFTools::always;
                else
                    stokes_coupling[c][d] = DoFTools::none;

        BlockDynamicSparsityPattern dsp(dofs_per_block,
                                        dofs_per_block);

        DoFTools::make_sparsity_pattern(
                stokes_dof_handler,
                stokes_coupling,
                dsp,
                stokes_constraints);

        stokes_sparsity_pattern.copy_from(dsp);

        stokes_matrix.reinit(stokes_sparsity_pattern);
    }
    {
        TimerOutput::Scope timer_section(computing_timer, "setup stokes laplace matrix");

        Table<2,DoFTools::Coupling> stokes_laplace_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c==dim && d==dim)
                    stokes_laplace_coupling[c][d] = DoFTools::always;
                else
                    stokes_laplace_coupling[c][d] = DoFTools::none;

        BlockDynamicSparsityPattern laplace_dsp(dofs_per_block, dofs_per_block);

        const ConstraintMatrix &constraints_used =
                (parameters.assemble_schur_complement?
                                stokes_laplace_constraints: stokes_constraints);

        DoFTools::make_sparsity_pattern(stokes_dof_handler,
                                        stokes_laplace_coupling,
                                        laplace_dsp,
                                        constraints_used);

        if (!parameters.assemble_schur_complement)
            laplace_dsp.block(1,1)
            .compute_mmult_pattern(stokes_sparsity_pattern.block(1,0),
                                   stokes_sparsity_pattern.block(0,1));


        auxiliary_stokes_sparsity_pattern.copy_from(laplace_dsp);

        stokes_laplace_matrix.reinit(auxiliary_stokes_sparsity_pattern);

        stokes_laplace_matrix.block(0,0).reinit(stokes_sparsity_pattern.block(0,0));
    }
}

}  // namespace BuoyantFluid


// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::setup_dofs();
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_dofs();

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_temperature_matrices(const unsigned int n_temperature_dofs);
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_temperature_matrices(const unsigned int n_temperature_dofs);

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_stokes_matrix(const std::vector<types::global_dof_index> dofs_per_block);
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_stokes_matrix(const std::vector<types::global_dof_index> dofs_per_block);
