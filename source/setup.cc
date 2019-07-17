/*
 * setup.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"
#include "grid_factory.h"

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::setup_dofs()
{
    pcout << "Setup dofs..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    const types::global_dof_index n_dofs_temperature = setup_temperature_dofs();

    const std::pair<types::global_dof_index,types::global_dof_index>
    n_navier_stokes_dofs = setup_navier_stokes_dofs();

    std::pair<types::global_dof_index,types::global_dof_index>
    n_magnetic_dofs;
    if (parameters.magnetism)
        n_magnetic_dofs = setup_magnetic_dofs();

    // print info message
    pcout << "   Number of active cells: "
          << triangulation.n_global_active_cells()
          << std::endl
          << "   Number of degrees of freedom: "
          << navier_stokes_dof_handler.n_dofs() + n_dofs_temperature +
             (parameters.magnetism ? magnetic_dof_handler.n_dofs() : 0)
          << std::endl
          << "   Number of velocity degrees of freedom: "
          << n_navier_stokes_dofs.first
          << std::endl
          << "   Number of pressure degrees of freedom: "
          << n_navier_stokes_dofs.second
          << std::endl
          << "   Number of temperature degrees of freedom: "
          << n_dofs_temperature
          << std::endl;
    if (parameters.magnetism)
    {
        pcout << "   Number of magnetic degrees of freedom: "
              << n_magnetic_dofs.first
              << std::endl
              << "   Number of pseudo pressure degrees of freedom: "
              << n_magnetic_dofs.second
              << std::endl;
    }
}

template<int dim>
types::global_dof_index BuoyantFluidSolver<dim>::setup_temperature_dofs()
{
    if (parameters.verbose)
        pcout << "   Setup temperature dofs..." << std::endl << std::flush;

    locally_owned_temperature_dofs.clear();
    locally_relevant_temperature_dofs.clear();
    temperature_dof_handler.distribute_dofs(temperature_fe);

    const types::global_dof_index n_dofs_temperature
    = temperature_dof_handler.n_dofs();

    DoFRenumbering::Cuthill_McKee(temperature_dof_handler);

    // extract locally owned and relevant dofs
    locally_owned_temperature_dofs = temperature_dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs(temperature_dof_handler,
                                            locally_relevant_temperature_dofs);

    // temperature constraints
    {
        temperature_constraints.clear();
        temperature_constraints.reinit(locally_relevant_temperature_dofs);

        DoFTools::make_hanging_node_constraints(
                temperature_dof_handler,
                temperature_constraints);

        if (parameters.geometry == GeometryType::SphericalShell)
        {
            const Functions::ConstantFunction<dim>  icb_temperature(1.0);
            const Functions::ConstantFunction<dim>  cmb_temperature(0.0);

            const std::map<typename types::boundary_id, const Function<dim>*>
            temperature_boundary_values = {{GridFactory::BoundaryIds::ICB, &icb_temperature},
                                           {GridFactory::BoundaryIds::CMB, &cmb_temperature}};

            VectorTools::interpolate_boundary_values(temperature_dof_handler,
                                                     temperature_boundary_values,
                                                     temperature_constraints);
        }
        else if (parameters.geometry == GeometryType::Cavity)
        {
            const Functions::ConstantFunction<dim>  left_temperature(0.5);
            const Functions::ConstantFunction<dim>  right_temperature(-0.5);

            const std::map<typename types::boundary_id, const Function<dim>*>
            temperature_boundary_values = {{GridFactory::BoundaryIds::Left, &left_temperature},
                                           {GridFactory::BoundaryIds::Right, &right_temperature}};

            VectorTools::interpolate_boundary_values(temperature_dof_handler,
                                                     temperature_boundary_values,
                                                     temperature_constraints);
        }

        temperature_constraints.close();
    }
    // temperature matrix and vector setup
    setup_temperature_matrix(locally_owned_temperature_dofs,
                             locally_relevant_temperature_dofs);

    temperature_solution.reinit(locally_relevant_temperature_dofs,
                                mpi_communicator);
    old_temperature_solution.reinit(temperature_solution);
    old_old_temperature_solution.reinit(temperature_solution);

    temperature_rhs.reinit(locally_owned_temperature_dofs,
                           locally_relevant_temperature_dofs,
                           mpi_communicator,
                           true);

    return n_dofs_temperature;
}


template<int dim>
void BuoyantFluidSolver<dim>::setup_temperature_matrix
(const IndexSet &locally_owned_dofs,
 const IndexSet &locally_relevant_dofs)
{
    preconditioner_temperature.reset();

    temperature_matrix.clear();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();

    LA::DynamicSparsityPattern  dsp(locally_owned_dofs,
                                    locally_owned_dofs,
                                    locally_relevant_dofs,
                                    mpi_communicator);

    DoFTools::make_sparsity_pattern(temperature_dof_handler,
                                    dsp,
                                    temperature_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));

    dsp.compress();

    temperature_matrix.reinit(dsp);
    temperature_mass_matrix.reinit(dsp);
    temperature_stiffness_matrix.reinit(dsp);

    rebuild_temperature_matrices = true;
}


template<int dim>
std::pair<types::global_dof_index,types::global_dof_index>
BuoyantFluidSolver<dim>::setup_navier_stokes_dofs()
{
    if (parameters.verbose)
        pcout << "   Setup Navier-Stokes dofs..." << std::endl << std::flush;

    locally_owned_stokes_dofs.clear();
    locally_relevant_stokes_dofs.clear();
    navier_stokes_dof_handler.distribute_dofs(navier_stokes_fe);

    DoFRenumbering::Cuthill_McKee(navier_stokes_dof_handler);
    DoFRenumbering::block_wise(navier_stokes_dof_handler);

    // count stokes dofs
    std::vector<unsigned int> stokes_block_component(2,0);
    stokes_block_component[1] = 1;
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(navier_stokes_dof_handler,
                                   dofs_per_block,
                                   stokes_block_component);

    const types::global_dof_index
    n_velocity_dofs = dofs_per_block[0],
    n_pressure_dofs = dofs_per_block[1];

    // extract locally owned and relevant dofs
    IndexSet    locally_relevant_stokes_set;
    {
        const IndexSet locally_owned_stokes_set
        = navier_stokes_dof_handler.locally_owned_dofs();

        locally_owned_stokes_dofs.push_back(
                locally_owned_stokes_set.get_view(0, n_velocity_dofs));
        locally_owned_stokes_dofs.push_back(
                locally_owned_stokes_set.get_view(n_velocity_dofs,
                                                  n_velocity_dofs + n_pressure_dofs));

        DoFTools::extract_locally_relevant_dofs(navier_stokes_dof_handler,
                                                locally_relevant_stokes_set);

        locally_relevant_stokes_dofs.push_back(
                locally_relevant_stokes_set.get_view(0, n_velocity_dofs));
        locally_relevant_stokes_dofs.push_back(
                locally_relevant_stokes_set.get_view(n_velocity_dofs,
                                                     n_velocity_dofs + n_pressure_dofs));
    }
    // stokes constraints
    {
        navier_stokes_constraints.clear();
        navier_stokes_constraints.reinit(locally_relevant_stokes_set);

        DoFTools::make_hanging_node_constraints(navier_stokes_dof_handler,
                                                navier_stokes_constraints);

        const Functions::ZeroFunction<dim> zero_function(dim+1);

        if (parameters.geometry == GeometryType::SphericalShell)
        {
            const std::map<typename types::boundary_id, const Function<dim>*>
            velocity_boundary_values = {{GridFactory::BoundaryIds::ICB, &zero_function},
                                        {GridFactory::BoundaryIds::CMB, &zero_function}};

            const FEValuesExtractors::Vector velocities(0);
            VectorTools::interpolate_boundary_values(
                    navier_stokes_dof_handler,
                    velocity_boundary_values,
                    navier_stokes_constraints,
                    navier_stokes_fe.component_mask(velocities));
        }
        else if (parameters.geometry == GeometryType::Cavity)
        {
            std::map<typename types::boundary_id, const Function<dim>*>
            velocity_boundary_values = {{GridFactory::BoundaryIds::Left, &zero_function},
                                        {GridFactory::BoundaryIds::Right, &zero_function},
                                        {GridFactory::BoundaryIds::Top, &zero_function},
                                        {GridFactory::BoundaryIds::Bottom, &zero_function}};
            if (dim == 3)
            {
                velocity_boundary_values.insert
                (std::pair<typename types::boundary_id, const Function<dim>*>(GridFactory::BoundaryIds::Front,&zero_function));
                velocity_boundary_values.insert
                (std::pair<typename types::boundary_id, const Function<dim>*>(GridFactory::BoundaryIds::Back,&zero_function));
            }

            const FEValuesExtractors::Vector velocities(0);
            VectorTools::interpolate_boundary_values(
                    navier_stokes_dof_handler,
                    velocity_boundary_values,
                    navier_stokes_constraints,
                    navier_stokes_fe.component_mask(velocities));
        }

        navier_stokes_constraints.close();
    }
    // add pressure constraints
    {
        stokes_pressure_constraints.clear();
        stokes_pressure_constraints.reinit(locally_relevant_stokes_set);

        stokes_pressure_constraints.merge(navier_stokes_constraints);

        const FEValuesExtractors::Scalar    pressure(dim);

        IndexSet    boundary_pressure_dofs;

        DoFTools::extract_boundary_dofs(navier_stokes_dof_handler,
                                        navier_stokes_fe.component_mask(pressure),
                                        boundary_pressure_dofs);

        types::global_dof_index local_idx = numbers::invalid_dof_index;

        IndexSet::ElementIterator
        idx = boundary_pressure_dofs.begin(),
        endidx = boundary_pressure_dofs.end();
        for(; idx != endidx; ++idx)
            if (stokes_pressure_constraints.can_store_line(*idx)
                    && !stokes_pressure_constraints.is_constrained(*idx))
                local_idx = *idx;

        // Make a reduction to find the smallest index (processors that
        // found a larger candidate just happened to not be able to store
        // that index with the minimum value). Note that it is possible that
        // some processors might not be able to find a potential DoF, for
        // example because they don't own any DoFs. On those processors we
        // will use dof_handler.n_dofs() when building the minimum (larger
        // than any valid DoF index).
        const types::global_dof_index global_idx
        = Utilities::MPI::min(
                (local_idx != numbers::invalid_dof_index) ?
                        local_idx :
                        navier_stokes_dof_handler.n_dofs(),
                mpi_communicator);

        Assert(global_idx < navier_stokes_dof_handler.n_dofs(),
               ExcMessage("Error, couldn't find a pressure DoF to constrain."));

        // Finally set this DoF to zero (if we care about it):
        if (stokes_pressure_constraints.can_store_line(global_idx))
        {
            Assert(!stokes_pressure_constraints.is_constrained(global_idx), ExcInternalError());
            stokes_pressure_constraints.add_line(global_idx);
        }

        stokes_pressure_constraints.close();
    }

    // stokes matrix and vector setup
    setup_navier_stokes_system(locally_owned_stokes_dofs,
                               locally_relevant_stokes_dofs);

    // reinit block vectors
    navier_stokes_solution.reinit(locally_relevant_stokes_dofs,
                                  mpi_communicator);
    old_navier_stokes_solution.reinit(navier_stokes_solution);
    old_old_navier_stokes_solution.reinit(navier_stokes_solution);

    navier_stokes_rhs.reinit(locally_owned_stokes_dofs,
                             locally_relevant_stokes_dofs,
                             mpi_communicator,
                             true);

    // reinit pressure vectors
    phi_pressure.reinit(navier_stokes_solution);
    old_phi_pressure.reinit(navier_stokes_solution);
    old_old_phi_pressure.reinit(navier_stokes_solution);

    return std::pair<types::global_dof_index,types::global_dof_index>
    (n_velocity_dofs,n_pressure_dofs);
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_navier_stokes_system
(const std::vector<IndexSet>    &locally_owned_dofs,
 const std::vector<IndexSet>    &locally_relevant_dofs)
{
    preconditioner_symmetric_diffusion.reset();
    preconditioner_asymmetric_diffusion.reset();
    preconditioner_projection.reset();
    preconditioner_pressure_mass.reset();

    navier_stokes_matrix.clear();
    navier_stokes_mass_matrix.clear();
    navier_stokes_laplace_matrix.clear();

    // sparsity pattern for stokes matrix
    LA::BlockDynamicSparsityPattern dsp(locally_owned_dofs,
                                        locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);
    {
        Table<2,DoFTools::Coupling> stokes_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c<dim)
                    stokes_coupling[c][d] = DoFTools::always;
                else if ((c<dim && d==dim) || (c==dim && d<dim))
                    stokes_coupling[c][d] = DoFTools::always;
                else
                    stokes_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                navier_stokes_dof_handler,
                stokes_coupling,
                dsp,
                navier_stokes_constraints,
                false,
                Utilities::MPI::this_mpi_process(mpi_communicator));

        dsp.compress();
    }
    navier_stokes_matrix.reinit(dsp);

    // sparsity pattern for laplace matrix
    LA::BlockDynamicSparsityPattern laplace_dsp(locally_owned_dofs,
                                                locally_owned_dofs,
                                                locally_relevant_dofs,
                                                mpi_communicator);
    // sparsity pattern for mass matrix
    LA::BlockDynamicSparsityPattern mass_dsp(locally_owned_dofs,
                                             locally_owned_dofs,
                                             locally_relevant_dofs,
                                             mpi_communicator);
    {
        // auxiliary coupling structure for pressure part
        Table<2,DoFTools::Coupling> pressure_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c==dim)
                    pressure_coupling[c][d] = DoFTools::always;
                else
                    pressure_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                navier_stokes_dof_handler,
                pressure_coupling,
                laplace_dsp,
                stokes_pressure_constraints,
                false,
                Utilities::MPI::this_mpi_process(mpi_communicator));
        laplace_dsp.compress();

        DoFTools::make_sparsity_pattern(
                navier_stokes_dof_handler,
                pressure_coupling,
                mass_dsp,
                navier_stokes_constraints);
        mass_dsp.compress();
    }
    navier_stokes_laplace_matrix.reinit(laplace_dsp);
    navier_stokes_laplace_matrix.block(0,0).reinit(dsp.block(0,0));

    navier_stokes_mass_matrix.reinit(mass_dsp);
    navier_stokes_mass_matrix.block(0,0).reinit(dsp.block(0,0));

    rebuild_navier_stokes_matrices = true;
}

template<int dim>
std::pair<types::global_dof_index,types::global_dof_index>
BuoyantFluidSolver<dim>::setup_magnetic_dofs()
{
    if (parameters.verbose)
        pcout << "   Setup magnetic dofs..." << std::endl << std::flush;

    locally_owned_magnetic_dofs.clear();
    locally_relevant_magnetic_dofs.clear();
    magnetic_dof_handler.distribute_dofs(magnetic_fe);

    DoFRenumbering::Cuthill_McKee(magnetic_dof_handler);
    DoFRenumbering::block_wise(magnetic_dof_handler);

    // count magnetic dofs
    std::vector<unsigned int> magnetic_block_component(2,0);
    magnetic_block_component[1] = 1;
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(magnetic_dof_handler,
                                   dofs_per_block,
                                   magnetic_block_component);

    const types::global_dof_index   n_magnetic_dofs = dofs_per_block[0],
                                    n_pseudo_pressure_dofs = dofs_per_block[1];

    // extract locally owned and relevant dofs
    IndexSet    locally_relevant_magnetic_set;
    {
        const IndexSet locally_owned_magnetic_set
        = magnetic_dof_handler.locally_owned_dofs();

        locally_owned_magnetic_dofs.push_back(
                locally_owned_magnetic_set.get_view(0, n_magnetic_dofs));
        locally_owned_magnetic_dofs.push_back(
                locally_owned_magnetic_set.get_view(n_magnetic_dofs,
                                                    n_magnetic_dofs + n_pseudo_pressure_dofs));

        DoFTools::extract_locally_relevant_dofs(magnetic_dof_handler,
                                                locally_relevant_magnetic_set);

        locally_relevant_magnetic_dofs.push_back(
                locally_relevant_magnetic_set.get_view(0, n_magnetic_dofs));
        locally_relevant_magnetic_dofs.push_back(
                locally_relevant_magnetic_set.get_view(n_magnetic_dofs,
                                                       n_magnetic_dofs + n_pseudo_pressure_dofs));
    }
    // magnetic constraints
    {
        magnetic_constraints.clear();
        magnetic_constraints.reinit(locally_relevant_magnetic_set);

        DoFTools::make_hanging_node_constraints(magnetic_dof_handler,
                                                magnetic_constraints);
        // zero pseudo pressure constraint
        {
            const Functions::ZeroFunction<dim> zero_function(dim+1);

            const std::map<typename types::boundary_id, const Function<dim>*>
            pseudo_pressure_boundary_values
            = {{GridFactory::BoundaryIds::ICB, &zero_function},
               {GridFactory::BoundaryIds::CMB, &zero_function}};

            const FEValuesExtractors::Scalar pseudo_pressure(dim);
            VectorTools::interpolate_boundary_values(
                    magnetic_dof_handler,
                    pseudo_pressure_boundary_values,
                    magnetic_constraints,
                    magnetic_fe.component_mask(pseudo_pressure));
        }
        // zero tangential magnetic field constraint
        {
            const Functions::ZeroFunction<dim>  zero_function(dim);
            typename FunctionMap<dim,double>::type  function_map
            {{types::boundary_id(DomainIdentifiers::BoundaryIds::ICB),&zero_function},
             {types::boundary_id(DomainIdentifiers::BoundaryIds::CMB),&zero_function}};

            const std::set<types::boundary_id>  boundary_ids
            {DomainIdentifiers::BoundaryIds::ICB,
             DomainIdentifiers::BoundaryIds::CMB};

            VectorTools::compute_nonzero_tangential_flux_constraints(
                    magnetic_dof_handler,
                    0,
                    boundary_ids,
                    function_map,
                    magnetic_constraints);
        }
        magnetic_constraints.close();
    }

    // magnetic matrix and vector setup
    setup_magnetic_system(locally_owned_magnetic_dofs,
                          locally_relevant_magnetic_dofs);

    // reinit block vectors
    magnetic_solution.reinit(locally_relevant_magnetic_dofs,
                             mpi_communicator);
    old_magnetic_solution.reinit(magnetic_solution);
    old_old_magnetic_solution.reinit(magnetic_solution);

    magnetic_rhs.reinit(locally_owned_magnetic_dofs,
                        locally_relevant_magnetic_dofs,
                        mpi_communicator,
                        true);

    // reinit pressure vectors
    phi_pseudo_pressure.reinit(magnetic_solution);
    old_phi_pseudo_pressure.reinit(magnetic_solution);
    old_old_phi_pseudo_pressure.reinit(magnetic_solution);

    return std::pair<types::global_dof_index,types::global_dof_index>
    (n_magnetic_dofs,n_pseudo_pressure_dofs);
}


template<int dim>
void BuoyantFluidSolver<dim>::setup_magnetic_system
(const std::vector<IndexSet>    &locally_owned_dofs,
 const std::vector<IndexSet>    &locally_relevant_dofs)
{
    preconditioner_magnetic_diffusion.reset();
    preconditioner_magnetic_projection.reset();
    preconditioner_magnetic_pressure_mass.reset();

    magnetic_matrix.clear();
    magnetic_mass_matrix.clear();
    magnetic_laplace_matrix.clear();
    magnetic_stabilization_matrix.clear();

    // sparsity pattern for magnetic matrix
    LA::BlockDynamicSparsityPattern dsp(locally_owned_dofs,
                                        locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);
    {
        Table<2,DoFTools::Coupling> magnetic_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                // magnetic-magnetic coupling
                if (c<dim && d<dim)
                    magnetic_coupling[c][d] = DoFTools::Coupling::always;
                // magnetic-pseudo pressure coupling
                else if ((c<dim && d==dim) || (c==dim && d<dim))
                    magnetic_coupling[c][d] = DoFTools::Coupling::always;
                else
                    magnetic_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                magnetic_dof_handler,
                magnetic_coupling,
                dsp,
                magnetic_constraints,
                false,
                Utilities::MPI::this_mpi_process(mpi_communicator));
        dsp.compress();
    }
    magnetic_matrix.reinit(dsp);

    // sparsity pattern for mass matrix
    LA::BlockDynamicSparsityPattern mass_dsp(locally_owned_dofs,
                                             locally_owned_dofs,
                                             locally_relevant_dofs,
                                             mpi_communicator);
    // sparsity pattern for laplace matrix
    LA::BlockDynamicSparsityPattern laplace_dsp(locally_owned_dofs,
                                                locally_owned_dofs,
                                                locally_relevant_dofs,
                                                mpi_communicator);
    {
        // auxiliary coupling structure for pressure part
        Table<2,DoFTools::Coupling> pseudo_pressure_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c==dim)
                    pseudo_pressure_coupling[c][d] = DoFTools::always;
                else
                    pseudo_pressure_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                magnetic_dof_handler,
                pseudo_pressure_coupling,
                laplace_dsp,
                magnetic_constraints,
                false,
                Utilities::MPI::this_mpi_process(mpi_communicator));
        laplace_dsp.compress();

        DoFTools::make_sparsity_pattern(
                magnetic_dof_handler,
                pseudo_pressure_coupling,
                mass_dsp,
                magnetic_constraints,
                true,
                Utilities::MPI::this_mpi_process(mpi_communicator));
        mass_dsp.compress();
    }
    magnetic_laplace_matrix.reinit(laplace_dsp);
    magnetic_laplace_matrix.block(0,0).reinit(dsp.block(0,0));

    magnetic_stabilization_matrix.reinit(laplace_dsp);
    magnetic_stabilization_matrix.block(0,0).reinit(dsp.block(0,0));

    magnetic_mass_matrix.reinit(mass_dsp);
    magnetic_mass_matrix.block(0,0).reinit(dsp.block(0,0));

    rebuild_magnetic_matrices = true;
}

}  // namespace BuoyantFluid


// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::setup_dofs();
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_dofs();

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_temperature_matrix
(const IndexSet &,
 const IndexSet &);
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_temperature_matrix
(const IndexSet &,
 const IndexSet &);

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_navier_stokes_system
(const std::vector<IndexSet>    &,
 const std::vector<IndexSet>    &);
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_navier_stokes_system
(const std::vector<IndexSet>    &,
 const std::vector<IndexSet>    &);

template void BuoyantFluid::BuoyantFluidSolver<2>::setup_magnetic_system
(const std::vector<IndexSet>    &,
 const std::vector<IndexSet>    &);
template void BuoyantFluid::BuoyantFluidSolver<3>::setup_magnetic_system
(const std::vector<IndexSet>    &,
 const std::vector<IndexSet>    &);

