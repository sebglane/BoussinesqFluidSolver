/*
 * setup.cc
 *
 *  Created on: Jul 29, 2019
 *      Author: sg
 */

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/vector_tools.h>

#include <adsolic/navier_stokes_solver.h>

namespace adsolic
{


template<int dim>
void NavierStokesSolver<dim>::setup_dofs()
{
    this->pcout << "Setup dofs..." << std::endl;

    TimerOutput::Scope timer_section(*(this->computing_timer), "Nav.-St. setup dofs");

    locally_owned_dofs.clear();
    locally_relevant_dofs.clear();
    this->dof_handler.distribute_dofs(fe);

    DoFRenumbering::Cuthill_McKee(this->dof_handler);
    DoFRenumbering::block_wise(this->dof_handler);

    // count dofs
    std::vector<unsigned int> block_component(2,0);
    block_component[1] = 1;
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(this->dof_handler,
                                   dofs_per_block,
                                   block_component);

    const unsigned int  n_velocity_dofs = dofs_per_block[0],
                        n_pressure_dofs = dofs_per_block[1];

    // extract locally owned and relevant dofs
    IndexSet    locally_relevant_set;
    {
        const IndexSet locally_owned_set
        = this->dof_handler.locally_owned_dofs();

        locally_owned_dofs.push_back(
                locally_owned_set.get_view(0, n_velocity_dofs));
        locally_owned_dofs.push_back(
                locally_owned_set.get_view(n_velocity_dofs,
                                           n_velocity_dofs + n_pressure_dofs));

        DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                                locally_relevant_set);

        locally_relevant_dofs.push_back(
                locally_relevant_set.get_view(0, n_velocity_dofs));
        locally_relevant_dofs.push_back(
                locally_relevant_set.get_view(n_velocity_dofs,
                                              n_velocity_dofs + n_pressure_dofs));
    }
    // constraints
    {
        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(locally_relevant_set);

        DoFTools::make_hanging_node_constraints(this->dof_handler,
                                                hanging_node_constraints);

        /*
        const FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(this->dof_handler,
                                                 velocity_boundary_values,
                                                 tentative_velocity_constraints,
                                                 fe.component_mask(velocities));
        */

        hanging_node_constraints.close();
    }
    // add pressure constraints
    {
        pressure_constraints.clear();
        pressure_constraints.reinit(locally_relevant_set);

        pressure_constraints.merge(hanging_node_constraints);

        const FEValuesExtractors::Scalar    pressure(dim);

        IndexSet    boundary_pressure_dofs;

        DoFTools::extract_boundary_dofs(this->dof_handler,
                                        fe.component_mask(pressure),
                                        boundary_pressure_dofs);

        types::global_dof_index local_idx = numbers::invalid_dof_index;

        IndexSet::ElementIterator
        idx = boundary_pressure_dofs.begin(),
        endidx = boundary_pressure_dofs.end();
        for(; idx != endidx; ++idx)
            if (pressure_constraints.can_store_line(*idx)
                    && !pressure_constraints.is_constrained(*idx))
                local_idx = *idx;

        // Make a reduction to find the smallest index (processors that
        // found a larger candidate just happened to not be able to store
        // that index with the minimum value). Note that it is possible that
        // some processors might not be able to find a potential DoF, for
        // example because they don't own any DoFs. On those processors we
        // will use dof_handler.n_dofs() when building the minimum (larger
        // than any valid DoF index).
        const types::global_dof_index global_idx
        = Utilities::MPI::min((local_idx != numbers::invalid_dof_index) ? local_idx : this->dof_handler.n_dofs(),
                              this->triangulation.get_communicator());

        Assert(global_idx < this->dof_handler.n_dofs(),
               ExcMessage("Error, couldn't find a pressure DoF to constrain."));

        // Finally set this DoF to zero (if we care about it):
        if (pressure_constraints.can_store_line(global_idx))
        {
            Assert(!pressure_constraints.is_constrained(global_idx), ExcInternalError());
            pressure_constraints.add_line(global_idx);
        }

        pressure_constraints.close();
    }
    /*
     * TODO: constraints and boundary conditions...
     */

    // stokes matrix and vector setup
    setup_system_matrix(locally_owned_dofs,
                        locally_relevant_dofs);

    // reinit block vectors
    this->solution.reinit(locally_relevant_dofs,
                          this->triangulation.get_communicator());
    this->old_solution.reinit(this->solution);
    this->old_old_solution.reinit(this->solution);

    this->rhs.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     this->triangulation.get_communicator(),
                     true);
}


template<int dim>
void NavierStokesSolver<dim>::setup_system_matrix
(const std::vector<IndexSet>    &locally_owned_dofs,
 const std::vector<IndexSet>    &locally_relevant_dofs)
{
    preconditioner_diffusion.reset();
    preconditioner_projection.reset();
    preconditioner_pressure_mass.reset();
    preconditioner_velocity_mass.reset();

    system_matrix.clear();
    mass_matrix.clear();
    stiffness_matrix.clear();

    const MPI_Comm  mpi_communicator = this->triangulation.get_communicator();

    // sparsity pattern for matrix
    LA::BlockDynamicSparsityPattern dsp(locally_owned_dofs,
                                        locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);
    {
        Table<2,DoFTools::Coupling> coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c<dim)
                    coupling[c][d] = DoFTools::always;
                else if ((c<dim && d==dim) || (c==dim && d<dim))
                    coupling[c][d] = DoFTools::always;
                else
                    coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                this->dof_handler,
                coupling,
                dsp,
                hanging_node_constraints,
                false,
                Utilities::MPI::this_mpi_process(mpi_communicator));

        dsp.compress();
    }
    system_matrix.reinit(dsp);

    // sparsity pattern for laplace matrix
    LA::BlockDynamicSparsityPattern laplace_dsp(locally_owned_dofs,
                                                locally_owned_dofs,
                                                locally_relevant_dofs,
                                                mpi_communicator);
    {
        // auxiliary coupling structure
        Table<2,DoFTools::Coupling> pressure_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c==dim)
                    pressure_coupling[c][d] = DoFTools::always;
                else
                    pressure_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                this->dof_handler,
                pressure_coupling,
                laplace_dsp,
                pressure_constraints,
                false,
                Utilities::MPI::this_mpi_process(mpi_communicator));

        laplace_dsp.compress();
    }
    system_matrix.reinit(laplace_dsp);
    stiffness_matrix.block(0,0).reinit(dsp.block(0,0));

    // sparsity pattern for mass matrix
    LA::BlockDynamicSparsityPattern mass_dsp(locally_owned_dofs,
                                             locally_owned_dofs,
                                             locally_relevant_dofs,
                                             mpi_communicator);
    {
        // auxiliary coupling structure
        Table<2,DoFTools::Coupling> pressure_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c==dim)
                    pressure_coupling[c][d] = DoFTools::always;
                else
                    pressure_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
                this->dof_handler,
                pressure_coupling,
                mass_dsp,
                hanging_node_constraints);

        mass_dsp.compress();
    }
    mass_matrix.reinit(mass_dsp);
    mass_matrix.block(0,0).reinit(dsp.block(0,0));

    rebuild_matrices = true;
}

// explicit instantiation
template void NavierStokesSolver<2>::setup_dofs();
template void NavierStokesSolver<3>::setup_dofs();

template void NavierStokesSolver<2>::setup_system_matrix
(const std::vector<IndexSet>    &,
 const std::vector<IndexSet>    &);
template void NavierStokesSolver<3>::setup_system_matrix
(const std::vector<IndexSet>    &,
 const std::vector<IndexSet>    &);

}  // namespace adsolic



