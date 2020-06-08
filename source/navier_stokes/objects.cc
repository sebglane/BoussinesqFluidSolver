/*
 * sub_objects.cc
 *
 *  Created on: Aug 7, 2019
 *      Author: sg
 */

#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/vector_tools.h>

#include <adsolic/navier_stokes_solver.h>

namespace adsolic
{

namespace NavierStokesObjects
{

using namespace NavierStokesAssembly;

template<int dim>
DefaultObjects<dim>::DefaultObjects
(const parallel::distributed::Triangulation<dim>  &triangulation,
 const Mapping<dim>        &mapping)
:
mpi_communicator(triangulation.get_communicator()),
mapping(mapping),
dof_handler(triangulation),
rebuild_matrices(true)
{}

template<int dim>
void
DefaultObjects<dim>::setup_matrices
(const IndexSet &locally_owned_dofs,
 const IndexSet &locally_relevant_dofs)
{
    mass_matrix.clear();
    stiffness_matrix.clear();

    // sparsity pattern for matrix
    LA::DynamicSparsityPattern dsp(locally_owned_dofs,
                                   locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    dsp,
                                    current_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));
    dsp.compress();

    mass_matrix.reinit(dsp);
    stiffness_matrix.reinit(dsp);

    rebuild_matrices = true;
}

template<int dim>
PressureObjects<dim>::PressureObjects
(const parallel::distributed::Triangulation<dim>  &triangulation,
 const Mapping<dim>        &mapping,
 const unsigned int         degree)
:
DefaultObjects<dim>(triangulation,mapping),
quadrature(degree + 1),
fe(degree)
{}

template<int dim>
void
PressureObjects<dim>::setup_dofs
(/*const typename FunctionMap<dim>::type &dirichlet_bcs*/)
{
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();

    // distribute dofs
    this->dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(this->dof_handler);
    DoFRenumbering::block_wise(this->dof_handler);

    // extract locally owned and relevant dofs
    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                            this->locally_relevant_dofs);

    // hanging node constraints and periodic boundary conditions
    {
        this->hanging_node_constraints.clear();
        this->hanging_node_constraints.reinit(this->locally_relevant_dofs);

        DoFTools::make_hanging_node_constraints(this->dof_handler,
                                                this->hanging_node_constraints);
        /*
        for (unsigned int d=0; d<dim; ++d)
            if (boundary_conditions->periodic_bcs[d] !=
                std::pair<types::boundary_id,types::boundary_id>
                (numbers::invalid_boundary_id, numbers::invalid_boundary_id))
            {
                const types::boundary_id first_id = boundary_conditions->periodic_bcs[d].first;
                const types::boundary_id second_id = boundary_conditions->periodic_bcs[d].second;
                AssertThrow(boundary_conditions->open_bcs_pressure.find(first_id) ==
                            boundary_conditions->open_bcs_pressure.end() &&
                            boundary_conditions->open_bcs_pressure.find(second_id) ==
                            boundary_conditions->open_bcs_pressure.end() &&
                            boundary_conditions->dirichlet_bcs_velocity.find(first_id) ==
                            boundary_conditions->dirichlet_bcs_velocity.end() &&
                            boundary_conditions->dirichlet_bcs_velocity.find(second_id) ==
                            boundary_conditions->dirichlet_bcs_velocity.end() &&
                            boundary_conditions->no_slip.find(first_id) ==
                            boundary_conditions->no_slip.end() &&
                            boundary_conditions->no_slip.find(second_id) ==
                            boundary_conditions->no_slip.end() &&
                            boundary_conditions->normal_flux.find(first_id) ==
                            boundary_conditions->normal_flux.end() &&
                            boundary_conditions->normal_flux.find(second_id) ==
                            boundary_conditions->normal_flux.end(),
                            ExcMessage("Cannot mix periodic boundary conditions with "
                                       "other types of boundary conditions on same "
                                       "boundary!"));
                AssertThrow(first_id != second_id,
                            ExcMessage("The two faces for periodic boundary conditions "
                                       "must have different boundary indicators!"));
                DoFTools::make_periodicity_constraints(this->dof_handler,
                                                       first_id,
                                                       second_id,
                                                       d,
                                                       hanging_node_constraints);
            }
         */
        this->hanging_node_constraints.close();
    }
    // pressure constraints
    {
        this->current_constraints.clear();
        this->current_constraints.reinit(this->locally_relevant_dofs);

        this->current_constraints.merge(this->hanging_node_constraints);

        // TODO: add true Dirichlet pressure boundary conditions

        this->current_constraints.close();
    }


    // setup matrix
    this->setup_matrices(this->locally_owned_dofs,
                         this->locally_relevant_dofs);

    // reinit block vectors
    this->solution.reinit(this->locally_relevant_dofs,
                          this->mpi_communicator);
    this->old_solution.reinit(this->solution);
    this->old_old_solution.reinit(this->solution);

    update.reinit(this->solution);
    old_update.reinit(this->old_solution);

    this->rhs.reinit(this->locally_owned_dofs,
                     this->locally_relevant_dofs,
                     this->mpi_communicator,
                     true);
}

template<int dim>
void
PressureObjects<dim>::assemble_matrices()
{
    Assert(this->rebuild_matrices == true,
           ExcMessage("Cannot assemble_system_matrix because flag is false"));

    // reset all entries
    this->mass_matrix = 0;
    this->stiffness_matrix = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // assemble matrix
    WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                               this->dof_handler.begin_active()),
                    CellFilter(IteratorFilters::LocallyOwnedCell(),
                               this->dof_handler.end()),
                    std::bind(&PressureObjects<dim>::local_assemble_matrix,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&PressureObjects<dim>::copy_local_to_global_matrix,
                              this,
                              std::placeholders::_1),
                    Scratch::PressureMatrix<dim>(fe,
                                                 this->mapping,
                                                 quadrature,
                                                 update_values|
                                                 update_gradients|
                                                 update_JxW_values),
                    CopyData::Matrix<dim>(fe));

    this->mass_matrix.compress(VectorOperation::add);
    this->stiffness_matrix.compress(VectorOperation::add);

    // rebuild both preconditionerss
    /*
    rebuild_preconditioner_stiffness = true;
    rebuild_preconditioner_mass = true;
    */

    // do not rebuild matrices again
    this->rebuild_matrices = false;
}

template<int dim>
void
PressureObjects<dim>::local_assemble_matrix
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 Scratch::PressureMatrix<dim>                          &scratch,
 CopyData::Matrix<dim>                                 &data)
{
    scratch.fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_laplace_matrix = 0;

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int k=0; k<data.dofs_per_cell; ++k)
        {
            scratch.phi[k]     = scratch.fe_values.shape_value(k, q);
            scratch.grad_phi[k]= scratch.fe_values.shape_grad(k, q);
        }

        const double JxW = scratch.fe_values.JxW(q);

        for (unsigned int i=0; i<data.dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += scratch.phi[i] *
                       scratch.phi[j] *
                       JxW;
                data.local_laplace_matrix(i,j)
                    += scratch.grad_phi[i] *
                       scratch.grad_phi[j] *
                       JxW;
            }
    }
    for (unsigned int i=0; i<data.dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<data.dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_laplace_matrix(i,j) = data.local_laplace_matrix(j,i);
        }
}

template<int dim>
void
PressureObjects<dim>::copy_local_to_global_matrix
(const CopyData::Matrix<dim>  &data)
{
    this->current_constraints.distribute_local_to_global
    (data.local_mass_matrix,
     data.local_dof_indices,
     data.local_dof_indices,
     this->mass_matrix);

    this->current_constraints.distribute_local_to_global
    (data.local_laplace_matrix,
     data.local_dof_indices,
     data.local_dof_indices,
     this->stiffness_matrix);
}

template<int dim>
VelocityObjects<dim>::VelocityObjects
(const parallel::distributed::Triangulation<dim>  &triangulation,
 const Mapping<dim>        &mapping,
 const unsigned int         degree)
:
DefaultObjects<dim>(triangulation,mapping),
quadrature(degree + 1),
fe(FE_Q<dim>(degree), dim)
{}

template<int dim>
void
VelocityObjects<dim>::setup_dofs()
{
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();

    // distribute dofs
    this->dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(this->dof_handler);
    DoFRenumbering::block_wise(this->dof_handler);

    // extract locally owned and relevant dofs
    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                            this->locally_relevant_dofs);

    // hanging node constraints and periodic boundary conditions
    {
        this->hanging_node_constraints.clear();
        this->hanging_node_constraints.reinit(this->locally_relevant_dofs);

        DoFTools::make_hanging_node_constraints(this->dof_handler,
                                                this->hanging_node_constraints);
        /*
        for (unsigned int d=0; d<dim; ++d)
            if (boundary_conditions->periodic_bcs[d] !=
                std::pair<types::boundary_id,types::boundary_id>
                (numbers::invalid_boundary_id, numbers::invalid_boundary_id))
            {
                const types::boundary_id first_id = boundary_conditions->periodic_bcs[d].first;
                const types::boundary_id second_id = boundary_conditions->periodic_bcs[d].second;
                AssertThrow(boundary_conditions->open_bcs_pressure.find(first_id) ==
                            boundary_conditions->open_bcs_pressure.end() &&
                            boundary_conditions->open_bcs_pressure.find(second_id) ==
                            boundary_conditions->open_bcs_pressure.end() &&
                            boundary_conditions->dirichlet_bcs_velocity.find(first_id) ==
                            boundary_conditions->dirichlet_bcs_velocity.end() &&
                            boundary_conditions->dirichlet_bcs_velocity.find(second_id) ==
                            boundary_conditions->dirichlet_bcs_velocity.end() &&
                            boundary_conditions->no_slip.find(first_id) ==
                            boundary_conditions->no_slip.end() &&
                            boundary_conditions->no_slip.find(second_id) ==
                            boundary_conditions->no_slip.end() &&
                            boundary_conditions->normal_flux.find(first_id) ==
                            boundary_conditions->normal_flux.end() &&
                            boundary_conditions->normal_flux.find(second_id) ==
                            boundary_conditions->normal_flux.end(),
                            ExcMessage("Cannot mix periodic boundary conditions with "
                                       "other types of boundary conditions on same "
                                       "boundary!"));
                AssertThrow(first_id != second_id,
                            ExcMessage("The two faces for periodic boundary conditions "
                                       "must have different boundary indicators!"));
                DoFTools::make_periodicity_constraints(this->dof_handler,
                                                       first_id,
                                                       second_id,
                                                       d,
                                                       hanging_node_constraints);
            }
         */
        this->hanging_node_constraints.close();
    }
    // velocity constraints
    {
        this->current_constraints.clear();
        this->current_constraints.reinit(this->locally_relevant_dofs);

        this->current_constraints.merge(this->hanging_node_constraints);

        // TODO: add true Dirichlet velocity boundary conditions

        this->current_constraints.close();
    }

    // velocity correction
    {
        correction_constraints.clear();
        correction_constraints.reinit(this->locally_relevant_dofs);

        correction_constraints.merge(this->hanging_node_constraints);

        /*
         *
         * Maybe we need an additional constraint but I think that this object
         * is not needed.
         *
         */
        correction_constraints.close();
    }

    // setup matrix
    this->setup_matrices(this->locally_owned_dofs,
                         this->locally_relevant_dofs);

    // reinit block vectors
    this->solution.reinit(this->locally_relevant_dofs,
                          this->mpi_communicator);
    this->old_solution.reinit(this->solution);
    this->old_old_solution.reinit(this->solution);
    tentative_solution.reinit(this->solution);

    this->rhs.reinit(this->locally_owned_dofs,
                     this->locally_relevant_dofs,
                     this->mpi_communicator,
                     true);
}

template<int dim>
void
VelocityObjects<dim>::setup_matrices
(const IndexSet &locally_owned_dofs,
 const IndexSet &locally_relevant_dofs)
{
    this->mass_matrix.clear();
    this->stiffness_matrix.clear();
    system_matrix.clear();

    // sparsity pattern
    LA::DynamicSparsityPattern dsp(locally_owned_dofs,
                                   locally_owned_dofs,
                                   locally_relevant_dofs,
                                   this->mpi_communicator);
    {
        Table<2,DoFTools::Coupling> coupling(dim,dim);
        for (unsigned int c=0; c<dim; ++c)
            for (unsigned int d=0; d<dim; ++d)
                if (c==d)
                    coupling[c][d] = DoFTools::always;
                else
                    coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(this->dof_handler,
                                        coupling,
                                        dsp,
                                        this->hanging_node_constraints,
                                        false,
                                        Utilities::MPI::this_mpi_process(this->mpi_communicator));

        dsp.compress();
    }
    // initialize matrices
    this->mass_matrix.reinit(dsp);
    this->stiffness_matrix.reinit(dsp);
    correction_mass_matrix.reinit(dsp);
    system_matrix.reinit(dsp);
}

template<int dim>
void VelocityObjects<dim>::assemble_matrices()
{
    Assert(this->rebuild_matrices == true,
           ExcMessage("Cannot assemble_system_matrix because flag is false"));

    // reset all entries
    this->mass_matrix = 0;
    this->stiffness_matrix = 0;
    correction_mass_matrix = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // assemble matrix
    WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                               this->dof_handler.begin_active()),
                    CellFilter(IteratorFilters::LocallyOwnedCell(),
                               this->dof_handler.end()),
                    std::bind(&VelocityObjects<dim>::local_assemble_matrix,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&VelocityObjects<dim>::copy_local_to_global_matrix,
                              this,
                              std::placeholders::_1),
                    Scratch::VelocityMatrix<dim>(fe,
                                                 this->mapping,
                                                 quadrature,
                                                 update_values|
                                                 update_gradients|
                                                 update_JxW_values),
                    CopyData::Matrix<dim>(fe));

    this->mass_matrix.compress(VectorOperation::add);
    this->stiffness_matrix.compress(VectorOperation::add);
    correction_mass_matrix.compress(VectorOperation::add);

    // rebuild both preconditionerss
    /*
    rebuild_preconditioner_diffusion = true;
    rebuild_preconditioner_mass = true;
    */

    // do not rebuild matrices again
    this->rebuild_matrices = false;
}

template<int dim>
void
VelocityObjects<dim>::local_assemble_matrix
(const typename DoFHandler<dim>::active_cell_iterator   &cell,
 Scratch::VelocityMatrix<dim>                           &scratch,
 CopyData::Matrix<dim>                                  &data)
{
    scratch.fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_laplace_matrix = 0;

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int k=0; k<data.dofs_per_cell; ++k)
        {
            scratch.phi[k]     = scratch.fe_values[scratch.velocity].value(k, q);
            scratch.grad_phi[k]= scratch.fe_values[scratch.velocity].gradient(k, q);
        }

        const double JxW = scratch.fe_values.JxW(q);

        for (unsigned int i=0; i<data.dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += scratch.phi[i] *
                       scratch.phi[j] *
                       JxW;

                data.local_laplace_matrix(i,j)
                    += scalar_product(scratch.grad_phi[i],
                                      scratch.grad_phi[j]) *
                       JxW;
            }
    }
    for (unsigned int i=0; i<data.dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<data.dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_laplace_matrix(i,j) = data.local_laplace_matrix(j,i);
        }
}

template<int dim>
void
VelocityObjects<dim>::copy_local_to_global_matrix
(const CopyData::Matrix<dim>  &data)
{
    this->current_constraints.distribute_local_to_global
    (data.local_mass_matrix,
     data.local_dof_indices,
     data.local_dof_indices,
     this->mass_matrix);

    this->current_constraints.distribute_local_to_global
    (data.local_laplace_matrix,
     data.local_dof_indices,
     data.local_dof_indices,
     this->stiffness_matrix);

    correction_constraints.distribute_local_to_global
    (data.local_mass_matrix,
     data.local_dof_indices,
     data.local_dof_indices,
     correction_mass_matrix);
}

// explicit instantiation
template struct PressureObjects<2>;
template struct PressureObjects<3>;

template struct VelocityObjects<2>;
template struct VelocityObjects<3>;

}  // namespace NavierStokesObjects

}  // namespace adsolic



