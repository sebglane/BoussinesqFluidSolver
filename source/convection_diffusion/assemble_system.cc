/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */
#include <adsolic/convection_diffusion_solver.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/matrix_tools.h>


namespace adsolic {

template<int dim>
void ConvectionDiffusionSolver<dim>::assemble_system()
{
    if (parameters.verbose)
        pcout << "      Assembling convection diffussion system..." << std::endl;

    computing_timer->enter_subsection("assemble convect. diff. system");

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // quadrature formula
    const QGauss<dim> quadrature_formula(parameters.fe_degree + 2);

    // time stepping coefficients
    const std::array<double,3> alpha = timestepper.alpha();
    const std::array<double,3> gamma = timestepper.gamma();

    // assemble matrices
    if (rebuild_matrices)
    {
        mass_matrix = 0;
        stiffness_matrix = 0;

        // assemble right-hand side
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           dof_handler.end()),
                std::bind(&ConvectionDiffusionSolver<dim>::local_assemble_matrix,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&ConvectionDiffusionSolver<dim>::copy_local_to_global_matrix,
                          this,
                          std::placeholders::_1),
                ConvectionDiffusionAssembly::Scratch::Matrix<dim>(fe,
                                                          mapping,
                                                          quadrature_formula,
                                                          update_values|
                                                          update_gradients|
                                                          update_JxW_values),
                ConvectionDiffusionAssembly::CopyData::Matrix<dim>(fe));

        mass_matrix.compress(VectorOperation::add);
        stiffness_matrix.compress(VectorOperation::add);

        system_matrix.copy_from(mass_matrix);
        system_matrix *= alpha[0] / timestepper.step_size();
        system_matrix.add(gamma[0] * equation_coefficient,
                               stiffness_matrix);

        system_matrix.compress(VectorOperation::add);

        rebuild_matrices = false;
        rebuild_preconditioner = true;
    }

    if (timestepper.coefficients_have_changed())
    {
        system_matrix.copy_from(mass_matrix);
        system_matrix *= alpha[0] / timestepper.step_size();
        system_matrix.add(gamma[0] * equation_coefficient,
                               stiffness_matrix);

        system_matrix.compress(VectorOperation::add);

        rebuild_preconditioner = true;
    }

    // reset all entries
    rhs = 0;

    ZeroTensorFunction<1,dim>   dummy_function;

    // assemble right-hand side
    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       dof_handler.end()),
            std::bind(&ConvectionDiffusionSolver<dim>::local_assemble_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&ConvectionDiffusionSolver<dim>::copy_local_to_global_rhs,
                      this,
                      std::placeholders::_1),
            ConvectionDiffusionAssembly::Scratch::RightHandSide<dim>(
                    fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_gradients|
                    update_JxW_values,
                    dummy_function,
                    alpha,
                    timestepper.beta(),
                    gamma),
            ConvectionDiffusionAssembly::CopyData::RightHandSide<dim>(fe));

    rhs.compress(VectorOperation::add);
}

template <int dim>
void ConvectionDiffusionSolver<dim>::local_assemble_matrix
(const typename DoFHandler<dim>::active_cell_iterator   &cell,
 ConvectionDiffusionAssembly::Scratch::Matrix<dim>              &scratch,
 ConvectionDiffusionAssembly::CopyData::Matrix<dim>             &data)
{
    const unsigned int dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.fe_values.n_quadrature_points;

    scratch.fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_laplace_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi[k]     = scratch.fe_values.shape_value(k, q);
            scratch.grad_phi[k]= scratch.fe_values.shape_grad(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += scratch.phi[i] * scratch.phi[j]
                       * scratch.fe_values.JxW(q);
                data.local_laplace_matrix(i,j)
                    += scratch.grad_phi[i] * scratch.grad_phi[j]
                       * scratch.fe_values.JxW(q);
            }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_laplace_matrix(i,j) = data.local_laplace_matrix(j,i);
        }
    }
}

template<int dim>
void ConvectionDiffusionSolver<dim>::copy_local_to_global_matrix
(const ConvectionDiffusionAssembly::CopyData::Matrix<dim> &data)
{
    constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            mass_matrix);
    constraints.distribute_local_to_global(
            data.local_laplace_matrix,
            data.local_dof_indices,
            stiffness_matrix);
}

template <int dim>
void ConvectionDiffusionSolver<dim>::local_assemble_rhs
(const typename DoFHandler<dim>::active_cell_iterator    &cell,
 ConvectionDiffusionAssembly::Scratch::RightHandSide<dim>        &scratch,
 ConvectionDiffusionAssembly::CopyData::RightHandSide<dim>       &data)
{
    data.matrix_for_bc = 0;
    data.local_rhs = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.fe_values.reinit(cell);

    scratch.fe_values.get_function_values(old_solution,
                                          scratch.old_values);
    scratch.fe_values.get_function_values(old_old_solution,
                                          scratch.old_old_values);
    scratch.fe_values.get_function_gradients(old_solution,
                                             scratch.old_gradients);
    scratch.fe_values.get_function_gradients(old_old_solution,
                                             scratch.old_old_gradients);

    scratch.advection_field.set_time(timestepper.previous());
    scratch.advection_field.value_list(scratch.fe_values.get_quadrature_points(),
                                       scratch.old_velocity_values);

    scratch.advection_field.set_time(timestepper.previous());
    scratch.advection_field.value_list(scratch.fe_values.get_quadrature_points(),
                                       scratch.old_old_velocity_values);

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int i=0; i<scratch.dofs_per_cell; ++i)
        {
            scratch.phi[i]      = scratch.fe_values.shape_value(i, q);
            scratch.grad_phi[i] = scratch.fe_values.shape_grad(i, q);
        }

        const double time_derivative =
                scratch.alpha[1] / timestepper.step_size() * scratch.old_values[q]
                    + scratch.alpha[2] / timestepper.step_size() * scratch.old_old_values[q];

        const double nonlinear_term =
                scratch.beta[0] * scratch.old_gradients[q] * scratch.old_velocity_values[q]
                    + scratch.beta[1] * scratch.old_old_gradients[q] * scratch.old_old_velocity_values[q];

        const Tensor<1,dim> linear_term =
                scratch.gamma[1] * scratch.old_gradients[q]
                    + scratch.gamma[2] * scratch.old_old_gradients[q];

        for (unsigned int i=0; i<scratch.dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                    - time_derivative * scratch.phi[i]
                    - nonlinear_term * scratch.phi[i]
                    - equation_coefficient * linear_term * scratch.grad_phi[i]
                    ) * scratch.fe_values.JxW(q);

            if (constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                for (unsigned int j=0; j<scratch.dofs_per_cell; ++j)
                    data.matrix_for_bc(j,i) += (
                                  scratch.alpha[0] / timestepper.step_size() *
                                  scratch.phi[i] * scratch.phi[j]
                                + scratch.gamma[0] * equation_coefficient
                                  * scratch.grad_phi[i] * scratch.grad_phi[j]
                                ) * scratch.fe_values.JxW(q);
        }

    }
}

template <int dim>
void ConvectionDiffusionSolver<dim>::copy_local_to_global_rhs
(const ConvectionDiffusionAssembly::CopyData::RightHandSide<dim> &data)
{
    constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            rhs,
            data.matrix_for_bc);
}

// explicit instantiation
template void ConvectionDiffusionSolver<2>::assemble_system();
template void ConvectionDiffusionSolver<3>::assemble_system();

template void ConvectionDiffusionSolver<2>::local_assemble_matrix
(const typename DoFHandler<2>::active_cell_iterator &,
 ConvectionDiffusionAssembly::Scratch::Matrix<2>            &,
 ConvectionDiffusionAssembly::CopyData::Matrix<2>           &);
template void ConvectionDiffusionSolver<3>::local_assemble_matrix
(const typename DoFHandler<3>::active_cell_iterator &,
 ConvectionDiffusionAssembly::Scratch::Matrix<3>            &,
 ConvectionDiffusionAssembly::CopyData::Matrix<3>           &);

template void ConvectionDiffusionSolver<2>::copy_local_to_global_matrix
(const ConvectionDiffusionAssembly::CopyData::Matrix<2> &);
template void ConvectionDiffusionSolver<3>::copy_local_to_global_matrix
(const ConvectionDiffusionAssembly::CopyData::Matrix<3> &);

template void ConvectionDiffusionSolver<2>::local_assemble_rhs
(const typename DoFHandler<2>::active_cell_iterator &,
 ConvectionDiffusionAssembly::Scratch::RightHandSide<2>     &,
 ConvectionDiffusionAssembly::CopyData::RightHandSide<2>    &);
template void ConvectionDiffusionSolver<3>::local_assemble_rhs
(const typename DoFHandler<3>::active_cell_iterator &,
 ConvectionDiffusionAssembly::Scratch::RightHandSide<3>     &,
 ConvectionDiffusionAssembly::CopyData::RightHandSide<3>    &);

template void ConvectionDiffusionSolver<2>::copy_local_to_global_rhs
(const ConvectionDiffusionAssembly::CopyData::RightHandSide<2> &);
template void ConvectionDiffusionSolver<3>::copy_local_to_global_rhs
(const ConvectionDiffusionAssembly::CopyData::RightHandSide<3> &);


}  // namespace adsolic
