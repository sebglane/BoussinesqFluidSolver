/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/matrix_tools.h>

#include <adsolic/temperature_solver.h>

namespace adsolic {

template<int dim>
void TemperatureSolver<dim>::assemble_temperature_system()
{
    if (parameters.verbose)
        pcout << "      Assembling temperature system..." << std::endl;

    computing_timer->enter_subsection("assemble temperature system");

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // quadrature formula
    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    // time stepping coefficients
    const std::array<double,3> alpha = timestepper.alpha();
    const std::array<double,3> gamma = timestepper.gamma();

    // assemble temperature matrices
    if (rebuild_temperature_matrices)
    {
        temperature_mass_matrix = 0;
        temperature_stiffness_matrix = 0;

        // assemble temperature right-hand side
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           temperature_dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           temperature_dof_handler.end()),
                std::bind(&TemperatureSolver<dim>::local_assemble_temperature_matrix,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&TemperatureSolver<dim>::copy_local_to_global_temperature_matrix,
                          this,
                          std::placeholders::_1),
                TemperatureAssembly::Scratch::Matrix<dim>(temperature_fe,
                                                          mapping,
                                                          quadrature_formula,
                                                          update_values|
                                                          update_gradients|
                                                          update_JxW_values),
                TemperatureAssembly::CopyData::Matrix<dim>(temperature_fe));

        temperature_mass_matrix.compress(VectorOperation::add);
        temperature_stiffness_matrix.compress(VectorOperation::add);

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0] / timestepper.step_size();
        temperature_matrix.add(gamma[0] * equation_coefficient,
                               temperature_stiffness_matrix);

        temperature_matrix.compress(VectorOperation::add);

        rebuild_temperature_matrices = false;
        rebuild_temperature_preconditioner = true;
    }

    if (timestepper.coefficients_have_changed())
    {
        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0] / timestepper.step_size();
        temperature_matrix.add(gamma[0] * equation_coefficient,
                               temperature_stiffness_matrix);

        temperature_matrix.compress(VectorOperation::add);

        rebuild_temperature_preconditioner = true;
    }

    // reset all entries
    temperature_rhs = 0;

    ZeroTensorFunction<1,dim>   dummy_function;

    // assemble temperature right-hand side
    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       temperature_dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       temperature_dof_handler.end()),
            std::bind(&TemperatureSolver<dim>::local_assemble_temperature_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&TemperatureSolver<dim>::copy_local_to_global_temperature_rhs,
                      this,
                      std::placeholders::_1),
            TemperatureAssembly::Scratch::RightHandSide<dim>(
                    temperature_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_gradients|
                    update_JxW_values,
                    dummy_function,
                    alpha,
                    timestepper.beta(),
                    gamma),
            TemperatureAssembly::CopyData::RightHandSide<dim>(temperature_fe));

    temperature_rhs.compress(VectorOperation::add);
}

template <int dim>
void TemperatureSolver<dim>::local_assemble_temperature_matrix
(const typename DoFHandler<dim>::active_cell_iterator   &cell,
 TemperatureAssembly::Scratch::Matrix<dim>              &scratch,
 TemperatureAssembly::CopyData::Matrix<dim>             &data)
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
void TemperatureSolver<dim>::copy_local_to_global_temperature_matrix
(const TemperatureAssembly::CopyData::Matrix<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            temperature_mass_matrix);
    temperature_constraints.distribute_local_to_global(
            data.local_laplace_matrix,
            data.local_dof_indices,
            temperature_stiffness_matrix);
}

template <int dim>
void TemperatureSolver<dim>::local_assemble_temperature_rhs
(const typename DoFHandler<dim>::active_cell_iterator    &cell,
 TemperatureAssembly::Scratch::RightHandSide<dim>        &scratch,
 TemperatureAssembly::CopyData::RightHandSide<dim>       &data)
{
    data.matrix_for_bc = 0;
    data.local_rhs = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.temperature_fe_values.reinit(cell);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);
    scratch.temperature_fe_values.get_function_gradients(old_temperature_solution,
                                                         scratch.old_temperature_gradients);
    scratch.temperature_fe_values.get_function_gradients(old_old_temperature_solution,
                                                         scratch.old_old_temperature_gradients);

    scratch.advection_field.set_time(timestepper.previous());
    scratch.advection_field.value_list(scratch.temperature_fe_values.get_quadrature_points(),
                                       scratch.old_velocity_values);

    scratch.advection_field.set_time(timestepper.previous());
    scratch.advection_field.value_list(scratch.temperature_fe_values.get_quadrature_points(),
                                       scratch.old_old_velocity_values);

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int i=0; i<scratch.dofs_per_cell; ++i)
        {
            scratch.phi_temperature[i]      = scratch.temperature_fe_values.shape_value(i, q);
            scratch.grad_phi_temperature[i] = scratch.temperature_fe_values.shape_grad(i, q);
        }

        const double time_derivative_temperature =
                scratch.alpha[1] / timestepper.step_size() * scratch.old_temperature_values[q]
                    + scratch.alpha[2] / timestepper.step_size() * scratch.old_old_temperature_values[q];

        const double nonlinear_term_temperature =
                scratch.beta[0] * scratch.old_temperature_gradients[q] * scratch.old_velocity_values[q]
                    + scratch.beta[1] * scratch.old_old_temperature_gradients[q] * scratch.old_old_velocity_values[q];

        const Tensor<1,dim> linear_term_temperature =
                scratch.gamma[1] * scratch.old_temperature_gradients[q]
                    + scratch.gamma[2] * scratch.old_old_temperature_gradients[q];

        for (unsigned int i=0; i<scratch.dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                    - time_derivative_temperature * scratch.phi_temperature[i]
                    - nonlinear_term_temperature * scratch.phi_temperature[i]
                    - equation_coefficient * linear_term_temperature * scratch.grad_phi_temperature[i]
                    ) * scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                for (unsigned int j=0; j<scratch.dofs_per_cell; ++j)
                    data.matrix_for_bc(j,i) += (
                                  scratch.alpha[0] / timestepper.step_size() *
                                  scratch.phi_temperature[i] * scratch.phi_temperature[j]
                                + scratch.gamma[0] * equation_coefficient
                                  * scratch.grad_phi_temperature[i] * scratch.grad_phi_temperature[j]
                                ) * scratch.temperature_fe_values.JxW(q);
        }

    }
}

template <int dim>
void TemperatureSolver<dim>::copy_local_to_global_temperature_rhs
(const TemperatureAssembly::CopyData::RightHandSide<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            temperature_rhs,
            data.matrix_for_bc);
}

// explicit instantiation
template void TemperatureSolver<2>::assemble_temperature_system();
template void TemperatureSolver<3>::assemble_temperature_system();

template void TemperatureSolver<2>::local_assemble_temperature_matrix
(const typename DoFHandler<2>::active_cell_iterator &,
 TemperatureAssembly::Scratch::Matrix<2>            &,
 TemperatureAssembly::CopyData::Matrix<2>           &);
template void TemperatureSolver<3>::local_assemble_temperature_matrix
(const typename DoFHandler<3>::active_cell_iterator &,
 TemperatureAssembly::Scratch::Matrix<3>            &,
 TemperatureAssembly::CopyData::Matrix<3>           &);

template void TemperatureSolver<2>::copy_local_to_global_temperature_matrix
(const TemperatureAssembly::CopyData::Matrix<2> &);
template void TemperatureSolver<3>::copy_local_to_global_temperature_matrix
(const TemperatureAssembly::CopyData::Matrix<3> &);

template void TemperatureSolver<2>::local_assemble_temperature_rhs
(const typename DoFHandler<2>::active_cell_iterator &,
 TemperatureAssembly::Scratch::RightHandSide<2>     &,
 TemperatureAssembly::CopyData::RightHandSide<2>    &);
template void TemperatureSolver<3>::local_assemble_temperature_rhs
(const typename DoFHandler<3>::active_cell_iterator &,
 TemperatureAssembly::Scratch::RightHandSide<3>     &,
 TemperatureAssembly::CopyData::RightHandSide<3>    &);

template void TemperatureSolver<2>::copy_local_to_global_temperature_rhs
(const TemperatureAssembly::CopyData::RightHandSide<2> &);
template void TemperatureSolver<3>::copy_local_to_global_temperature_rhs
(const TemperatureAssembly::CopyData::RightHandSide<3> &);


}  // namespace adsolic
