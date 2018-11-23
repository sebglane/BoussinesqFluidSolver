/*
 * local_temperature_assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include "buoyant_fluid_solver.h"

namespace BuoyantFluid {

template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_temperature_matrix(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        TemperatureAssembly::Scratch::Matrix<dim> &scratch,
        TemperatureAssembly::CopyData::Matrix<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    scratch.temperature_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.grad_phi_T[k] = scratch.temperature_fe_values.shape_grad(k,q);
            scratch.phi_T[k]      = scratch.temperature_fe_values.shape_value(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += scratch.phi_T[i] * scratch.phi_T[j] * scratch.temperature_fe_values.JxW(q);
                data.local_stiffness_matrix(i,j)
                    += scratch.grad_phi_T[i] * scratch.grad_phi_T[j] * scratch.temperature_fe_values.JxW(q);
            }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_stiffness_matrix(i,j) = data.local_stiffness_matrix(j,i);
        }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_temperature_matrix(
        const TemperatureAssembly::CopyData::Matrix<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            temperature_mass_matrix);
    temperature_constraints.distribute_local_to_global(
            data.local_stiffness_matrix,
            data.local_dof_indices,
            temperature_stiffness_matrix);
}


template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_temperature_rhs(
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        TemperatureAssembly::Scratch::RightHandSide<dim>        &scratch,
        TemperatureAssembly::CopyData::RightHandSide<dim>       &data)
{
    const std::vector<double> alpha = (timestep_number != 0?
                                            imex_coefficients.alpha(timestep/old_timestep):
                                            std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> beta = (timestep_number != 0?
                                            imex_coefficients.beta(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                            imex_coefficients.gamma(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0,0.0}));

    const FEValuesExtractors::Vector    velocity(0);

    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    data.matrix_for_bc = 0;
    data.local_rhs = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.temperature_fe_values.reinit (cell);

    typename DoFHandler<dim>::active_cell_iterator
    stokes_cell(&triangulation,
                cell->level(),
                cell->index(),
                &stokes_dof_handler);
    scratch.stokes_fe_values.reinit(stokes_cell);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);
    scratch.temperature_fe_values.get_function_gradients(old_temperature_solution,
                                                         scratch.old_temperature_gradients);
    scratch.temperature_fe_values.get_function_gradients(old_old_temperature_solution,
                                                         scratch.old_old_temperature_gradients);

    scratch.stokes_fe_values[velocity].get_function_values(old_stokes_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_stokes_solution,
                                                           scratch.old_old_velocity_values);


    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            scratch.phi_T[i]      = scratch.temperature_fe_values.shape_value(i, q);
            scratch.grad_phi_T[i] = scratch.temperature_fe_values.shape_grad(i, q);
        }

        const double time_derivative_temperature =
                alpha[1] * scratch.old_temperature_values[q]
                    + alpha[2] * scratch.old_old_temperature_values[q];

        const double nonlinear_term_temperature =
                beta[0] * scratch.old_temperature_gradients[q] * scratch.old_velocity_values[q]
                    + beta[1] * scratch.old_old_temperature_gradients[q] * scratch.old_old_velocity_values[q];

        const Tensor<1,dim> linear_term_temperature =
                gamma[1] * scratch.old_temperature_gradients[q]
                    + gamma[2] * scratch.old_old_temperature_gradients[q];

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                    - time_derivative_temperature * scratch.phi_T[i]
                    - timestep * nonlinear_term_temperature * scratch.phi_T[i]
                    - timestep * equation_coefficients[3] * linear_term_temperature * scratch.grad_phi_T[i]
                    ) * scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    data.matrix_for_bc(j,i) += (
                                  alpha[0] * scratch.phi_T[i] * scratch.phi_T[j]
                                + gamma[0] * timestep * equation_coefficients[3] * scratch.grad_phi_T[i] * scratch.grad_phi_T[j]
                                ) * scratch.temperature_fe_values.JxW(q);

        }

    }
}

template <int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_temperature_rhs(
        const TemperatureAssembly::CopyData::RightHandSide<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            temperature_rhs,
            data.matrix_for_bc);
}

}  // namespace BuoyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_temperature_matrix(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        TemperatureAssembly::Scratch::Matrix<2> &scratch,
        TemperatureAssembly::CopyData::Matrix<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_temperature_matrix(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        TemperatureAssembly::Scratch::Matrix<3> &scratch,
        TemperatureAssembly::CopyData::Matrix<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_temperature_matrix(
        const TemperatureAssembly::CopyData::Matrix<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_temperature_matrix(
        const TemperatureAssembly::CopyData::Matrix<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_temperature_rhs(
        const typename DoFHandler<2>::active_cell_iterator    &cell,
        TemperatureAssembly::Scratch::RightHandSide<2>        &scratch,
        TemperatureAssembly::CopyData::RightHandSide<2>       &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_temperature_rhs(
        const typename DoFHandler<3>::active_cell_iterator    &cell,
        TemperatureAssembly::Scratch::RightHandSide<3>        &scratch,
        TemperatureAssembly::CopyData::RightHandSide<3>       &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_temperature_rhs(
        const TemperatureAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_temperature_rhs(
        const TemperatureAssembly::CopyData::RightHandSide<3> &data);
