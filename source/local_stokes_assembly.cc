/*
 * local_assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include "buoyant_fluid_solver.h"
#include "initial_values.h"

namespace BuoyantFluid {

template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_stokes_matrix(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        StokesAssembly::Scratch::Matrix<dim> &scratch,
        StokesAssembly::CopyData::Matrix<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    scratch.stokes_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_v[k] = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_v[k] = scratch.stokes_fe_values[velocity].gradient(k, q);
            scratch.div_phi_v[k] = scratch.stokes_fe_values[velocity].divergence(k, q);
            scratch.phi_p[k] = scratch.stokes_fe_values[pressure].value(k, q);
            if (parameters.assemble_schur_complement)
            scratch.grad_phi_p[k] = scratch.stokes_fe_values[pressure].gradient(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_matrix(i,j)
                    += (
                          scratch.phi_v[i] * scratch.phi_v[j]
                        - scratch.phi_p[i] * scratch.div_phi_v[j]
                        - scratch.div_phi_v[i] * scratch.phi_p[j]
                        + scratch.phi_p[i] * scratch.phi_p[j]
                        ) * scratch.stokes_fe_values.JxW(q);
                data.local_stiffness_matrix(i,j)
                    += (
                          scalar_product(scratch.grad_phi_v[i], scratch.grad_phi_v[j])
                        + (parameters.assemble_schur_complement?
                                scratch.grad_phi_p[i] * scratch.grad_phi_p[j] : 0)
                        ) * scratch.stokes_fe_values.JxW(q);
            }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_matrix(i,j) = data.local_matrix(j,i);
            data.local_stiffness_matrix(i,j) = data.local_stiffness_matrix(j,i);
        }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix(
        const StokesAssembly::CopyData::Matrix<dim> &data)
{
    stokes_constraints.distribute_local_to_global(
            data.local_matrix,
            data.local_dof_indices,
            stokes_matrix);

    const ConstraintMatrix &constraints_used
    = (parameters.assemble_schur_complement?
            stokes_laplace_constraints: stokes_constraints);

    constraints_used.distribute_local_to_global(
            data.local_stiffness_matrix,
            data.local_dof_indices,
            stokes_laplace_matrix);
}


template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_stokes_rhs(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        StokesAssembly::Scratch::RightHandSide<dim> &scratch,
        StokesAssembly::CopyData::RightHandSide<dim> &data)
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

    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    scratch.stokes_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&triangulation,
                     cell->level(),
                     cell->index(),
                     &temperature_dof_handler);
    scratch.temperature_fe_values.reinit(temperature_cell);

    data.local_rhs = 0;

    scratch.stokes_fe_values[velocity].get_function_values(old_stokes_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_stokes_solution,
                                                           scratch.old_old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_stokes_solution,
                                                             scratch.old_velocity_gradients);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_old_stokes_solution,
                                                             scratch.old_old_velocity_gradients);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_v[k] = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_v[k] = scratch.stokes_fe_values[velocity].gradient(k, q);
        }

        const Tensor<1,dim> time_derivative_velocity
            = alpha[1] * scratch.old_velocity_values[q]
                + alpha[2] * scratch.old_old_velocity_values[q];

        const Tensor<1,dim> nonlinear_term_velocity
            = beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];

        const Tensor<2,dim> linear_term_velocity
            = gamma[1] * scratch.old_velocity_gradients[q]
                + gamma[2] * scratch.old_old_velocity_gradients[q];

        const Tensor<1,dim> extrapolated_velocity
            = (timestep != 0 ?
                (scratch.old_velocity_values[q] * (1 + timestep/old_timestep)
                        - scratch.old_old_velocity_values[q] * timestep/old_timestep)
                        : scratch.old_velocity_values[q]);
        const double extrapolated_temperature
            = (timestep != 0 ?
                (scratch.old_temperature_values[q] * (1 + timestep/old_timestep)
                        - scratch.old_old_temperature_values[q] * timestep/old_timestep)
                        : scratch.old_temperature_values[q]);

        const Tensor<1,dim> gravity_vector = EquationData::GravityVector<dim>().value(scratch.stokes_fe_values.quadrature_point(q));

        Tensor<1,dim>   coriolis_term;
        if (parameters.rotation)
        {
            if (dim == 2)
                coriolis_term = cross_product_2d(extrapolated_velocity);
            else if (dim == 3)
                coriolis_term = cross_product_3d(rotation_vector,
                                                 extrapolated_velocity);
            else
            {
                Assert(false, ExcInternalError());
            }
        }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            data.local_rhs(i)
                += (
                    - time_derivative_velocity * scratch.phi_v[i]
                    - timestep * nonlinear_term_velocity * scratch.phi_v[i]
                    - timestep * equation_coefficients[1] * scalar_product(linear_term_velocity, scratch.grad_phi_v[i])
                    - timestep * (parameters.rotation ? equation_coefficients[0] * coriolis_term * scratch.phi_v[i]: 0)
                    - timestep * equation_coefficients[2] * extrapolated_temperature * gravity_vector * scratch.phi_v[i]
                    ) * scratch.stokes_fe_values.JxW(q);
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs(
        const StokesAssembly::CopyData::RightHandSide<dim> &data)
{
    stokes_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            stokes_rhs);
}

}  // namespace BuoyantFluid


template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_stokes_matrix(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        StokesAssembly::Scratch::Matrix<2> &scratch,
        StokesAssembly::CopyData::Matrix<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_stokes_matrix(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        StokesAssembly::Scratch::Matrix<3> &scratch,
        StokesAssembly::CopyData::Matrix<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_stokes_matrix(
        const StokesAssembly::CopyData::Matrix<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_stokes_matrix(
        const StokesAssembly::CopyData::Matrix<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_stokes_rhs(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        StokesAssembly::Scratch::RightHandSide<2> &scratch,
        StokesAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_stokes_rhs(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        StokesAssembly::Scratch::RightHandSide<3> &scratch,
        StokesAssembly::CopyData::RightHandSide<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_stokes_rhs(
        const StokesAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_stokes_rhs(
        const StokesAssembly::CopyData::RightHandSide<3> &data);
