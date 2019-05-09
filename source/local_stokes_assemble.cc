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
void BuoyantFluidSolver<dim>::local_assemble_stokes_matrix
(const typename DoFHandler<dim>::active_cell_iterator   &cell,
 NavierStokesAssembly::Scratch::Matrix<dim>             &scratch,
 NavierStokesAssembly::CopyData::Matrix<dim>            &data)
{
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    scratch.stokes_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_matrix = 0;
    data.local_mass_matrix = 0;
    data.local_laplace_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_velocity[k]     = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_velocity[k]= scratch.stokes_fe_values[velocity].gradient(k, q);
            scratch.div_phi_velocity[k] = trace(scratch.grad_phi_velocity[k]);
            scratch.phi_pressure[k]     = scratch.stokes_fe_values[pressure].value(k, q);
            scratch.grad_phi_pressure[k]= scratch.stokes_fe_values[pressure].gradient(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += (
                        scratch.phi_velocity[i] * scratch.phi_velocity[j]
                      + scratch.phi_pressure[i] * scratch.phi_pressure[j]
                        ) * scratch.stokes_fe_values.JxW(q);
                data.local_laplace_matrix(i,j)
                    += (
                        scalar_product(scratch.grad_phi_velocity[i], scratch.grad_phi_velocity[j])
                      + scratch.grad_phi_pressure[i] * scratch.grad_phi_pressure[j]
                        ) * scratch.stokes_fe_values.JxW(q);
            }
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                data.local_matrix(i,j)
                    += (
                        - scratch.div_phi_velocity[i] * scratch.phi_pressure[j]
                        + scratch.grad_phi_pressure[i] * scratch.phi_velocity[j]
                        ) * scratch.stokes_fe_values.JxW(q);
        }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_laplace_matrix(i,j) = data.local_laplace_matrix(j,i);
        }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix(
        const NavierStokesAssembly::CopyData::Matrix<dim> &data)
{
    navier_stokes_constraints.distribute_local_to_global(
            data.local_matrix,
            data.local_dof_indices,
            navier_stokes_matrix);
    navier_stokes_constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            navier_stokes_mass_matrix);
    stokes_pressure_constraints.distribute_local_to_global(
            data.local_laplace_matrix,
            data.local_dof_indices,
            navier_stokes_laplace_matrix);
}

template<int dim>
void BuoyantFluidSolver<dim>::local_assemble_stokes_convection_matrix
(const typename DoFHandler<dim>::active_cell_iterator   &cell,
 NavierStokesAssembly::Scratch::ConvectionMatrix<dim>   &scratch,
 NavierStokesAssembly::CopyData::ConvectionMatrix<dim>  &data)
{
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);

    scratch.stokes_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_matrix = 0;

    scratch.stokes_fe_values[velocity].get_function_values(old_navier_stokes_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_navier_stokes_solution,
                                                           scratch.old_old_velocity_values);
    if (parameters.convective_weak_form == ConvectiveWeakForm::DivergenceForm)
    {
        scratch.stokes_fe_values[velocity].get_function_divergences(old_navier_stokes_solution,
                                                                    scratch.old_velocity_divergences);
        scratch.stokes_fe_values[velocity].get_function_divergences(old_old_navier_stokes_solution,
                                                                    scratch.old_old_velocity_divergences);
    }

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_velocity[k]     = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_velocity[k]= scratch.stokes_fe_values[velocity].gradient(k, q);
        }

        const Tensor<1,dim> extrapolated_velocity
        = (timestep != 0 ?
                (scratch.old_velocity_values[q] * (1 + timestep/old_timestep)
                        - scratch.old_old_velocity_values[q] * timestep/old_timestep)
                        : scratch.old_velocity_values[q]);

        switch (parameters.convective_weak_form)
        {
            case ConvectiveWeakForm::Standard:
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        data.local_matrix(i,j)
                            += scratch.grad_phi_velocity[j] * extrapolated_velocity
                             * scratch.phi_velocity[i] * scratch.stokes_fe_values.JxW(q);
                break;
            case ConvectiveWeakForm::DivergenceForm:
            {
                const double extrapolated_velocity_divergence
                = (timestep != 0 ?
                        (scratch.old_velocity_divergences[q] * (1 + timestep/old_timestep)
                                - scratch.old_old_velocity_divergences[q] * timestep/old_timestep)
                                : scratch.old_velocity_divergences[q]);

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        data.local_matrix(i,j)
                            +=(
                                  scratch.grad_phi_velocity[j] * extrapolated_velocity * scratch.phi_velocity[i]
                                + 0.5 * extrapolated_velocity_divergence * scratch.phi_velocity[j] * scratch.phi_velocity[i]
                              ) * scratch.stokes_fe_values.JxW(q);
                break;
            }
            case ConvectiveWeakForm::SkewSymmetric:
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        data.local_matrix(i,j)
                            +=(
                                  0.5 * scratch.grad_phi_velocity[j] * extrapolated_velocity * scratch.phi_velocity[i]
                                - 0.5 * scratch.grad_phi_velocity[i] * extrapolated_velocity * scratch.phi_velocity[j]
                              ) * scratch.stokes_fe_values.JxW(q);
                break;
            default:
                Assert(false, ExcNotImplemented());
                break;
        }
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_convection_matrix
(const NavierStokesAssembly::CopyData::ConvectionMatrix<dim>   &data)
{
    navier_stokes_constraints.distribute_local_to_global(
            data.local_matrix,
            data.local_dof_indices,
            navier_stokes_matrix);
}


template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_stokes_rhs(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::RightHandSide<dim> &scratch,
        NavierStokesAssembly::CopyData::RightHandSide<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);

    scratch.stokes_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&triangulation,
                     cell->level(),
                     cell->index(),
                     &temperature_dof_handler);
    scratch.temperature_fe_values.reinit(temperature_cell);

    data.local_rhs = 0;

    scratch.stokes_fe_values[velocity].get_function_values(old_navier_stokes_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_navier_stokes_solution,
                                                           scratch.old_old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_navier_stokes_solution,
                                                              scratch.old_velocity_gradients);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_old_navier_stokes_solution,
                                                              scratch.old_old_velocity_gradients);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_velocity[k]     = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_velocity[k]= scratch.stokes_fe_values[velocity].gradient(k, q);
        }

        const Tensor<1,dim> time_derivative_velocity
            = scratch.alpha[1] / timestep * scratch.old_velocity_values[q]
                + scratch.alpha[2] / timestep * scratch.old_old_velocity_values[q];

        const Tensor<2,dim> linear_term_velocity
            = scratch.gamma[1] * scratch.old_velocity_gradients[q]
                + scratch.gamma[2] * scratch.old_old_velocity_gradients[q];

        const double extrapolated_temperature
            = (timestep != 0 ?
                (scratch.old_temperature_values[q] * (1 + timestep/old_timestep)
                        - scratch.old_old_temperature_values[q] * timestep/old_timestep)
                        : scratch.old_temperature_values[q]);

        const Tensor<1,dim> gravity_vector = EquationData::GravityVector<dim>().value(scratch.stokes_fe_values.quadrature_point(q));

        Tensor<1,dim>   coriolis_term;
        if (parameters.rotation)
        {
            const Tensor<1,dim> extrapolated_velocity
            = (timestep != 0 ?
                    (scratch.old_velocity_values[q] * (1 + timestep/old_timestep)
                            - scratch.old_old_velocity_values[q] * timestep/old_timestep)
                            : scratch.old_velocity_values[q]);
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

        if (parameters.convective_scheme == ConvectiveDiscretizationType::Explicit)
        {
        Tensor<1,dim> nonlinear_term_velocity;
        bool skew = false;
        switch (parameters.convective_weak_form)
        {
        case ConvectiveWeakForm::Standard:
            nonlinear_term_velocity
                = scratch.beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + scratch.beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];
            break;
        case ConvectiveWeakForm::DivergenceForm:
            nonlinear_term_velocity
                = scratch.beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + 0.5 * scratch.beta[0] * trace(scratch.old_velocity_gradients[q]) * scratch.old_velocity_values[q]
                + scratch.beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q]
                + 0.5 * scratch.beta[1] * trace(scratch.old_old_velocity_gradients[q]) * scratch.old_old_velocity_values[q];
            break;
        case ConvectiveWeakForm::SkewSymmetric:
            nonlinear_term_velocity
                = 0.5 * scratch.beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + 0.5 * scratch.beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];
            skew = true;
            break;
        default:
            Assert(false, ExcNotImplemented());
            break;
        }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            data.local_rhs(i)
                += (
                    - time_derivative_velocity * scratch.phi_velocity[i]
                    - nonlinear_term_velocity * scratch.phi_velocity[i]
                    + (skew ? 0.5 * scratch.beta[0] * (scratch.grad_phi_velocity[i] * scratch.old_velocity_values[q]) * scratch.old_velocity_values[q]
                            + 0.5 * scratch.beta[1] * (scratch.grad_phi_velocity[i] * scratch.old_old_velocity_values[q]) * scratch.old_old_velocity_values[q]
                            : 0.0)
                    - equation_coefficients[1] * scalar_product(linear_term_velocity, scratch.grad_phi_velocity[i])
                    - (parameters.rotation ? equation_coefficients[0] * coriolis_term * scratch.phi_velocity[i]: 0)
                    - equation_coefficients[2] * extrapolated_temperature * gravity_vector * scratch.phi_velocity[i]
                    ) * scratch.stokes_fe_values.JxW(q);
        }
        else
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                data.local_rhs(i)
                    += (
                        - time_derivative_velocity * scratch.phi_velocity[i]
                        - equation_coefficients[1] * scalar_product(linear_term_velocity, scratch.grad_phi_velocity[i])
                        - (parameters.rotation ? equation_coefficients[0] * coriolis_term * scratch.phi_velocity[i]: 0)
                        - equation_coefficients[2] * extrapolated_temperature * gravity_vector * scratch.phi_velocity[i]
                        ) * scratch.stokes_fe_values.JxW(q);
        }
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs(
        const NavierStokesAssembly::CopyData::RightHandSide<dim> &data)
{
    navier_stokes_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            navier_stokes_rhs);
}
}  // namespace BuoyantFluid

template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_stokes_matrix(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::Matrix<2> &scratch,
        NavierStokesAssembly::CopyData::Matrix<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_stokes_matrix(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::Matrix<3> &scratch,
        NavierStokesAssembly::CopyData::Matrix<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_stokes_matrix(
        const NavierStokesAssembly::CopyData::Matrix<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_stokes_matrix(
        const NavierStokesAssembly::CopyData::Matrix<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_stokes_convection_matrix(
        const typename DoFHandler<2>::active_cell_iterator  &,
        NavierStokesAssembly::Scratch::ConvectionMatrix<2>  &,
        NavierStokesAssembly::CopyData::ConvectionMatrix<2> &);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_stokes_convection_matrix(
        const typename DoFHandler<3>::active_cell_iterator  &,
        NavierStokesAssembly::Scratch::ConvectionMatrix<3>  &,
        NavierStokesAssembly::CopyData::ConvectionMatrix<3> &);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_stokes_convection_matrix(
        const NavierStokesAssembly::CopyData::ConvectionMatrix<2>   &);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_stokes_convection_matrix(
        const NavierStokesAssembly::CopyData::ConvectionMatrix<3>   &);


template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_stokes_rhs(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::RightHandSide<2> &scratch,
        NavierStokesAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_stokes_rhs(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::RightHandSide<3> &scratch,
        NavierStokesAssembly::CopyData::RightHandSide<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_stokes_rhs(
        const NavierStokesAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_stokes_rhs(
        const NavierStokesAssembly::CopyData::RightHandSide<3> &data);
