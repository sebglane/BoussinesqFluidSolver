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
void BuoyantFluidSolver<dim>::local_assemble_velocity_rhs(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::RightHandSide<dim> &scratch,
        NavierStokesAssembly::CopyData::RightHandSide<dim> &data)
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

    scratch.stokes_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&triangulation,
                     cell->level(),
                     cell->index(),
                     &temperature_dof_handler),
    pressure_cell(&triangulation,
                  cell->level(),
                  cell->index(),
                  &pressure_dof_handler);

    scratch.temperature_fe_values.reinit(temperature_cell);
    scratch.pressure_fe_values.reinit(pressure_cell);

    data.local_rhs = 0;

    scratch.stokes_fe_values[velocity].get_function_values(old_velocity_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_velocity_solution,
                                                           scratch.old_old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_velocity_solution,
                                                              scratch.old_velocity_gradients);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_old_velocity_solution,
                                                              scratch.old_old_velocity_gradients);

    scratch.pressure_fe_values.get_function_values(old_pressure_solution,
                                                   scratch.old_pressure_values);
    scratch.pressure_fe_values.get_function_values(old_phi_solution,
                                                   scratch.old_phi_values);
    scratch.pressure_fe_values.get_function_values(old_old_phi_solution,
                                                   scratch.old_old_phi_values);

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
            = alpha[1] / timestep * scratch.old_velocity_values[q]
                + alpha[2] / timestep * scratch.old_old_velocity_values[q];

        Tensor<1,dim> nonlinear_term_velocity;
        if (parameters.convective_discretization == ConvectiveDiscretizationType::Standard)
            nonlinear_term_velocity = beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                                    + beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];
        else if (parameters.convective_discretization == ConvectiveDiscretizationType::DivergenceForm)
            nonlinear_term_velocity = beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                                    + 0.5 * beta[0] * trace(scratch.old_velocity_gradients[q]) * scratch.old_velocity_values[q]
                                    + beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q]
                                    + 0.5 * beta[1] * trace(scratch.old_old_velocity_gradients[q]) * scratch.old_old_velocity_values[q];
        else
            AssertThrow(false, ExcNotImplemented());

        const Tensor<2,dim> linear_term_velocity
            = gamma[1] * scratch.old_velocity_gradients[q]
                + gamma[2] * scratch.old_old_velocity_gradients[q];

        const double extrapolated_pressure
            = ((timestep != 0 && timestep != 1)?
                   scratch.old_pressure_values[q]
                 - alpha[1]/alpha[0] * scratch.old_phi_values[q]
                 - alpha[2]/alpha[0] * scratch.old_old_phi_values[q]
                 : scratch.old_pressure_values[q]);

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

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            data.local_rhs(i)
                += (
                    - time_derivative_velocity * scratch.phi_velocity[i]
                    - nonlinear_term_velocity * scratch.phi_velocity[i]
                    + extrapolated_pressure * trace(scratch.grad_phi_velocity[i])
                    - equation_coefficients[1] * scalar_product(linear_term_velocity, scratch.grad_phi_velocity[i])
                    - (parameters.rotation ? equation_coefficients[0] * coriolis_term * scratch.phi_velocity[i]: 0)
                    - equation_coefficients[2] * extrapolated_temperature * gravity_vector * scratch.phi_velocity[i]
                    ) * scratch.stokes_fe_values.JxW(q);
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_velocity_rhs(
        const NavierStokesAssembly::CopyData::RightHandSide<dim> &data)
{
    velocity_constraints.distribute_local_to_global(data.local_rhs,
                                                    data.local_dof_indices,
                                                    velocity_rhs);
}

template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_pressure_rhs(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        PressureAssembly::Scratch::RightHandSide<dim> &scratch,
        PressureAssembly::CopyData::RightHandSide<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.pressure_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.pressure_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);

    scratch.pressure_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell(&triangulation,
                     cell->level(),
                     cell->index(),
                     &velocity_dof_handler);
    scratch.velocity_fe_values.reinit(velocity_cell);

    data.local_rhs = 0;

    scratch.velocity_fe_values[velocity].get_function_divergences(velocity_solution,
                                                                scratch.velocity_divergences);

    for (unsigned int q=0; q<n_q_points; ++q)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            data.local_rhs(i)
                += scratch.velocity_divergences[q] * scratch.pressure_fe_values.shape_value(i, q)
                   * scratch.pressure_fe_values.JxW(q);
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_pressure_rhs(
        const PressureAssembly::CopyData::RightHandSide<dim> &data)
{
    pressure_constraints.distribute_local_to_global(data.local_rhs,
                                                    data.local_dof_indices,
                                                    pressure_rhs);
}

}  // namespace BuoyantFluid

/*
 *
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
 *
 */

template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_velocity_rhs(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::RightHandSide<2> &scratch,
        NavierStokesAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_velocity_rhs(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        NavierStokesAssembly::Scratch::RightHandSide<3> &scratch,
        NavierStokesAssembly::CopyData::RightHandSide<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_velocity_rhs(
        const NavierStokesAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_velocity_rhs(
        const NavierStokesAssembly::CopyData::RightHandSide<3> &data);


template void BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_pressure_rhs(
        const typename DoFHandler<2>::active_cell_iterator &cell,
        PressureAssembly::Scratch::RightHandSide<2> &scratch,
        PressureAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_pressure_rhs(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        PressureAssembly::Scratch::RightHandSide<3> &scratch,
        PressureAssembly::CopyData::RightHandSide<3> &data);

template void BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_pressure_rhs(
        const PressureAssembly::CopyData::RightHandSide<2> &data);
template void BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_pressure_rhs(
        const PressureAssembly::CopyData::RightHandSide<3> &data);
