/*
b * local_magnetic_assemble.cc
 *
 *  Created on: Jun 28, 2019
 *      Author: sg
 */

#include "buoyant_fluid_solver.h"

namespace BuoyantFluid {

template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_magnetic_matrix
(const typename DoFHandler<dim>::active_cell_iterator   &cell,
 MagneticAssembly::Scratch::Matrix<dim>                 &scratch,
 MagneticAssembly::CopyData::Matrix<dim>                &data)
{
    const unsigned int dofs_per_cell = scratch.magnetic_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.magnetic_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    magnetic_field(0);
    const FEValuesExtractors::Scalar    pseudo_pressure(dim);

    scratch.magnetic_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_matrix = 0;
    data.local_mass_matrix = 0;
    data.local_laplace_matrix = 0;
    data.local_stabilization_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_magnetic_field[k]     = scratch.magnetic_fe_values[magnetic_field].value(k, q);
            scratch.curl_phi_magnetic_field[k]= scratch.magnetic_fe_values[magnetic_field].curl(k, q);
            scratch.div_phi_magnetic_field[k] = scratch.magnetic_fe_values[magnetic_field].divergence(k, q);

            scratch.phi_pseudo_pressure[k] = scratch.magnetic_fe_values[pseudo_pressure].value(k, q);
            scratch.grad_phi_pseudo_pressure[k] = scratch.magnetic_fe_values[pseudo_pressure].gradient(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += (
                        scratch.phi_magnetic_field[i] * scratch.phi_magnetic_field[j]
                      + scratch.phi_pseudo_pressure[i] * scratch.phi_pseudo_pressure[j]
                        ) * scratch.magnetic_fe_values.JxW(q);
                data.local_laplace_matrix(i,j)
                    += (
                        scratch.curl_phi_magnetic_field[i] * scratch.curl_phi_magnetic_field[j]
                      + scratch.grad_phi_pseudo_pressure[i] * scratch.grad_phi_pseudo_pressure[j]
                        ) * scratch.magnetic_fe_values.JxW(q);
                data.local_matrix(i,j)
                    += (
                          scratch.grad_phi_pseudo_pressure[i] * scratch.phi_magnetic_field[j]
                        + scratch.phi_magnetic_field[i] * scratch.grad_phi_pseudo_pressure[j]
                        ) * scratch.magnetic_fe_values.JxW(q);
            }
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                data.local_stabilization_matrix(i,j)
                    += (
                          tau[0] * scratch.div_phi_magnetic_field[i] * scratch.div_phi_magnetic_field[j]
                        + tau[1] * cell->diameter() * cell->diameter() * scratch.grad_phi_pseudo_pressure[i] * scratch.grad_phi_pseudo_pressure[j]
                        + tau[1] * cell->diameter() * cell->diameter() * scratch.phi_magnetic_field[i] * scratch.grad_phi_pseudo_pressure[j]
                        ) * scratch.magnetic_fe_values.JxW(q);
        }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_matrix(i,j) = data.local_matrix(j,i);
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_laplace_matrix(i,j) = data.local_laplace_matrix(j,i);
        }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_magnetic_matrix
(const MagneticAssembly::CopyData::Matrix<dim> &data)
{
    magnetic_constraints.distribute_local_to_global(
            data.local_matrix,
            data.local_dof_indices,
            magnetic_matrix);
    magnetic_constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            magnetic_mass_matrix);
    magnetic_constraints.distribute_local_to_global(
            data.local_laplace_matrix,
            data.local_dof_indices,
            magnetic_laplace_matrix);
    magnetic_constraints.distribute_local_to_global(
            data.local_stabilization_matrix,
            data.local_dof_indices,
            magnetic_stabilization_matrix);
}

template<>
void BuoyantFluidSolver<2>::local_assemble_magnetic_rhs
(const typename DoFHandler<2>::active_cell_iterator   &cell,
 MagneticAssembly::Scratch::RightHandSide<2>          &scratch,
 MagneticAssembly::CopyData::RightHandSide<2>         &data)
{
    scratch.magnetic_fe_values.reinit(cell);

    typename DoFHandler<2>::active_cell_iterator
    stokes_cell(&triangulation,
                cell->level(),
                cell->index(),
                &navier_stokes_dof_handler);
    scratch.stokes_fe_values.reinit(stokes_cell);

    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_values
    (old_magnetic_solution, scratch.old_magnetic_values);
    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_values
    (old_old_magnetic_solution, scratch.old_old_magnetic_values);

    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_curls
    (old_magnetic_solution, scratch.old_magnetic_curls);
    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_curls
    (old_old_magnetic_solution, scratch.old_old_magnetic_curls);

    scratch.stokes_fe_values[scratch.velocity].get_function_values
    (old_navier_stokes_solution, scratch.old_velocity_values);
    scratch.stokes_fe_values[scratch.velocity].get_function_values
    (old_old_navier_stokes_solution, scratch.old_old_velocity_values);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_rhs= 0;

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int k=0; k<scratch.dofs_per_cell; ++k)
        {
            scratch.phi_magnetic_field[k]     = scratch.magnetic_fe_values[scratch.magnetic_field].value(k, q);
            scratch.curl_phi_magnetic_field[k]= scratch.magnetic_fe_values[scratch.magnetic_field].curl(k, q);
        }

        const Tensor<1,2> time_derivative_magnetic_field
            = scratch.alpha[1] / timestep * scratch.old_magnetic_values[q]
                + scratch.alpha[2] / timestep * scratch.old_old_magnetic_values[q];

        const typename FEValuesViews::Vector<2>::curl_type
        linear_term_magnetic_field
            = scratch.gamma[1] * scratch.old_magnetic_curls[q]
                + scratch.gamma[2] * scratch.old_old_magnetic_curls[q];

        for (unsigned int i=0; i<scratch.dofs_per_cell; ++i)
            data.local_rhs(i) += (
                    - time_derivative_magnetic_field * scratch.phi_magnetic_field[i]
                    - equation_coefficients[5] * linear_term_magnetic_field * scratch.curl_phi_magnetic_field[i]
                    ) * scratch.magnetic_fe_values.JxW(q);
    }
}


template<>
void BuoyantFluidSolver<3>::local_assemble_magnetic_rhs
(const typename DoFHandler<3>::active_cell_iterator   &cell,
 MagneticAssembly::Scratch::RightHandSide<3>          &scratch,
 MagneticAssembly::CopyData::RightHandSide<3>         &data)
{
    scratch.magnetic_fe_values.reinit(cell);

    typename DoFHandler<3>::active_cell_iterator
    stokes_cell(&triangulation,
                cell->level(),
                cell->index(),
                &navier_stokes_dof_handler);
    scratch.stokes_fe_values.reinit(stokes_cell);

    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_values
    (old_magnetic_solution, scratch.old_magnetic_values);
    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_values
    (old_old_magnetic_solution, scratch.old_old_magnetic_values);

    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_curls
    (old_magnetic_solution, scratch.old_magnetic_curls);
    scratch.magnetic_fe_values[scratch.magnetic_field].get_function_curls
    (old_old_magnetic_solution, scratch.old_old_magnetic_curls);

    scratch.stokes_fe_values[scratch.velocity].get_function_values
    (old_navier_stokes_solution, scratch.old_velocity_values);
    scratch.stokes_fe_values[scratch.velocity].get_function_values
    (old_old_navier_stokes_solution, scratch.old_old_velocity_values);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_rhs= 0;

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int k=0; k<scratch.dofs_per_cell; ++k)
        {
            scratch.phi_magnetic_field[k]     = scratch.magnetic_fe_values[scratch.magnetic_field].value(k, q);
            scratch.curl_phi_magnetic_field[k]= scratch.magnetic_fe_values[scratch.magnetic_field].curl(k, q);
        }

        const Tensor<1,3> time_derivative_magnetic_field
            = scratch.alpha[1] / timestep * scratch.old_magnetic_values[q]
                + scratch.alpha[2] / timestep * scratch.old_old_magnetic_values[q];

        const typename FEValuesViews::Vector<3>::curl_type
        linear_term_magnetic_field
            = scratch.gamma[1] * scratch.old_magnetic_curls[q]
                + scratch.gamma[2] * scratch.old_old_magnetic_curls[q];

        const Tensor<1,3>
        nonlinear_term_magnetic_field
            = scratch.beta[0] * cross_product_3d(scratch.old_velocity_values[q],
                                                 scratch.old_magnetic_values[q])
            + scratch.beta[1] * cross_product_3d(scratch.old_old_velocity_values[q],
                                                 scratch.old_old_magnetic_values[q]);

        for (unsigned int i=0; i<scratch.dofs_per_cell; ++i)
            data.local_rhs(i) += (
                    - time_derivative_magnetic_field * scratch.phi_magnetic_field[i]
                    + nonlinear_term_magnetic_field * scratch.curl_phi_magnetic_field[i]
                    - equation_coefficients[5] * linear_term_magnetic_field * scratch.curl_phi_magnetic_field[i]
                    ) * scratch.magnetic_fe_values.JxW(q);
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_magnetic_rhs
(const MagneticAssembly::CopyData::RightHandSide<dim> &data)
{
    magnetic_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            magnetic_rhs);
}
}  // namespace BuoyantFluid


template void
BuoyantFluid::BuoyantFluidSolver<2>::local_assemble_magnetic_matrix
(const typename DoFHandler<2>::active_cell_iterator &,
 MagneticAssembly::Scratch::Matrix<2>   &,
 MagneticAssembly::CopyData::Matrix<2>  &);
template void
BuoyantFluid::BuoyantFluidSolver<3>::local_assemble_magnetic_matrix
(const typename DoFHandler<3>::active_cell_iterator   &,
 MagneticAssembly::Scratch::Matrix<3>   &,
 MagneticAssembly::CopyData::Matrix<3>  &);

template void
BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_magnetic_matrix
(const MagneticAssembly::CopyData::Matrix<2> &);
template void
BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_magnetic_matrix
(const MagneticAssembly::CopyData::Matrix<3> &);

template void
BuoyantFluid::BuoyantFluidSolver<2>::copy_local_to_global_magnetic_rhs
(const MagneticAssembly::CopyData::RightHandSide<2> &);
template void
BuoyantFluid::BuoyantFluidSolver<3>::copy_local_to_global_magnetic_rhs
(const MagneticAssembly::CopyData::RightHandSide<3> &);
