/*
 * assembly_data.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include <adsolic/temperature_solver.h>

namespace adsolic
{

namespace TemperatureAssembly
{

namespace Scratch
{

template<int dim>
RightHandSide<dim>::RightHandSide
(const FiniteElement<dim>   &temperature_fe,
 const Mapping<dim>         &mapping,
 const Quadrature<dim>      &temperature_quadrature,
 const UpdateFlags           temperature_update_flags,
 TensorFunction<1,dim>      &advection_field,
 const std::array<double,3> &alpha,
 const std::array<double,2> &beta,
 const std::array<double,3> &gamma)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      temperature_update_flags),
phi_temperature(temperature_fe.dofs_per_cell),
grad_phi_temperature(temperature_fe.dofs_per_cell),
old_temperature_values(temperature_quadrature.size()),
old_old_temperature_values(temperature_quadrature.size()),
old_temperature_gradients(temperature_quadrature.size()),
old_old_temperature_gradients(temperature_quadrature.size()),
/*
stokes_fe_values(mapping,
                 stokes_fe,
                 temperature_quadrature,
                 stokes_update_flags),
*/
advection_field(advection_field),
old_velocity_values(temperature_quadrature.size()),
old_old_velocity_values(temperature_quadrature.size()),
/*
velocity(first_velocity_component),
*/
alpha(alpha),
beta(beta),
gamma(gamma),
dofs_per_cell(temperature_fe.dofs_per_cell),
n_q_points(temperature_quadrature.size())
{}

template<int dim>
RightHandSide<dim>::RightHandSide
(const RightHandSide<dim> &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_temperature(scratch.phi_temperature),
grad_phi_temperature(scratch.grad_phi_temperature),
old_temperature_values(scratch.old_temperature_values),
old_old_temperature_values(scratch.old_old_temperature_values),
old_temperature_gradients(scratch.old_temperature_gradients),
old_old_temperature_gradients(scratch.old_old_temperature_gradients),
/*
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
 */
advection_field(scratch.advection_field),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
/*
velocity(scratch.velocity.first_vector_component),
*/
alpha(scratch.alpha),
beta(scratch.beta),
gamma(scratch.gamma),
dofs_per_cell(scratch.dofs_per_cell),
n_q_points(scratch.n_q_points)
{}

template <int dim>
Matrix<dim>::Matrix
(const FiniteElement<dim> &temperature_fe,
 const Mapping<dim>       &mapping,
 const Quadrature<dim>    &temperature_quadrature,
 const UpdateFlags         temperature_update_flags)
:
fe_values(mapping,
          temperature_fe,
          temperature_quadrature,
          temperature_update_flags),
phi(temperature_fe.dofs_per_cell),
grad_phi(temperature_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix
(const Matrix<dim>   &scratch)
:
fe_values(scratch.fe_values.get_mapping(),
          scratch.fe_values.get_fe(),
          scratch.fe_values.get_quadrature(),
          scratch.fe_values.get_update_flags()),
phi(scratch.phi),
grad_phi(scratch.grad_phi)
{}

// explicit instantiation
template class RightHandSide<2>;
template class RightHandSide<3>;
template class Matrix<2>;
template class Matrix<3>;

}  // namespace Scratch

namespace CopyData {

template <int dim>
RightHandSide<dim>::RightHandSide
(const FiniteElement<dim> &temperature_fe)
:
local_rhs(temperature_fe.dofs_per_cell),
matrix_for_bc(temperature_fe.dofs_per_cell,
              temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
RightHandSide<dim>::RightHandSide
(const RightHandSide<dim> &data)
:
local_rhs(data.local_rhs),
matrix_for_bc(data.matrix_for_bc),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
Matrix<dim>::Matrix
(const FiniteElement<dim>    &temperature_fe)
:
local_mass_matrix(temperature_fe.dofs_per_cell,
                  temperature_fe.dofs_per_cell),
local_laplace_matrix(temperature_fe.dofs_per_cell,
                     temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix
(const Matrix<dim>   &data)
:
local_mass_matrix(data.local_mass_matrix),
local_laplace_matrix(data.local_laplace_matrix),
local_dof_indices(data.local_dof_indices)
{}

// explicit instantiation
template class RightHandSide<2>;
template class RightHandSide<3>;
template class Matrix<2>;
template class Matrix<3>;

}  // namespace CopyData

}  // namespace TemperatureAssembly



}  // namespace adsolic
