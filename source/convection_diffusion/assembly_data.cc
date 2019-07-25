/*
 * assembly_data.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include <adsolic/convection_diffusion_solver.h>

namespace adsolic
{

namespace ConvectionDiffusionAssembly
{

namespace Scratch
{

template<int dim>
RightHandSide<dim>::RightHandSide
(const FiniteElement<dim>   &fe,
 const Mapping<dim>         &mapping,
 const Quadrature<dim>      &quadrature,
 const UpdateFlags           update_flags,
 const std::array<double,3> &alpha,
 const std::array<double,2> &beta,
 const std::array<double,3> &gamma)
:
fe_values(mapping,
          fe,
          quadrature,
          update_flags),
phi(fe.dofs_per_cell),
grad_phi(fe.dofs_per_cell),
old_values(quadrature.size()),
old_old_values(quadrature.size()),
old_gradients(quadrature.size()),
old_old_gradients(quadrature.size()),
/*
stokes_fe_values(mapping,
                 stokes_fe,
                 temperature_quadrature,
                 stokes_update_flags),
*/
old_velocity_values(quadrature.size()),
old_old_velocity_values(quadrature.size()),
/*
velocity(first_velocity_component),
*/
alpha(alpha),
beta(beta),
gamma(gamma),
dofs_per_cell(fe.dofs_per_cell),
n_q_points(quadrature.size())
{}

template<int dim>
RightHandSide<dim>::RightHandSide
(const RightHandSide<dim> &scratch)
:
fe_values(scratch.fe_values.get_mapping(),
                      scratch.fe_values.get_fe(),
                      scratch.fe_values.get_quadrature(),
                      scratch.fe_values.get_update_flags()),
phi(scratch.phi),
grad_phi(scratch.grad_phi),
old_values(scratch.old_values),
old_old_values(scratch.old_old_values),
old_gradients(scratch.old_gradients),
old_old_gradients(scratch.old_old_gradients),
/*
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
 */
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
(const FiniteElement<dim> &fe,
 const Mapping<dim>       &mapping,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags         update_flags)
:
fe_values(mapping,
          fe,
          quadrature,
          update_flags),
phi(fe.dofs_per_cell),
grad_phi(fe.dofs_per_cell)
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
(const FiniteElement<dim> &fe)
:
local_rhs(fe.dofs_per_cell),
matrix_for_bc(fe.dofs_per_cell,
              fe.dofs_per_cell),
local_dof_indices(fe.dofs_per_cell)
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
(const FiniteElement<dim>    &fe)
:
local_mass_matrix(fe.dofs_per_cell,
                  fe.dofs_per_cell),
local_laplace_matrix(fe.dofs_per_cell,
                     fe.dofs_per_cell),
local_dof_indices(fe.dofs_per_cell)
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

}  // namespace ConvectionDiffusionAssembly



}  // namespace adsolic
