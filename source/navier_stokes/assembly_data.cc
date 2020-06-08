/*
 * assembly_data.cc
 *
 *  Created on: Jul 30, 2019
 *      Author: sg
 */

#include <adsolic/navier_stokes_solver.h>

namespace adsolic
{

namespace NavierStokesAssembly
{

namespace Scratch {

template <int dim>
PressureMatrix<dim>::PressureMatrix
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
grad_phi(fe.dofs_per_cell),
n_q_points(quadrature.size())
{}

template <int dim>
PressureMatrix<dim>::PressureMatrix
(const PressureMatrix<dim>   &scratch)
:
fe_values(scratch.fe_values.get_mapping(),
          scratch.fe_values.get_fe(),
          scratch.fe_values.get_quadrature(),
          scratch.fe_values.get_update_flags()),
phi(scratch.phi),
grad_phi(scratch.grad_phi),
n_q_points(scratch.n_q_points)
{}

template <int dim>
VelocityMatrix<dim>::VelocityMatrix
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
grad_phi(fe.dofs_per_cell),
n_q_points(quadrature.size()),
velocity(0)
{}

template <int dim>
VelocityMatrix<dim>::VelocityMatrix
(const VelocityMatrix<dim>   &scratch)
:
fe_values(scratch.fe_values.get_mapping(),
          scratch.fe_values.get_fe(),
          scratch.fe_values.get_quadrature(),
          scratch.fe_values.get_update_flags()),
phi(scratch.phi),
grad_phi(scratch.grad_phi),
n_q_points(scratch.n_q_points),
velocity(scratch.velocity.first_vector_component)
{}

template <int dim>
VelocityDiffusion<dim>::VelocityDiffusion
(const FiniteElement<dim>  &velocity_fe,
 const FiniteElement<dim>  &pressure_fe,
 const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature,
 const UpdateFlags          velocity_update_flags,
 const UpdateFlags          pressure_update_flags,
 const std::array<double,3>&alpha,
 const std::array<double,2>&beta,
 const std::array<double,3>&gamma)
:
fe_values_velocity(mapping,
                   velocity_fe,
                   quadrature,
                   velocity_update_flags),
fe_values_pressure(mapping,
                   pressure_fe,
                   quadrature,
                   pressure_update_flags),
phi_velocity(velocity_fe.dofs_per_cell),
grad_phi_velocity(velocity_fe.dofs_per_cell),
div_phi_velocity(velocity_fe.dofs_per_cell),
old_velocity_values(quadrature.size()),
old_old_velocity_values(quadrature.size()),
old_velocity_gradients(quadrature.size()),
old_old_velocity_gradients(quadrature.size()),
old_pressure_values(quadrature.size()),
pressure_update_values(quadrature.size()),
old_pressure_update_values(quadrature.size()),
alpha(alpha),
beta(beta),
gamma(gamma),
n_q_points(quadrature.size()),
velocity(0)
{}

template <int dim>
VelocityDiffusion<dim>::VelocityDiffusion
(const VelocityDiffusion<dim>  &scratch)
:
fe_values_velocity(scratch.fe_values_velocity.get_mapping(),
                   scratch.fe_values_velocity.get_fe(),
                   scratch.fe_values_velocity.get_quadrature(),
                   scratch.fe_values_velocity.get_update_flags()),
fe_values_pressure(scratch.fe_values_pressure.get_mapping(),
                   scratch.fe_values_pressure.get_fe(),
                   scratch.fe_values_pressure.get_quadrature(),
                   scratch.fe_values_pressure.get_update_flags()),
phi_velocity(scratch.phi_velocity),
grad_phi_velocity(scratch.grad_phi_velocity),
div_phi_velocity(scratch.div_phi_velocity),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
old_velocity_gradients(scratch.old_velocity_gradients),
old_old_velocity_gradients(scratch.old_velocity_gradients),
old_pressure_values(scratch.old_pressure_values),
pressure_update_values(scratch.pressure_update_values),
old_pressure_update_values(scratch.old_pressure_update_values),
alpha(scratch.alpha),
beta(scratch.beta),
gamma(scratch.gamma),
n_q_points(scratch.n_q_points),
velocity(scratch.velocity.first_vector_component)
{}


template <int dim>
PressureProjection<dim>::PressureProjection
(const FiniteElement<dim>  &velocity_fe,
 const FiniteElement<dim>  &pressure_fe,
 const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature,
 const UpdateFlags          velocity_update_flags,
 const UpdateFlags          pressure_update_flags,
 const std::array<double,3>&alpha)
:
fe_values_velocity(mapping,
                   velocity_fe,
                   quadrature,
                   velocity_update_flags),
fe_values_pressure(mapping,
                   pressure_fe,
                   quadrature,
                   pressure_update_flags),
phi_pressure(pressure_fe.dofs_per_cell),
grad_phi_pressure(pressure_fe.dofs_per_cell),
velocity_divergences(quadrature.size()),
alpha(alpha),
n_q_points(quadrature.size()),
velocity(0)
{}

template <int dim>
PressureProjection<dim>::PressureProjection
(const PressureProjection<dim>  &scratch)
:
fe_values_velocity(scratch.fe_values_velocity.get_mapping(),
                   scratch.fe_values_velocity.get_fe(),
                   scratch.fe_values_velocity.get_quadrature(),
                   scratch.fe_values_velocity.get_update_flags()),
fe_values_pressure(scratch.fe_values_pressure.get_mapping(),
                   scratch.fe_values_pressure.get_fe(),
                   scratch.fe_values_pressure.get_quadrature(),
                   scratch.fe_values_pressure.get_update_flags()),
phi_pressure(scratch.phi_pressure),
grad_phi_pressure(scratch.grad_phi_pressure),
velocity_divergences(scratch.velocity_divergences),
alpha(scratch.alpha),
n_q_points(scratch.n_q_points),
velocity(scratch.velocity.first_vector_component)
{}

template <int dim>
VelocityCorrection<dim>::VelocityCorrection
(const FiniteElement<dim>  &velocity_fe,
 const FiniteElement<dim>  &pressure_fe,
 const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature,
 const UpdateFlags          velocity_update_flags,
 const UpdateFlags          pressure_update_flags,
 const std::array<double,3>&alpha)
:
fe_values_velocity(mapping,
                   velocity_fe,
                   quadrature,
                   velocity_update_flags),
fe_values_pressure(mapping,
                   pressure_fe,
                   quadrature,
                   pressure_update_flags),
phi_velocity(velocity_fe.dofs_per_cell),
tentative_velocity_values(quadrature.size()),
pressure_gradients(quadrature.size()),
alpha(alpha),
n_q_points(quadrature.size()),
velocity(0)
{}

template <int dim>
VelocityCorrection<dim>::VelocityCorrection
(const VelocityCorrection<dim>  &scratch)
:
fe_values_velocity(scratch.fe_values_velocity.get_mapping(),
                   scratch.fe_values_velocity.get_fe(),
                   scratch.fe_values_velocity.get_quadrature(),
                   scratch.fe_values_velocity.get_update_flags()),
fe_values_pressure(scratch.fe_values_pressure.get_mapping(),
                   scratch.fe_values_pressure.get_fe(),
                   scratch.fe_values_pressure.get_quadrature(),
                   scratch.fe_values_pressure.get_update_flags()),
phi_velocity(scratch.phi_velocity),
tentative_velocity_values(scratch.tentative_velocity_values),
pressure_gradients(scratch.pressure_gradients),
alpha(scratch.alpha),
n_q_points(scratch.n_q_points),
velocity(scratch.velocity.first_vector_component)
{}

}  // namespace Scratch

namespace CopyData {

template <int dim>
Matrix<dim>::Matrix
(const FiniteElement<dim>    &fe)
:
dofs_per_cell(fe.dofs_per_cell),
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
dofs_per_cell(data.dofs_per_cell),
local_mass_matrix(data.local_mass_matrix),
local_laplace_matrix(data.local_laplace_matrix),
local_dof_indices(data.local_dof_indices)
{}


template <int dim>
RightHandSides<dim>::RightHandSides
(const FiniteElement<dim>  &fe,
 const ConstraintMatrix    &constraints)
:
constraints(constraints),
dofs_per_cell(fe.dofs_per_cell),
local_matrix_for_bc(fe.dofs_per_cell,
                    fe.dofs_per_cell),
local_rhs(fe.dofs_per_cell),
local_dof_indices(fe.dofs_per_cell)
{}

template <int dim>
RightHandSides<dim>::RightHandSides
(const RightHandSides<dim> &data)
:
constraints(data.constraints),
dofs_per_cell(data.dofs_per_cell),
local_matrix_for_bc(data.local_matrix_for_bc),
local_rhs(data.local_rhs),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace Copy

// explicit instantiation
template struct Scratch::PressureMatrix<2>;
template class Scratch::PressureMatrix<3>;

template class Scratch::VelocityMatrix<2>;
template class Scratch::VelocityMatrix<3>;

template class Scratch::VelocityDiffusion<2>;
template class Scratch::VelocityDiffusion<3>;

template class Scratch::PressureProjection<2>;
template class Scratch::PressureProjection<3>;

template class Scratch::VelocityCorrection<2>;
template class Scratch::VelocityCorrection<3>;

template class CopyData::Matrix<2>;
template class CopyData::Matrix<3>;

template class CopyData::RightHandSides<2>;
template class CopyData::RightHandSides<3>;

}  // namespace NavierStokesAssembly

}  // namespace adsolic



