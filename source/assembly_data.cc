/*
 * assembly_data.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include "assembly_data.h"

namespace TemperatureAssembly {

namespace Scratch {

template<int dim>
RightHandSide<dim>::RightHandSide(
        const FiniteElement<dim>    &temperature_fe,
        const Mapping<dim>          &mapping,
        const Quadrature<dim>       &temperature_quadrature,
        const UpdateFlags            temperature_update_flags,
        const FiniteElement<dim>    &stokes_fe,
        const UpdateFlags            stokes_update_flags,
        const std::vector<double>   &alpha,
        const std::vector<double>   &beta,
        const std::vector<double>   &gamma)
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
stokes_fe_values(mapping,
                 stokes_fe,
                 temperature_quadrature,
                 stokes_update_flags),
old_velocity_values(temperature_quadrature.size()),
old_old_velocity_values(temperature_quadrature.size()),
alpha(alpha),
beta(beta),
gamma(gamma),
dofs_per_cell(temperature_fe.dofs_per_cell),
n_q_points(temperature_quadrature.size()),
velocity(0)
{}

template<int dim>
RightHandSide<dim>::RightHandSide(
        const RightHandSide<dim> &scratch)
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
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
alpha(scratch.alpha),
beta(scratch.beta),
gamma(scratch.gamma),
dofs_per_cell(scratch.dofs_per_cell),
n_q_points(scratch.n_q_points),
velocity(0)
{}

template <int dim>
Matrix<dim>::Matrix(
        const FiniteElement<dim> &temperature_fe,
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
Matrix<dim>::Matrix(const Matrix<dim>   &scratch)
:
fe_values(scratch.fe_values.get_mapping(),
          scratch.fe_values.get_fe(),
          scratch.fe_values.get_quadrature(),
          scratch.fe_values.get_update_flags()),
phi(scratch.phi),
grad_phi(scratch.grad_phi)
{}

}  // namespace Scratch

namespace CopyData {

template <int dim>
RightHandSide<dim>::RightHandSide(
    const FiniteElement<dim> &temperature_fe)
:
local_rhs(temperature_fe.dofs_per_cell),
matrix_for_bc(temperature_fe.dofs_per_cell,
              temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
RightHandSide<dim>::RightHandSide(
    const RightHandSide<dim> &data)
:
local_rhs(data.local_rhs),
matrix_for_bc(data.matrix_for_bc),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
Matrix<dim>::Matrix(const FiniteElement<dim>    &temperature_fe)
:
local_mass_matrix(temperature_fe.dofs_per_cell,
                  temperature_fe.dofs_per_cell),
local_laplace_matrix(temperature_fe.dofs_per_cell,
                     temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix<dim>   &data)
:
local_mass_matrix(data.local_mass_matrix),
local_laplace_matrix(data.local_laplace_matrix),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace CopyData

}  // namespace TemperatureAssembly


namespace NavierStokesAssembly {

namespace Scratch {

template <int dim>
Matrix<dim>::Matrix(
        const FiniteElement<dim> &stokes_fe,
        const Mapping<dim>       &mapping,
        const Quadrature<dim>    &stokes_quadrature,
        const UpdateFlags         stokes_update_flags)
:
stokes_fe_values(mapping,
                 stokes_fe,
                 stokes_quadrature,
                 stokes_update_flags),
div_phi_velocity(stokes_fe.dofs_per_cell),
phi_velocity(stokes_fe.dofs_per_cell),
grad_phi_velocity(stokes_fe.dofs_per_cell),
phi_pressure(stokes_fe.dofs_per_cell),
grad_phi_pressure(dim * stokes_fe.base_element(0).dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix<dim>   &scratch)
:
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
div_phi_velocity(scratch.div_phi_velocity),
phi_velocity(scratch.phi_velocity),
grad_phi_velocity(scratch.grad_phi_velocity),
phi_pressure(scratch.phi_pressure),
grad_phi_pressure(scratch.grad_phi_pressure)
{}

template <int dim>
ConvectionMatrix<dim>::ConvectionMatrix
(const FiniteElement<dim>   &stokes_fe,
 const Mapping<dim>         &mapping,
 const Quadrature<dim>     &stokes_quadrature,
 const UpdateFlags          stokes_update_flags)
:
stokes_fe_values(mapping,
                 stokes_fe,
                 stokes_quadrature,
                 stokes_update_flags),
phi_velocity(stokes_fe.dofs_per_cell),
grad_phi_velocity(stokes_fe.dofs_per_cell),
old_velocity_values(stokes_quadrature.size()),
old_old_velocity_values(stokes_quadrature.size()),
old_velocity_divergences(stokes_quadrature.size()),
old_old_velocity_divergences(stokes_quadrature.size())
{}

template <int dim>
ConvectionMatrix<dim>::ConvectionMatrix
(const ConvectionMatrix<dim>   &scratch)
:
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
phi_velocity(scratch.phi_velocity),
grad_phi_velocity(scratch.grad_phi_velocity),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
old_velocity_divergences(scratch.old_velocity_divergences),
old_old_velocity_divergences(scratch.old_old_velocity_divergences)
{}


template <int dim>
RightHandSide<dim>::RightHandSide
(const FiniteElement<dim>   &stokes_fe,
 const Mapping<dim>         &mapping,
 const Quadrature<dim>      &stokes_quadrature,
 const UpdateFlags           stokes_update_flags,
 const FiniteElement<dim>   &temperature_fe,
 const UpdateFlags           temperature_update_flags,
 const std::vector<double>  &alpha,
 const std::vector<double>  &beta,
 const std::vector<double>  &gamma)
:
stokes_fe_values(mapping,
                 stokes_fe,
                 stokes_quadrature,
                 stokes_update_flags),
phi_velocity(stokes_fe.dofs_per_cell),
grad_phi_velocity(stokes_fe.dofs_per_cell),
old_velocity_values(stokes_quadrature.size()),
old_old_velocity_values(stokes_quadrature.size()),
old_velocity_gradients(stokes_quadrature.size()),
old_old_velocity_gradients(stokes_quadrature.size()),
temperature_fe_values(mapping,
                      temperature_fe,
                      stokes_quadrature,
                      temperature_update_flags),
old_temperature_values(stokes_quadrature.size()),
old_old_temperature_values(stokes_quadrature.size()),
alpha(alpha),
beta(beta),
gamma(gamma),
gravity_vectors(stokes_quadrature.size()),
dofs_per_cell(stokes_fe.dofs_per_cell),
n_q_points(stokes_quadrature.size()),
velocity(0)
{}

template <int dim>
RightHandSide<dim>::RightHandSide(const RightHandSide<dim>  &scratch)
:
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
phi_velocity(scratch.phi_velocity),
grad_phi_velocity(scratch.grad_phi_velocity),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
old_velocity_gradients(scratch.old_velocity_gradients),
old_old_velocity_gradients(scratch.old_velocity_gradients),
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
old_temperature_values(scratch.old_temperature_values),
old_old_temperature_values(scratch.old_old_temperature_values),
alpha(scratch.alpha),
beta(scratch.beta),
gamma(scratch.gamma),
gravity_vectors(scratch.gravity_vectors),
dofs_per_cell(scratch.dofs_per_cell),
n_q_points(scratch.n_q_points),
velocity(0)
{}

}  // namespace Scratch

namespace CopyData {

template <int dim>
Matrix<dim>::Matrix(const FiniteElement<dim>    &navier_stokes_fe)
:
local_matrix(navier_stokes_fe.dofs_per_cell,
             navier_stokes_fe.dofs_per_cell),
local_mass_matrix(navier_stokes_fe.dofs_per_cell,
                  navier_stokes_fe.dofs_per_cell),
local_laplace_matrix(navier_stokes_fe.dofs_per_cell,
                     navier_stokes_fe.dofs_per_cell),
local_dof_indices(navier_stokes_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix<dim>   &data)
:
local_matrix(data.local_matrix),
local_mass_matrix(data.local_mass_matrix),
local_laplace_matrix(data.local_laplace_matrix),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
ConvectionMatrix<dim>::ConvectionMatrix
(const FiniteElement<dim>    &stokes_fe)
:
local_matrix(stokes_fe.dofs_per_cell,
             stokes_fe.dofs_per_cell),
local_dof_indices(stokes_fe.dofs_per_cell)
{}

template <int dim>
ConvectionMatrix<dim>::ConvectionMatrix
(const ConvectionMatrix<dim>   &data)
:
local_matrix(data.local_matrix),
local_dof_indices(data.local_dof_indices)
{}


template <int dim>
RightHandSide<dim>::RightHandSide(const FiniteElement<dim> &stokes_fe)
:
local_rhs(stokes_fe.dofs_per_cell),
local_dof_indices(stokes_fe.dofs_per_cell)
{}

template <int dim>
RightHandSide<dim>::RightHandSide(const RightHandSide<dim> &data)
:
local_rhs(data.local_rhs),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace Copy

}  // namespace StokesAssembly

// explicit instantiation
template class TemperatureAssembly::Scratch::RightHandSide<2>;
template class TemperatureAssembly::Scratch::RightHandSide<3>;
template class TemperatureAssembly::Scratch::Matrix<2>;
template class TemperatureAssembly::Scratch::Matrix<3>;

template class TemperatureAssembly::CopyData::RightHandSide<2>;
template class TemperatureAssembly::CopyData::RightHandSide<3>;
template class TemperatureAssembly::CopyData::Matrix<2>;
template class TemperatureAssembly::CopyData::Matrix<3>;

template class NavierStokesAssembly::Scratch::Matrix<2>;
template class NavierStokesAssembly::Scratch::Matrix<3>;
template class NavierStokesAssembly::Scratch::ConvectionMatrix<2>;
template class NavierStokesAssembly::Scratch::ConvectionMatrix<3>;
template class NavierStokesAssembly::Scratch::RightHandSide<2>;
template class NavierStokesAssembly::Scratch::RightHandSide<3>;

template class NavierStokesAssembly::CopyData::Matrix<2>;
template class NavierStokesAssembly::CopyData::Matrix<3>;
template class NavierStokesAssembly::CopyData::ConvectionMatrix<2>;
template class NavierStokesAssembly::CopyData::ConvectionMatrix<3>;
template class NavierStokesAssembly::CopyData::RightHandSide<2>;
template class NavierStokesAssembly::CopyData::RightHandSide<3>;
