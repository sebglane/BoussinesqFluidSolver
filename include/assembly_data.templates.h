/*
 * assembly_data.templates.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_DATA_TEMPLATES_H_
#define INCLUDE_ASSEMBLY_DATA_TEMPLATES_H_

#include "assembly_data.h"

namespace TemperatureAssembly {

namespace Scratch {

template <int dim>
Matrix<dim>::Matrix(
        const FiniteElement<dim> &temperature_fe,
        const Mapping<dim>       &mapping,
        const Quadrature<dim>    &temperature_quadrature)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      update_values|
                      update_gradients|
                      update_JxW_values),
phi_T(temperature_fe.dofs_per_cell),
grad_phi_T(temperature_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_T(scratch.phi_T),
grad_phi_T(scratch.grad_phi_T)
{}


template<int dim>
RightHandSide<dim>::RightHandSide(
        const FiniteElement<dim>    &temperature_fe,
        const Mapping<dim>          &mapping,
        const Quadrature<dim>       &temperature_quadrature,
        const UpdateFlags            temperature_update_flags,
        const FiniteElement<dim> &stokes_fe,
        const UpdateFlags         stokes_update_flags)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      temperature_update_flags),
phi_T(temperature_fe.dofs_per_cell),
grad_phi_T(temperature_fe.dofs_per_cell),
old_temperature_values(temperature_quadrature.size()),
old_old_temperature_values(temperature_quadrature.size()),
old_temperature_gradients(temperature_quadrature.size()),
old_old_temperature_gradients(temperature_quadrature.size()),
stokes_fe_values(mapping,
                 stokes_fe,
                 temperature_quadrature,
                 stokes_update_flags),
old_velocity_values(temperature_quadrature.size()),
old_old_velocity_values(temperature_quadrature.size())
{}

template<int dim>
RightHandSide<dim>::RightHandSide(
        const RightHandSide<dim> &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_T(scratch.phi_T),
grad_phi_T(scratch.grad_phi_T),
old_temperature_values(scratch.old_temperature_values),
old_old_temperature_values(scratch.old_old_temperature_values),
old_temperature_gradients(scratch.old_temperature_gradients),
old_old_temperature_gradients(scratch.old_old_temperature_gradients),
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values)
{}

}  // namespace Scratch

namespace CopyData {

template <int dim>
Matrix<dim>::Matrix(const FiniteElement<dim> &temperature_fe)
:
local_mass_matrix(temperature_fe.dofs_per_cell),
local_stiffness_matrix(temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix<dim> &data)
:
local_mass_matrix(data.local_mass_matrix),
local_stiffness_matrix(data.local_stiffness_matrix),
local_dof_indices(data.local_dof_indices)
{}

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

}  // namespace CopyData

}  // namespace TemperatureAssembly


namespace StokesAssembly {

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
div_phi_v(stokes_fe.dofs_per_cell),
phi_v(stokes_fe.dofs_per_cell),
grad_phi_v(stokes_fe.dofs_per_cell),
phi_p(stokes_fe.dofs_per_cell),
grad_phi_p(stokes_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix<dim> &scratch)
:
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
div_phi_v(scratch.div_phi_v),
phi_v(scratch.phi_v),
grad_phi_v(scratch.grad_phi_v),
phi_p(scratch.phi_p),
grad_phi_p(scratch.grad_phi_p)
{}

template <int dim>
RightHandSide<dim>::RightHandSide(
        const FiniteElement<dim> &stokes_fe,
        const Mapping<dim>       &mapping,
        const Quadrature<dim>    &stokes_quadrature,
        const UpdateFlags         stokes_update_flags,
        const FiniteElement<dim> &temperature_fe,
        const UpdateFlags         temperature_update_flags)
:
stokes_fe_values(mapping,
                 stokes_fe,
                 stokes_quadrature,
                 stokes_update_flags),
phi_v(stokes_fe.dofs_per_cell),
grad_phi_v(stokes_fe.dofs_per_cell),
old_velocity_values(stokes_quadrature.size()),
old_old_velocity_values(stokes_quadrature.size()),
old_velocity_gradients(stokes_quadrature.size()),
old_old_velocity_gradients(stokes_quadrature.size()),
temperature_fe_values(mapping,
                      temperature_fe,
                      stokes_quadrature,
                      temperature_update_flags),
old_temperature_values(stokes_quadrature.size()),
old_old_temperature_values(stokes_quadrature.size())
{}

template <int dim>
RightHandSide<dim>::RightHandSide(const RightHandSide<dim> &scratch)
:
stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                 scratch.stokes_fe_values.get_fe(),
                 scratch.stokes_fe_values.get_quadrature(),
                 scratch.stokes_fe_values.get_update_flags()),
phi_v(scratch.phi_v),
grad_phi_v(scratch.grad_phi_v),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
old_velocity_gradients(scratch.old_velocity_gradients),
old_old_velocity_gradients(scratch.old_velocity_gradients),
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
old_temperature_values(scratch.old_temperature_values),
old_old_temperature_values(scratch.old_old_temperature_values)
{}

}  // namespace Scratch

namespace CopyData {


template <int dim>
Matrix<dim>::Matrix(const FiniteElement<dim> &temperature_fe)
:
local_matrix(temperature_fe.dofs_per_cell,
                  temperature_fe.dofs_per_cell),
local_stiffness_matrix(temperature_fe.dofs_per_cell,
                       temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
Matrix<dim>::Matrix(const Matrix<dim> &data)
:
local_matrix(data.local_matrix),
local_stiffness_matrix(data.local_stiffness_matrix),
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



#endif /* INCLUDE_ASSEMBLY_DATA_TEMPLATES_H_ */
