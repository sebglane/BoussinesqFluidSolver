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
        const FiniteElement<dim> &stokes_fe,
        const UpdateFlags         stokes_update_flags)
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
old_old_velocity_values(scratch.old_old_velocity_values)
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

}  // namespace CopyData

}  // namespace TemperatureAssembly


namespace NavierStokesAssembly {

namespace Scratch {

template <int dim>
RightHandSide<dim>::RightHandSide
(const FiniteElement<dim>  &stokes_fe,
 const Mapping<dim>        &mapping,
 const Quadrature<dim>    &stokes_quadrature,
 const UpdateFlags         stokes_update_flags,
 const FiniteElement<dim> &pressure_fe,
 const UpdateFlags         pressure_update_flags,
 const FiniteElement<dim> &temperature_fe,
 const UpdateFlags         temperature_update_flags)
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
pressure_fe_values(mapping,
                   pressure_fe,
                   stokes_quadrature,
                   pressure_update_flags),
old_pressure_values(stokes_quadrature.size()),
old_phi_values(stokes_quadrature.size()),
old_old_phi_values(stokes_quadrature.size()),
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
phi_velocity(scratch.phi_velocity),
grad_phi_velocity(scratch.grad_phi_velocity),
old_velocity_values(scratch.old_velocity_values),
old_old_velocity_values(scratch.old_old_velocity_values),
old_velocity_gradients(scratch.old_velocity_gradients),
old_old_velocity_gradients(scratch.old_velocity_gradients),
pressure_fe_values(scratch.pressure_fe_values.get_mapping(),
                   scratch.pressure_fe_values.get_fe(),
                   scratch.pressure_fe_values.get_quadrature(),
                   scratch.pressure_fe_values.get_update_flags()),
old_pressure_values(scratch.old_pressure_values),
old_phi_values(scratch.old_phi_values),
old_old_phi_values(scratch.old_old_phi_values),
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

namespace PressureAssembly {

namespace Scratch {

template <int dim>
RightHandSide<dim>::RightHandSide
(const FiniteElement<dim>  &pressure_fe,
 const Mapping<dim>        &mapping,
 const Quadrature<dim>     &pressure_quadrature,
 const UpdateFlags          pressure_update_flags,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags          velocity_update_flags)
:
pressure_fe_values(mapping,
                   pressure_fe,
                   pressure_quadrature,
                   pressure_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   pressure_quadrature,
                   velocity_update_flags),
velocity_divergences(pressure_quadrature.size())
{}

template <int dim>
RightHandSide<dim>::RightHandSide(const RightHandSide<dim> &scratch)
:
pressure_fe_values(scratch.pressure_fe_values.get_mapping(),
                   scratch.pressure_fe_values.get_fe(),
                   scratch.pressure_fe_values.get_quadrature(),
                   scratch.pressure_fe_values.get_update_flags()),
velocity_fe_values(scratch.velocity_fe_values.get_mapping(),
                   scratch.velocity_fe_values.get_fe(),
                   scratch.velocity_fe_values.get_quadrature(),
                   scratch.velocity_fe_values.get_update_flags()),
velocity_divergences(scratch.velocity_divergences)
{}

}  // namespace Scratch

namespace CopyData {


template <int dim>
RightHandSide<dim>::RightHandSide(const FiniteElement<dim> &pressure_fe)
:
local_rhs(pressure_fe.dofs_per_cell),
local_dof_indices(pressure_fe.dofs_per_cell)
{}

template <int dim>
RightHandSide<dim>::RightHandSide(const RightHandSide<dim> &data)
:
local_rhs(data.local_rhs),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace Copy

}  // namespace PressureAssembly


// explicit instantiation
template class TemperatureAssembly::Scratch::RightHandSide<2>;
template class TemperatureAssembly::Scratch::RightHandSide<3>;

template class TemperatureAssembly::CopyData::RightHandSide<2>;
template class TemperatureAssembly::CopyData::RightHandSide<3>;

template class NavierStokesAssembly::Scratch::RightHandSide<2>;
template class NavierStokesAssembly::Scratch::RightHandSide<3>;

template class NavierStokesAssembly::CopyData::RightHandSide<2>;
template class NavierStokesAssembly::CopyData::RightHandSide<3>;

template class PressureAssembly::Scratch::RightHandSide<2>;
template class PressureAssembly::Scratch::RightHandSide<3>;

template class PressureAssembly::CopyData::RightHandSide<2>;
template class PressureAssembly::CopyData::RightHandSide<3>;
