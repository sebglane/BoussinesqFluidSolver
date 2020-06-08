/*
 * navier_stokes_boundary_conditions.cc
 *
 *  Created on: Jul 30, 2019
 *      Author: sg
 */

#include <adsolic/boundary_conditions.h>

#include <deal.II/base/function.h>

namespace adsolic
{

namespace BC
{

template<int dim>
NavierStokesBoundaryConditions<dim>::NavierStokesBoundaryConditions()
{
    for (unsigned int d=0; d<dim; ++d)
        periodic_bcs[d] = std::pair<types::boundary_id,types::boundary_id>
        (numbers::invalid_boundary_id,numbers::invalid_boundary_id);
}

template<int dim>
void NavierStokesBoundaryConditions<dim>::clear_all_boundary_conditions()
{
    dirichlet_bcs_velocity.clear();
    dirichlet_bcs_pressure.clear();
    open_bcs_pressure.clear();

    normal_flux.clear();
    no_slip.clear();
}

template<int dim>
void NavierStokesBoundaryConditions<dim>::set_dirichlet_bc_velocity
(const types::boundary_id   boundary_id,
 const std::shared_ptr<Function<dim>>  &velocity_function)
{
    check_boundary_id(boundary_id);

    if (velocity_function.get() == 0)
    {
        no_slip.insert(boundary_id);
        return;
    }

    AssertThrow(velocity_function->n_components == dim,
                ExcMessage("Velocity boundary function need to have dim components."));
    dirichlet_bcs_velocity[boundary_id] = velocity_function;
}

template<int dim>
void NavierStokesBoundaryConditions<dim>::set_dirichlet_bc_pressure
(const types::boundary_id   boundary_id,
 const std::shared_ptr<Function<dim>>  &pressure_function)
{
    check_boundary_id(boundary_id);

    if (pressure_function.get() == 0)
        dirichlet_bcs_pressure[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(1));
    else
    {
        AssertThrow(pressure_function->n_components == 1,
                    ExcMessage("Pressure boundary function needs to be scalar."));
        dirichlet_bcs_pressure[boundary_id] = pressure_function;
    }
}


template<int dim>
void NavierStokesBoundaryConditions<dim>::set_open_bc
(const types::boundary_id   boundary_id,
 const bool constrain_normal_flux,
 const std::shared_ptr<Function<dim>>   &pressure_function)
{
    check_boundary_id(boundary_id);

    if (pressure_function.get() == 0)
        open_bcs_pressure[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(1));
    else
    {
        AssertThrow(pressure_function->n_components == 1,
                    ExcMessage("Pressure boundary function needs to be scalar."));
        open_bcs_pressure[boundary_id] = pressure_function;
    }

    if (constrain_normal_flux)
        normal_flux.insert(boundary_id);
}

template<int dim>
void NavierStokesBoundaryConditions<dim>::set_periodic_bc
(const unsigned int direction,
 const types::boundary_id   first_boundary_id,
 const types::boundary_id   second_boundary_id)
{
    check_boundary_id(first_boundary_id);
    check_boundary_id(second_boundary_id);

    AssertThrow(direction >= 0 && direction < dim,
                ExcMessage("Coordinate direction must be between 0 and the dim"));
    periodic_bcs[direction] = std::make_pair(first_boundary_id,
                                             second_boundary_id);
}

template<int dim>
void NavierStokesBoundaryConditions<dim>::check_boundary_id
(const types::boundary_id boundary_id) const
{
    AssertThrow(dirichlet_bcs_velocity.find(boundary_id) == dirichlet_bcs_velocity.end(),
                ExcMessage("A Dirichlet velocity boundary condition was already "
                           "set on the given boundary."));

    AssertThrow(dirichlet_bcs_pressure.find(boundary_id) == dirichlet_bcs_pressure.end(),
                ExcMessage("A Dirichlet pressure boundary condition was already "
                           "set on the given boundary."));

    AssertThrow(open_bcs_pressure.find(boundary_id) == open_bcs_pressure.end(),
                ExcMessage("An open boundary condition was already "
                           "set on the given boundary."));

    AssertThrow(normal_flux.find(boundary_id) == normal_flux.end(),
                ExcMessage("A tangential velocity boundary condition was already "
                           "set on the given boundary."));

    AssertThrow(no_slip.find(boundary_id) == no_slip.end(),
                ExcMessage("A no-slip velocity boundary condition was already "
                           "set on the given boundary."));

    for (unsigned int d=0; d<dim; ++d)
        if (periodic_bcs[d].first == boundary_id ||
            periodic_bcs[d].second == boundary_id)
            AssertThrow(false,
                        ExcMessage("A periodic boundary condition was already set on "
                                   "the given boundary."));
}

// explicit instantiations
template class NavierStokesBoundaryConditions<1>;
template class NavierStokesBoundaryConditions<2>;
template class NavierStokesBoundaryConditions<3>;

}  // namespace BC

}  // namespace adsolic
