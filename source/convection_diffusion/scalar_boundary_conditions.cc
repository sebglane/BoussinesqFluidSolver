/*
 * scalar_boundary_conditions.cc
 *
 *  Created on: Jul 24, 2019
 *      Author: sg
 */
#include <adsolic/boundary_conditions.h>

#include <deal.II/base/function.h>

namespace adsolic
{

namespace BC
{

template <int dim>
ScalarBoundaryConditions<dim>::ScalarBoundaryConditions()
{
    for (unsigned int d=0; d<dim; ++d)
        periodic_bcs[d] = std::pair<types::boundary_id,types::boundary_id>
        (numbers::invalid_boundary_id,numbers::invalid_boundary_id);
}

template<int dim>
void ScalarBoundaryConditions<dim>::clear_all_boundary_conditions()
{
    dirichlet_bcs.clear();
    neumann_bcs.clear();
}

template<int dim>
void ScalarBoundaryConditions<dim>::set_dirichlet_bc
(const types::boundary_id                boundary_id,
 const std::shared_ptr<Function<dim>>   &dirichlet_function)
{
    check_boundary_id(boundary_id);

    if (dirichlet_function.get() == 0)
            dirichlet_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(1));
    else
    {
        AssertThrow(dirichlet_function->n_components == 1,
                    ExcMessage("Scalar boundary function need to have a single component."));

        dirichlet_bcs[boundary_id] = dirichlet_function;
    }
}

template<int dim>
void ScalarBoundaryConditions<dim>::set_neumann_bc
(const types::boundary_id              boundary_id,
 const std::shared_ptr<Function<dim>> &neumann_function)
{
    check_boundary_id(boundary_id);

    if (neumann_function.get() == 0)
        neumann_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim));
    else
    {
        AssertThrow(neumann_function->n_components == dim,
                ExcMessage("Neumann boundary function needs to have dim components."));
        neumann_bcs[boundary_id] = neumann_function;
    }
}

template<int dim>
void ScalarBoundaryConditions<dim>::set_periodic_bc
(const unsigned int         direction,
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
void ScalarBoundaryConditions<dim>::check_boundary_id
(const types::boundary_id boundary_id) const
{
    AssertThrow(dirichlet_bcs.find(boundary_id) == dirichlet_bcs.end(),
                ExcMessage("A Dirichlet boundary condition was already set on "
                           "the given boundary."));

    AssertThrow(neumann_bcs.find(boundary_id) == neumann_bcs.end(),
                ExcMessage("A Neumann boundary condition was already set on "
                           "the given boundary."));

    for (unsigned int d=0; d<dim; ++d)
        if (periodic_bcs[d].first == boundary_id ||
            periodic_bcs[d].second == boundary_id)
            AssertThrow(false,
                        ExcMessage("A periodic boundary condition was already set on "
                                   "the given boundary."));
}

// explicit instantiations
template class ScalarBoundaryConditions<1>;
template class ScalarBoundaryConditions<2>;
template class ScalarBoundaryConditions<3>;

}  // namespace BC

}  // namespace adsolic
