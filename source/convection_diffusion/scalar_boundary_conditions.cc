/*
 * scalar_boundary_conditions.cc
 *
 *  Created on: Jul 24, 2019
 *      Author: sg
 */
#include <adsolic/convection_diffusion_solver.h>

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
    dirichlet_ids.clear();
    neumann_ids.clear();
}

template<int dim>
void ScalarBoundaryConditions<dim>::set_dirichlet_bc
(const types::boundary_id                boundary_id,
 const std::shared_ptr<Function<dim>>   &dirichlet_function)
{
    if (dirichlet_function.get() == 0)
        return;
    AssertThrow(dirichlet_function->n_components == 1,
                ExcMessage("Scalar boundary function need to have a single component."));

    dirichlet_bcs[boundary_id] = dirichlet_function;
    dirichlet_ids.insert(boundary_id);
}

template<int dim>
void ScalarBoundaryConditions<dim>::set_neumann_bc
(const types::boundary_id              boundary_id,
 const std::shared_ptr<Function<dim>> &neumann_function)
{
    if (neumann_function.get() == 0)
        return;
    AssertThrow(neumann_function->n_components == 1,
                ExcMessage("Scalar boundary function need to have a single component."));

    neumann_bcs[boundary_id] = neumann_function;
    neumann_ids.insert(boundary_id);
}

template<int dim>
void ScalarBoundaryConditions<dim>::set_periodic_bc
(const unsigned int         direction,
 const types::boundary_id   first_boundary_id,
 const types::boundary_id   second_boundary_id)
{
    AssertThrow(direction >= 0 && direction < dim,
                ExcMessage("Coordinate direction must be between 0 and the dim"));
    periodic_bcs[direction] = std::make_pair(first_boundary_id,
                                                    second_boundary_id);
}

template class ScalarBoundaryConditions<1>;
template class ScalarBoundaryConditions<2>;
template class ScalarBoundaryConditions<3>;

}  // namespace BC

}  // namespace adsolic



