/*
 * grid_factory.h
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#ifndef INCLUDE_GRID_FACTORY_H_
#define INCLUDE_GRID_FACTORY_H_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <cmath>


namespace DomainIdentifiers {
/*
 *
 * enumeration for boundary identifiers
 *
 */
enum BoundaryIds
{
    // inner core boundary
    ICB,
    // core mantle boundary
    CMB,
    // fictitious vacuum boundary
    FVB
};

/*
 *
 * enumeration for material identifiers
 *
 */
enum MaterialIds
{
    Fluid,
    Vacuum,
    Solid
};
}  // namespace DomainIdentifiers


namespace GridFactory {

using namespace dealii;
using namespace DomainIdentifiers;

template<int dim>
class SphericalShell
{
public:
    SphericalShell(const double aspect_ratio,
                   const bool   include_core = false,
                   const bool   include_exterior = false,
                   const double exterior_length = 5.0);

    void create_coarse_mesh(Triangulation<dim> &coarse_grid);

private:
    const double    aspect_ratio;
    const bool      include_core;
    const bool      include_exterior;
    const double    exterior_length;

    SphericalManifold<dim> spherical_manifold;

    TransfiniteInterpolationManifold<dim> interpolation_manifold;


    const double    tol = 1e-12;
};

}  // namespace GridFactory


#endif /* INCLUDE_GRID_FACTORY_H_ */
