/*
 * grid_factory.h
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#ifndef INCLUDE_GRID_FACTORY_H_
#define INCLUDE_GRID_FACTORY_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

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
    // topographic boundary
    TopoBndry,
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

template<int dim>
class SinusoidalManifold: public ChartManifold<dim,dim,dim-1>
{
public:

    SinusoidalManifold(const double wavenumber = 2. * numbers::PI,
                       const double amplitude = 0.1);

    virtual std::unique_ptr<Manifold<dim,dim>> clone() const;

    virtual Point<dim-1>    pull_back(const Point<dim> &space_point) const;

    virtual Point<dim>      push_forward(const Point<dim-1> &chart_point) const;

private:

    const double wavenumber;

    const double amplitude;
};

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


template<int dim>
class TopographyBox
{
public:
    TopographyBox(const double wavenumber,
                  const double amplitude,
                  const bool   include_exterior = false,
                  const double exterior_length = 2.0);

    void create_coarse_mesh(Triangulation<dim> &coarse_grid);

private:
    const bool      include_exterior;
    const double    exterior_length;

    SinusoidalManifold<dim>  sinus_manifold;

    TransfiniteInterpolationManifold<dim> interpolation_manifold;

    const double    tol = 1e-12;
};
}  // namespace GridFactory

#endif /* INCLUDE_GRID_FACTORY_H_ */
