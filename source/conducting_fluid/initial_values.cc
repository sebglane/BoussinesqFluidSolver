/*
 * initial_values.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include "initial_values.h"

namespace EquationData
{

template <int dim>
MagneticInitialValues<dim>::MagneticInitialValues(const double outer_radius)
:
Function<dim>(dim+1),
ro(outer_radius)
{
    Assert(ro > 0, ExcLowerRangeType<double>(0, ro));
}

template<>
void MagneticInitialValues<3>::vector_value(
        const Point<3>    &point,
        Vector<double>    &value) const
{
    const unsigned int dim = 3;

    AssertDimension(value.size(), dim + 1);

    const double radius = point.distance(Point<dim>());

    if (radius - ro <= tol)
    {
        value[0] = - point[1] * (ro - radius);
        value[1] = point[0] * (ro - radius);
        value[2] = 0.;
    }
    else
    {
        value = 0.;
    }
}

}  // namespace EquationData

// explicit instantiation
template class EquationData::MagneticInitialValues<3>;
