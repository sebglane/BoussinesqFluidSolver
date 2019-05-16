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
InitialField<dim>::InitialField(const double inner_radius, const double outer_radius)
:
Function<dim>(dim+1),
inner_radius(inner_radius),
outer_radius(outer_radius)
{
    Assert(inner_radius < outer_radius, ExcLowerRangeType<double>(inner_radius, outer_radius));
    Assert(0 < inner_radius, ExcLowerRangeType<double>(0, inner_radius));
}

template<>
void InitialField<3>::vector_value(
        const Point<3>    &point,
        Vector<double>    &value) const
{
    const unsigned int dim = 3;

    AssertDimension(value.size(), dim + 1);

    const double radius = point.distance(Point<dim>());
    Assert(radius> 0., ExcLowerRangeType<double>(radius, 0.));

    const double cylinder_radius = sqrt(point[0]*point[0] + point[1]*point[1]);

    const double theta = atan2(cylinder_radius, point[2]);
    Assert(theta >= 0., ExcLowerRangeType<double>(theta, 0.));
    Assert(theta <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, theta));

    const double phi = atan2(point[1], point[0]);
    Assert(phi >= -numbers::PI, ExcLowerRangeType<double>(theta, -numbers::PI));
    Assert(phi <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, theta));

    const double b_r = 5./(8. * sqrt(2.)) *
            (
                - 48. * inner_radius * outer_radius
                + (4. * outer_radius + inner_radius * (4. + 3. * outer_radius) ) * 6. * radius
                - 4. * (4. + 3. * (inner_radius + outer_radius) ) * pow(radius, 2.)
                + 9. * pow(radius, 3.)
            ) / radius * cos(theta);
    AssertIsFinite(b_r);

    const double b_theta = 15. / (4. * sqrt(2.))
            * (radius - outer_radius)
            * (radius - inner_radius)
            * (3. * radius - 4.) / radius * sin(theta);
    AssertIsFinite(b_theta);

    const double b_phi = 15. / (8. * sqrt(2.))
            * sin(numbers::PI * (radius -  inner_radius) / (outer_radius - inner_radius) )
            * sin(2. * theta);
    AssertIsFinite(b_phi);

    value[0] = (b_r * sin(theta) + b_theta * cos(theta) ) * cos(phi)
            - b_phi * sin(phi);
    AssertIsFinite(value[0]);

    value[1] = (b_r * sin(theta) + b_theta * cos(theta) ) * sin(phi)
            + b_phi * cos(phi);
    AssertIsFinite(value[1]);

    value[2] = b_r * cos(theta) - b_theta * sin(theta);
    AssertIsFinite(value[2]);

    value[3] = 0.0;
}

}  // namespace EquationData

// explicit instantiation
template class EquationData::InitialField<3>;
