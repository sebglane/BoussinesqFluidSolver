/*
 * gravity_field.cc
 *
 *  Created on: Jul 11, 2019
 *      Author: sg
 */

#include <gravity_field.h>

namespace EquationData
{

template<int dim>
GravityFunction<dim>::GravityFunction
(const double           outer_radius,
 const GravityProfile   profile_type)
:
TensorFunction<1,dim>(),
outer_radius(outer_radius),
profile_type(profile_type)
{}

template<int dim>
Tensor<1,dim> GravityFunction<dim>::value(const Point<dim> &point) const
{
    const double r = point.norm();
    Assert(r > 0, ExcNegativeRadius(r));

    Tensor<1,dim> value;

    switch (profile_type)
    {
        case GravityProfile::Constant:
        {
            value = -point / r;
            break;
        }
        case GravityProfile::Linear:
        {
            value = -point * scaling_factor / outer_radius;
            break;
        }
        default:
            Assert(false, ExcInternalError());
            break;
    }
    return value;
}

template<int dim>
void GravityFunction<dim>::value_list(const std::vector<Point<dim>>    &points,
                                      std::vector<Tensor<1,dim>> &values) const
{
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = value(points[i]);
}

template<int dim>
GravityProfile GravityFunction<dim>::get_profile_type() const
{
    return profile_type;
}

}  // namespace EquationData

template class EquationData::GravityFunction<1>;
template class EquationData::GravityFunction<2>;
template class EquationData::GravityFunction<3>;
