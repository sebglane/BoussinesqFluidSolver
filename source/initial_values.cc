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
TemperatureInitialValues<dim>::TemperatureInitialValues(
        const double inner_radius,
        const double outer_radius,
        const double inner_temperature,
        const double outer_temperature)
:
Function<dim>(1),
ri(inner_radius),
ro(outer_radius),
Ti(inner_temperature),
To(outer_temperature)
{
    Assert(To < Ti, ExcLowerRangeType<double>(To, Ti));
    Assert(ri < ro, ExcLowerRangeType<double>(ri, ro));
}

template<int dim>
double TemperatureInitialValues<dim>::value(
        const Point<dim>    &point,
        const unsigned int  /* component */) const
{
    const double radius = point.distance(Point<dim>());
    const double value = Ti + (To - Ti) / (ro - ri) * (radius - ri);
    return value;
}

template<int dim>
GravityVector<dim>::GravityVector()
:
TensorFunction<1,dim>()
{}

template<int dim>
Tensor<1,dim> GravityVector<dim>::value(const Point<dim> &point) const
{
    const double r = point.norm();
    Assert(r > 0, StandardExceptions::ExcDivideByZero());
    return -point/r;
}

template<int dim>
void GravityVector<dim>::value_list(const std::vector<Point<dim>>    &points,
                                    std::vector<Tensor<1,dim>> &values) const
{
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = value(points[i]);
}

}  // namespace EquationData

// explicit instantiation
template class EquationData::TemperatureInitialValues<1>;
template class EquationData::TemperatureInitialValues<2>;
template class EquationData::TemperatureInitialValues<3>;

template class EquationData::GravityVector<1>;
template class EquationData::GravityVector<2>;
template class EquationData::GravityVector<3>;
