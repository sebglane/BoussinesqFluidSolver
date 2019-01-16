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

template<>
double TemperatureInitialValues<2>::value(
        const Point<2>    &point,
        const unsigned int  /* component */) const
{
    const double radius = point.distance(Point<2>());
    Assert(radius > 0.0, ExcLowerRangeType<double>(0, radius));
    const double log_radius = std::log(radius);
    AssertIsFinite(log_radius);
    const double log_ro = std::log(ro), log_ri = std::log(ri);
    AssertIsFinite(log_ro);
    AssertIsFinite(log_ri);
    const double value = (Ti - To) * log_radius / (log_ri - log_ro)
            + (To * log_ri - Ti *  log_ro) / (log_ri - log_ro);
    return value;
}

template<>
double TemperatureInitialValues<3>::value(
        const Point<3>    &point,
        const unsigned int  /* component */) const
{
    const double radius = point.distance(Point<3>());
    const double value = (ri * Ti - ro * To)/(ri - ro)
            + ri * ro * (To - Ti)/(radius * (ri - ro));
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
template class EquationData::TemperatureInitialValues<2>;
template class EquationData::TemperatureInitialValues<3>;

template class EquationData::GravityVector<1>;
template class EquationData::GravityVector<2>;
template class EquationData::GravityVector<3>;
