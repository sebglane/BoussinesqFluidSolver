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
VelocityTestValues<dim>::VelocityTestValues(const double inner_radius, const double outer_radius)
:
Function<dim>(dim+1),
inner_radius(inner_radius),
outer_radius(outer_radius)
{
    Assert(inner_radius < outer_radius, ExcLowerRangeType<double>(inner_radius, outer_radius));
    Assert(0 < inner_radius, ExcLowerRangeType<double>(0, inner_radius));
}

template<>
void VelocityTestValues<2>::vector_value(
        const Point<2>    &point,
        Vector<double>    &value) const
{
    const unsigned int dim = 2;

    AssertDimension(value.size(), dim + 1);

    const double radius = point.distance(Point<dim>());
    Assert(radius> 0., ExcLowerRangeType<double>(radius, 0.));

    const double phi = atan2(point[1], point[0]);
    Assert(phi >= -numbers::PI, ExcLowerRangeType<double>(phi, -numbers::PI));
    Assert(phi <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, phi));

    const double tol = 1e-12;
    const double xi = (2.0 * radius - outer_radius - inner_radius)
                    / (outer_radius - inner_radius);
    Assert(xi >= -1.0 - tol, ExcLowerRangeType<double>(xi, -1.0));
    Assert(xi <= 1.0 + tol, ExcLowerRangeType<double>(1.0, xi));

    const double v_r = 21. / sqrt(17920. * numbers::PI)
            * (1. - 3. * pow(xi, 2) + 3. * pow(xi, 4) - pow(xi, 6))
            * sin(4. * (phi - numbers::PI / 16.));

    value[0] = v_r * cos(phi);
    AssertIsFinite(value[0]);

    value[1] = v_r * sin(phi);
    AssertIsFinite(value[1]);

    value[2] = 0.0;
}

template<>
void VelocityTestValues<3>::vector_value(
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

    const double tol = 1e-12;
    const double xi = (2.0 * radius - outer_radius - inner_radius)
                    / (outer_radius - inner_radius);

    Assert(xi >= -1.0 - tol, ExcLowerRangeType<double>(xi, -1.0));
    Assert(xi <= 1.0 + tol, ExcLowerRangeType<double>(1.0, xi));

    const double v_r = 21. / sqrt(17920. * numbers::PI)
            * (1. - 3. * pow(xi, 2) + 3. * pow(xi, 4) - pow(xi, 6))
            * pow(sin(theta), 4) * sin(4.* (phi - numbers::PI / 16.) );


    value[0] = v_r * sin(theta) * cos(phi);
    AssertIsFinite(value[0]);

    value[1] = v_r * sin(theta) * sin(phi);
    AssertIsFinite(value[1]);

    value[2] = v_r * cos(theta);
    AssertIsFinite(value[2]);

    value[3] = 0.0;
}


template <int dim>
TemperatureInitialValues<dim>::TemperatureInitialValues(
        const double                    inner_radius,
        const double                    outer_radius,
        const TemperaturePerturbation   perturbation_type_)
:
Function<dim>(1),
ri(inner_radius),
ro(outer_radius),
perturbation_type(perturbation_type_)
{
    Assert(ri > 0.0, ExcLowerRangeType<double>(0, ri));
    Assert(ro > 0.0, ExcLowerRangeType<double>(0, ro));
    Assert(ri < ro, ExcLowerRangeType<double>(ri, ro));
}

template<>
double TemperatureInitialValues<2>::value(
        const Point<2>    &point,
        const unsigned int  /* component */) const
{
    const double r = point.distance(Point<2>());
    Assert(r > 0.0, ExcLowerRangeType<double>(0, r));

    const double log_r = log(r);
    AssertIsFinite(log_r);

    const double log_ro = log(ro), log_ri = log(ri);
    AssertIsFinite(log_ro);
    AssertIsFinite(log_ri);

    double value = 0.;

    switch (perturbation_type)
    {
        case TemperaturePerturbation::None:
            value = (log_ro - log_r) / (log_ro - log_ri);
            break;
        case TemperaturePerturbation::Sinusoidal:
        {
            value = (log_ro - log_r) / (log_ro - log_ri);;

            const double phi = atan2(point[1], point[0]);
            Assert(phi >= -numbers::PI, ExcLowerRangeType<double>(phi, -numbers::PI));
            Assert(phi <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, phi));

            const double tol = 1e-12;
            const double xi = (2.0 * r - ro - ri)/(ro - ri);
            Assert(xi >= -1.0 - tol, ExcLowerRangeType<double>(xi, -1.0));
            Assert(xi <= 1.0 + tol, ExcLowerRangeType<double>(1.0, xi));

            const double perturbation = 21. / sqrt(17920. * numbers::PI)
                    * (1. - 3. * pow(xi, 2) + 3. * pow(xi, 4) - pow(xi, 6))
                    * cos(4.*phi);
            value += perturbation;
            break;
        }
        default:
            break;
    }
    return value;
}

template<>
double TemperatureInitialValues<3>::value(
        const Point<3>    &point,
        const unsigned int  /* component */) const
{
    const double r = point.distance(Point<3>());
    Assert(r > 0.0, ExcLowerRangeType<double>(0, r));

    double value = 0.;

    switch (perturbation_type)
    {
        case TemperaturePerturbation::None:
            value = (ro - r) / (ro - ri) * ri / r;
            break;
        case TemperaturePerturbation::Sinusoidal:
        {
            value = (ro - r) / (ro - ri) * ri / r;

            const double r_cylinder = sqrt(point[0]*point[0] + point[1]*point[1]);
            Assert(r_cylinder >= 0.0, ExcLowerRangeType<double>(0.0, r_cylinder));

            const double theta = atan2(r_cylinder, point[2]);
            Assert(theta >= 0., ExcLowerRangeType<double>(theta, 0.));
            Assert(theta <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, theta));

            const double phi = atan2(point[1], point[0]);
            Assert(phi >= -numbers::PI, ExcLowerRangeType<double>(phi, -numbers::PI));
            Assert(phi <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, phi));

            const double tol = 1e-12;
            const double xi = (2.0 * r - ro - ri) / (ro - ri);
            Assert(xi >= -1.0 - tol, ExcLowerRangeType<double>(xi, -1.0));
            Assert(xi <= 1.0 + tol, ExcLowerRangeType<double>(1.0, xi));

            const double perturbation = 21. / sqrt(17920. * numbers::PI)
                    * (1. - 3. * pow(xi, 2) + 3. * pow(xi, 4) - pow(xi, 6))
                    * pow(sin(theta), 4) * cos(4.*phi);
            value += perturbation;
            break;
        }
        default:
            break;
    }
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
template class EquationData::VelocityTestValues<2>;
template class EquationData::VelocityTestValues<3>;

template class EquationData::TemperatureInitialValues<2>;
template class EquationData::TemperatureInitialValues<3>;

template class EquationData::GravityVector<1>;
template class EquationData::GravityVector<2>;
template class EquationData::GravityVector<3>;
