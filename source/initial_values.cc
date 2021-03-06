/*
 * initial_values.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include <adsolic/initial_values.h>

namespace EquationData
{

template <int dim>
VelocityTestValues<dim>::VelocityTestValues(const double inner_radius, const double outer_radius)
:
Function<dim>(dim+1),
inner_radius(inner_radius),
outer_radius(outer_radius)
{
    Assert(inner_radius > 0.0, ExcNegativeRadius(inner_radius));
    Assert(outer_radius > 0.0, ExcNegativeRadius(outer_radius));
    Assert(inner_radius < outer_radius, ExcLowerRangeType<double>(inner_radius, outer_radius));
}

template<>
void VelocityTestValues<2>::vector_value(
        const Point<2>    &point,
        Vector<double>    &value) const
{
    const unsigned int dim = 2;

    AssertDimension(value.size(), dim + 1);

    const double radius = point.distance(Point<dim>());
    Assert(radius> 0., ExcNegativeRadius(radius));

    const double phi = atan2(point[1], point[0]);
    Assert(phi >= -numbers::PI && phi <= numbers::PI,
           ExcAzimuthalAngleRange(phi));

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
    Assert(radius> 0., ExcNegativeRadius(radius));

    const double cylinder_radius = sqrt(point[0]*point[0] + point[1]*point[1]);

    const double theta = atan2(cylinder_radius, point[2]);
    Assert(theta >= 0. && theta <= numbers::PI,
           ExcPolarAngleRange(theta));

    const double phi = atan2(point[1], point[0]);
    Assert(phi >= -numbers::PI && phi <= numbers::PI,
           ExcAzimuthalAngleRange(phi));


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
inner_radius(inner_radius),
outer_radius(outer_radius),
perturbation_type(perturbation_type_)
{
    Assert(inner_radius > 0.0, ExcNegativeRadius(inner_radius));
    Assert(outer_radius > 0.0, ExcNegativeRadius(outer_radius));
    Assert(inner_radius < outer_radius,
           ExcLowerRangeType<double>(inner_radius, outer_radius));
}

template<>
double TemperatureInitialValues<2>::value(
        const Point<2>    &point,
        const unsigned int  /* component */) const
{
    const double r = point.distance(Point<2>());
    Assert(r > 0.0, ExcNegativeRadius(r));

    const double log_r = log(r);
    AssertIsFinite(log_r);

    const double log_ro = log(outer_radius), log_ri = log(inner_radius);
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
            value = (log_ro - log_r) / (log_ro - log_ri);

            const double phi = atan2(point[1], point[0]);
            Assert(phi >= -numbers::PI && phi <= numbers::PI,
                   ExcAzimuthalAngleRange(phi));


            const double tol = 1e-12;
            const double xi = (2.0 * r - outer_radius - inner_radius)/(outer_radius - inner_radius);
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
    Assert(r > 0.0, ExcNegativeRadius(r));

    double value = 0.;

    switch (perturbation_type)
    {
        case TemperaturePerturbation::None:
            value = (outer_radius - r) / (outer_radius - inner_radius) * inner_radius / r;
            break;
        case TemperaturePerturbation::Sinusoidal:
        {
            value = (outer_radius - r) / (outer_radius - inner_radius) * inner_radius / r;

            const double r_cylinder = sqrt(point[0]*point[0] + point[1]*point[1]);
            Assert(r_cylinder >= 0.0, ExcNegativeRadius(r_cylinder));

            const double theta = atan2(r_cylinder, point[2]);
            Assert(theta >= 0. && theta <= numbers::PI,
                   ExcPolarAngleRange(theta));

            const double phi = atan2(point[1], point[0]);
            Assert(phi >= -numbers::PI && phi <= numbers::PI,
                   ExcAzimuthalAngleRange(phi));
            const double tol = 1e-12;
            const double xi = (2.0 * r - outer_radius - inner_radius) / (outer_radius - inner_radius);
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

}  // namespace EquationData

// explicit instantiation
template class EquationData::VelocityTestValues<2>;
template class EquationData::VelocityTestValues<3>;

template class EquationData::TemperatureInitialValues<2>;
template class EquationData::TemperatureInitialValues<3>;
