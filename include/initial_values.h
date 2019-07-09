/*
 * initial_values.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_INITIAL_VALUES_H_
#define INCLUDE_INITIAL_VALUES_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <exceptions.h>

namespace EquationData {

/*
 *
 * enumeration for the type of the temperature perturbation
 *
 */
enum TemperaturePerturbation
{
    None,
    Sinusoidal,
    default_perturbation=None
};

/*
 *
 * enumeration for the type of the gravity profile
 *
 */
enum GravityProfile
{
    constant,
    linear,
    default_profile=linear
};

using namespace dealii;

using namespace GeometryExceptions;

template<int dim>
class VelocityTestValues : public Function<dim>
{
public:
    VelocityTestValues(const double inner_radius,
                       const double outer_radius);

    virtual void vector_value(const Point<dim>   &point,
                              Vector<double> &values) const;

private:
    const double inner_radius;
    const double outer_radius;
};


template<int dim>
class TemperatureInitialValues : public Function<dim>
{
public:
    TemperatureInitialValues(const double inner_radius,
                             const double outer_radius,
                             const TemperaturePerturbation  perturbation_type = TemperaturePerturbation::None);

    virtual double value(const Point<dim>   &point,
                         const unsigned int component = 0) const;

private:
    const double inner_radius;
    const double outer_radius;

    const TemperaturePerturbation   perturbation_type;
};

template<int dim>
class MagneticFieldInitialValues : public TensorFunction<1,dim>
{
public:
    MagneticFieldInitialValues(const double inner_radius,
                               const double outer_radius);

    virtual Tensor<1,dim> value(const Point<dim>  &point) const;

private:
    const double inner_radius;
    const double outer_radius;

    const double scaling_factors[2] = {7. / 13., 20. / 13.};

    const double coefficients[4]
    =
    {
            -48. * scaling_factors[0],
            6. * (4. * scaling_factors[1] + scaling_factors[0] * (4. + 3. * scaling_factors[1])),
            -4. * (4. + 3. * (scaling_factors[0] + scaling_factors[1])) * scaling_factors[1],
            9. * scaling_factors[1] * scaling_factors[1]
    };
};

template <int dim>
class GravityFunction : public TensorFunction<1,dim>
{
public:
    GravityFunction(const double    outer_radius = 1.0,
                    const GravityProfile    profile_type = GravityProfile::constant);

    virtual Tensor<1,dim>   value(const Point<dim> &p) const;

    virtual void            value_list(const std::vector<Point<dim>>    &points,
                                       std::vector<Tensor<1,dim>>       &values) const;

    GravityProfile  get_profile_type() const;

private:
    const double outer_radius;

    const GravityProfile    profile_type;
};

}  // namespace EquationData



#endif /* INCLUDE_INITIAL_VALUES_H_ */
