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

#include "exceptions.h"

namespace EquationData {

/*
 *
 * enumeration for the type of the temperature perturbation
 *
 */
enum TemperaturePerturbation
{
    None,
    Sinusoidal
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

template <int dim>
class GravityFunction : public TensorFunction<1,dim>
{
public:
    GravityFunction();

    virtual Tensor<1,dim>   value(const Point<dim> &p) const;

    virtual void            value_list(const std::vector<Point<dim>>    &points,
                                       std::vector<Tensor<1,dim>>       &values) const;
};

}  // namespace EquationData



#endif /* INCLUDE_INITIAL_VALUES_H_ */
