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

namespace EquationData {

using namespace dealii;

/*
 *
 * enumeration for boundary identifiers
 *
 */
enum BoundaryIds
{
    ICB,
    CMB,
};

template<int dim>
class TemperatureInitialValues : public Function<dim>
{
public:
    TemperatureInitialValues(const double inner_radius,
                             const double outer_radius,
                             const double inner_temperature,
                             const double outer_temperature);

    virtual double value(const Point<dim>   &point,
                         const unsigned int component = 0) const;

private:
    const double ri;
    const double ro;
    const double Ti;
    const double To;
};

template<int dim>
class InitialField : public Function<dim>
{
public:
    InitialField(const double outer_radius = 1.0);

    virtual void vector_value(const Point<dim>   &point,
                              Vector<double>     &values) const;

private:
    const double ro;
    const double tol = 1e-12;
};

template <int dim>
class GravityVector : public TensorFunction<1,dim>
{
public:
    GravityVector();

    virtual Tensor<1,dim>   value(const Point<dim> &p) const;

    virtual void            value_list(const std::vector<Point<dim>>    &points,
                                       std::vector<Tensor<1,dim>>       &values) const;
};

}  // namespace EquationData



#endif /* INCLUDE_INITIAL_VALUES_H_ */
