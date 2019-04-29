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
class InitialField : public Function<dim>
{
public:
    InitialField(const double inner_radius = 0.35, const double outer_radius = 1.0);

    virtual void vector_value(const Point<dim>   &point,
                              Vector<double>     &values) const;

private:
    const double inner_radius;
    const double outer_radius;
};


}  // namespace EquationData



#endif /* INCLUDE_INITIAL_VALUES_H_ */
