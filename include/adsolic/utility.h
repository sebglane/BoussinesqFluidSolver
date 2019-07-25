/*
 * utility.h
 *
 *  Created on: Jul 25, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_UTILITY_H_
#define INCLUDE_ADSOLIC_UTILITY_H_

#include<deal.II/base/tensor_function.h>

namespace adsolic
{

using namespace dealii;

namespace AuxiliaryFunctions
{
/*
 *
 * Auxiliary convective field with time history-
 *
 */
template <int dim>
class ConvectionFunction : public TensorFunction<1,dim>
{
public:
    ConvectionFunction(const double amplitude = 1.0,
                       const double kx = 2.0 * numbers::PI,
                       const double ky = 2.0 * numbers::PI,
                       const double kz = 2.0 * numbers::PI);

    virtual void
    set_time(const double new_time);

    virtual Tensor<1,dim>
    value(const Point<dim> &point) const;

    Tensor<1,dim>
    old_value(const Point<dim> &point) const;

    void
    old_value_list(const std::vector<Point<dim>> &points,
                   std::vector<Tensor<1,dim>> &values) const;

    Tensor<1,dim>
    old_old_value(const Point<dim> &point) const;

    void
    old_old_value_list(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1,dim>> &values) const;
private:
    const double amplitude;
    const double kx;
    const double ky;
    const double kz;

    double  old_time;
    double  old_old_time;
};

}  // namespace utility

}  // namespace adsolic


#endif /* INCLUDE_ADSOLIC_UTILITY_H_ */
