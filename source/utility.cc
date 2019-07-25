/*
 * utility.cc
 *
 *  Created on: Jul 25, 2019
 *      Author: sg
 */

#include<adsolic/utility.h>

namespace adsolic
{

namespace AuxiliaryFunctions
{

template <int dim>
ConvectionFunction<dim>::ConvectionFunction
(const double amplitude_in,
 const double kx_in,
 const double ky_in,
 const double kz_in)
:
amplitude(amplitude_in),
kx(kx_in),
ky(ky_in),
kz(kz_in),
old_time(this->get_time()),
old_old_time(this->get_time())
{}

template <int dim>
void
ConvectionFunction<dim>::set_time(const double new_time)
{
    old_old_time = old_time;
    old_time = this->get_time();
    this->set_time(new_time);
}

template<int dim>
Tensor<1,dim> ConvectionFunction<dim>::value
(const Point<dim> &point) const
{
    AssertThrow(dim == 2 || dim == 3,
                ExcImpossibleInDim(dim));

    Tensor<1,dim> value;
    value[0] = amplitude * std::sin(kx * point[0]) * std::cos(ky * point[1]);
    value[1] = -amplitude * kx / ky * std::cos(kx * point[0]) * std::sin(ky * point[1]);

    if (dim == 3)
        value[2] = 0;

    return value;
}

template<int dim>
Tensor<1,dim> ConvectionFunction<dim>::old_value
(const Point<dim> &point) const
{
    AssertThrow(dim == 2 || dim == 3,
                ExcImpossibleInDim(dim));

    Tensor<1,dim> value;
    value[0] = amplitude * std::sin(kx * point[0]) * std::cos(ky * point[1]);
    value[1] = -amplitude * kx / ky * std::cos(kx * point[0]) * std::sin(ky * point[1]);

    if (dim == 3)
        value[2] = 0;

    return value;
}

template<int dim>
Tensor<1,dim> ConvectionFunction<dim>::old_old_value
(const Point<dim> &point) const
{
    AssertThrow(dim == 2 || dim == 3,
                ExcImpossibleInDim(dim));

    Tensor<1,dim> value;
    value[0] = amplitude * std::sin(kx * point[0]) * std::cos(ky * point[1]);
    value[1] = -amplitude * kx / ky * std::cos(kx * point[0]) * std::sin(ky * point[1]);

    if (dim == 3)
        value[2] = 0;

    return value;
}

template<int dim>
void ConvectionFunction<dim>::old_value_list
(const std::vector<Point<dim>> &points,
 std::vector<Tensor<1,dim>> &values) const
{
    AssertDimension(points.size(), values.size());
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = old_value(points[i]);
}

template<int dim>
void ConvectionFunction<dim>::old_old_value_list
(const std::vector<Point<dim>> &points,
 std::vector<Tensor<1,dim>> &values) const
{
    AssertDimension(points.size(), values.size());
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = old_old_value(points[i]);
}

// explicit instantiation
template class ConvectionFunction<2>;
template class ConvectionFunction<3>;

}  // namespace utility

}  // namespace adsolic
