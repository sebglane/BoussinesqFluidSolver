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
 const double phi_x_in,
 const double phi_y_in)
:
amplitude(amplitude_in),
kx(kx_in),
ky(ky_in),
phi_x(phi_x_in),
phi_y(phi_y_in),
old_time(this->get_time()),
old_old_time(this->get_time())
{
    Assert(dim == 2,
           ExcMessage("This class is only implemented in 2D."));
}

template <int dim>
void
ConvectionFunction<dim>::set_time(const double new_time)
{
    old_old_time = old_time;
    old_time = this->get_time();
    FunctionTime<double>::set_time(new_time);
}

template<int dim>
Tensor<1,dim> ConvectionFunction<dim>::value
(const Point<dim> &point) const
{
    Tensor<1,dim> value;
    value[0] = amplitude
             * std::cos(kx * point[0] - phi_x) * std::cos(ky * point[1] - phi_y);
    value[1] = - amplitude * kx / ky
             * std::sin(kx * point[0] - phi_x) * std::sin(ky * point[1] - phi_y);

    return value;
}

template<int dim>
Tensor<1,dim> ConvectionFunction<dim>::old_value
(const Point<dim> &point) const
{
    Tensor<1,dim> value;
    value[0] = amplitude
             * std::cos(kx * point[0] - phi_x) * std::cos(ky * point[1] - phi_y);
    value[1] = - amplitude * kx / ky
             * std::sin(kx * point[0] - phi_x) * std::sin(ky * point[1] - phi_y);

    return value;
}

template<int dim>
Tensor<1,dim> ConvectionFunction<dim>::old_old_value
(const Point<dim> &point) const
{
    Tensor<1,dim> value;
    value[0] = amplitude
             * std::cos(kx * point[0] - phi_x) * std::cos(ky * point[1] - phi_y);
    value[1] = - amplitude * kx / ky
             * std::sin(kx * point[0] - phi_x) * std::sin(ky * point[1] - phi_y);

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
