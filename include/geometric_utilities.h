/*
 * geometric_utilities.h
 *
 *  Created on: Jul 11, 2019
 *      Author: sg
 */

#ifndef INCLUDE_GEOMETRIC_UTILITIES_H_
#define INCLUDE_GEOMETRIC_UTILITIES_H_

#include <deal.II/base/geometric_utilities.h>

#include <array>

namespace CoordinateTransformation {

using namespace dealii;

template<std::size_t dim>
std::array<Tensor<1,dim>,dim>
spherical_basis(const std::array<double,dim> &input);

template<std::size_t size, int dim>
std::array<double, size>
spherical_projections(const Tensor<1,dim> &field,
                      const std::array<Tensor<1,dim>, size> &sbasis);

template<std::size_t size, int dim>
std::array<std::array<double,size>,size>
spherical_projections(const Tensor<2,dim> &field,
                      const std::array<Tensor<1,dim>, size> &sbasis);

}  // namespace CoordinateTransformation


#endif /* INCLUDE_GEOMETRIC_UTILITIES_H_ */
