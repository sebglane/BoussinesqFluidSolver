/*
 * geometric_utilities.cc
 *
 *  Created on: Jul 11, 2019
 *      Author: sg
 */
#include <geometric_utilities.h>
#include <exceptions.h>

namespace CoordinateTransformation {

using namespace dealii;

template<std::size_t dim>
std::array<Tensor<1,dim>,dim>
spherical_basis(const std::array<double,dim> &scoord)
{
    std::array<Tensor<1,dim>,dim> sbasis;

    Assert(scoord[0] >= 0.,
           ExcNegativeRadius(scoord[0]));
    Assert(scoord[1] >= 0. && scoord[1] < 2.*numbers::PI,
           ExcPolarAngleRange(scoord[1]));

    switch (dim)
    {
        case 2:
        {
            // radial basis vector
            sbasis[0][0] = std::cos(scoord[1]);
            sbasis[0][1] = std::sin(scoord[1]);
            // azimuthal basis vector
            sbasis[1][0] = -std::sin(scoord[1]);
            sbasis[1][1] = std::cos(scoord[1]);
            break;
        }
        case 3:
        {
            Assert(scoord[2] >= 0. && scoord[2] <= numbers::PI,
                   ExcAzimuthalAngleRange(scoord[2]));

            // radial basis vector
            sbasis[0][0] = std::cos(scoord[2]) * std::sin(scoord[1]);
            sbasis[0][1] = std::sin(scoord[2]) * std::sin(scoord[1]);
            sbasis[0][2] = std::cos(scoord[2]);
            // polar basis vector
            sbasis[1][0] = std::cos(scoord[2]) * std::cos(scoord[1]);
            sbasis[1][1] = std::sin(scoord[2]) * std::cos(scoord[1]);
            sbasis[1][2] = -std::sin(scoord[2]);
            // azimuthal basis vector
            sbasis[2][0] = -std::sin(scoord[2]);
            sbasis[2][1] = std::cos(scoord[2]);
            sbasis[2][2] = 0.0;
            break;
        }
        default:
            Assert(false,ExcImpossibleInDim(dim));
            break;
    }
    return sbasis;
}

template<std::size_t size, int dim>
std::array<double,size>
spherical_projections
(const Tensor<1,dim>                    &field,
 const std::array<Tensor<1,dim>,size>    &sbasis)
{
    AssertDimension(size, dim);

    std::array<double,size> sprojections;

    for (unsigned int d=0; d<dim; ++d)
        sprojections[d] = field * sbasis[d];

    return sprojections;
}

template<std::size_t size, int dim>
std::array<std::array<double,size>,size>
spherical_projections
(const Tensor<2,dim>                    &field,
 const std::array<Tensor<1,dim>,size>    &sbasis)
{
    std::array<std::array<double,size>,size> sprojections;

    for (unsigned int d=0; d<dim; ++d)
        for (unsigned int e=0; e<dim; ++e)
            sprojections[d][e] = sbasis[d] * (field * sbasis[e] );

    return sprojections;
}

// explicit instantiations
template std::array<Tensor<1,2>,2> spherical_basis<2>(const std::array<double,2> &);
template std::array<Tensor<1,3>,3> spherical_basis<3>(const std::array<double,3> &);

template std::array<double,2> spherical_projections<2,2>(const Tensor<1,2> &, const std::array<Tensor<1,2>,2> &);
template std::array<double,3> spherical_projections<3,3>(const Tensor<1,3> &, const std::array<Tensor<1,3>,3> &);

template std::array<std::array<double,2>,2> spherical_projections<2,2>(const Tensor<2,2> &, const std::array<Tensor<1,2>,2> &);
template std::array<std::array<double,3>,3> spherical_projections<3,3>(const Tensor<2,3> &, const std::array<Tensor<1,3>,3> &);

}  // namespace CoordinateTransformation
