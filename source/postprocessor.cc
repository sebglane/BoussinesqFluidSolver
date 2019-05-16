/*
 * postprocessor.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <postprocessor.h>

namespace BuoyantFluid {

template<int dim>
PostProcessor<dim>::PostProcessor(const unsigned int partition)
:
partition(partition)
{}

template<int dim>
std::vector<std::string> PostProcessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    solution_names.emplace_back("temperature");
    solution_names.emplace_back("partition");

    solution_names.emplace_back("radial_velocity");
    switch (dim)
    {
        case 2:
            solution_names.emplace_back("azimuthal_velocity");
            break;
        case 3:
            solution_names.emplace_back("polar_velocity");
            solution_names.emplace_back("azimuthal_velocity");
            break;
        default:
            Assert(false, ExcDimensionMismatch2(dim, 2, 3));
            break;
    }

    return solution_names;
}

template<int dim>
UpdateFlags PostProcessor<dim>::get_needed_update_flags() const
{
    return update_values|update_quadrature_points;
}

template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostProcessor<dim>::get_data_component_interpretation() const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    switch (dim)
    {
        case 2:
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            break;
        case 3:
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);            break;
        default:
            Assert(false, ExcDimensionMismatch2(dim, 2, 3));
            break;
    }

    return component_interpretation;
}

template<>
void PostProcessor<2>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<2>   &inputs,
        std::vector<Vector<double>>                &computed_quantities) const
{
    const unsigned int dim = 2;

    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcDimensionMismatch(computed_quantities.size(),
                                n_quadrature_points));
    Assert(computed_quantities[0].size() == 2 * dim + 3,
           ExcDimensionMismatch(computed_quantities[0].size(),
                                2 * dim + 3));
    Assert(inputs.solution_values[0].size() == dim + 2,
           ExcDimensionMismatch(inputs.solution_values[0].size(),
                                dim + 2));
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        // velocity
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d) = inputs.solution_values[q](d);
        // pressure
        computed_quantities[q](dim) = inputs.solution_values[q](dim);
        // temperature
        computed_quantities[q](dim+1) = inputs.solution_values[q](dim+1);
        // mpi partition
        computed_quantities[q](dim+2) = partition;

        // spherical coordinates
        const Point<dim>    point = inputs.evaluation_points[q];
        const double phi = atan2(point[1], point[0]);
        Assert(phi >= -numbers::PI && phi <= numbers::PI,
               ExcAzimuthalAngleRange(phi));

        // spherical basis vectors
        const Tensor<1,dim>   radial_basis_vector({cos(phi) , sin(phi)});
        const Tensor<1,dim>   azimuthal_basis_vector({-sin(phi), cos(phi)});

        // compute projection
        for (unsigned int d=0; d<dim; ++d)
        {
            computed_quantities[q](dim+3) += radial_basis_vector[d] * inputs.solution_values[q](d);
            computed_quantities[q](dim+4) += azimuthal_basis_vector[d] * inputs.solution_values[q](d);
        }
    }
}

template<>
void PostProcessor<3>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<3>   &inputs,
        std::vector<Vector<double>>                &computed_quantities) const
{
    const unsigned int dim = 3;

    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcDimensionMismatch(computed_quantities.size(),
                                n_quadrature_points));
    Assert(computed_quantities[0].size() == 2 * dim + 3,
           ExcDimensionMismatch(computed_quantities[0].size(),
                                2 * dim + 3));
    Assert(inputs.solution_values[0].size() == dim + 2,
           ExcDimensionMismatch(inputs.solution_values[0].size(),
                                dim + 2));
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        // velocity
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d) = inputs.solution_values[q](d);
        // pressure
        computed_quantities[q](dim) = inputs.solution_values[q](dim);
        // temperature
        computed_quantities[q](dim+1) = inputs.solution_values[q](dim+1);
        // mpi partition
        computed_quantities[q](dim+2) = partition;

        // spherical coordinates
        const Point<dim>    point = inputs.evaluation_points[q];
        const double cylinder_radius = sqrt(point[0]*point[0] + point[1]*point[1]);
        const double theta = atan2(cylinder_radius, point[2]);
        Assert(theta >= 0. && theta <= numbers::PI,
               ExcPolarAngleRange(theta));
        const double phi = atan2(point[1], point[0]);
        Assert(phi >= -numbers::PI && phi <= numbers::PI,
               ExcAzimuthalAngleRange(phi));

        // spherical basis vectors
        const Tensor<1,dim>   radial_basis_vector({cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)});
        const Tensor<1,dim>   polar_basis_vector({cos(phi) * cos(theta), sin(phi) * cos(theta), sin(theta)});
        const Tensor<1,dim>   azimuthal_basis_vector({-sin(phi), cos(phi) , 0.});

        // compute projection
        for (unsigned int d=0; d<dim; ++d)
        {
            computed_quantities[q](dim+3) += radial_basis_vector[d] * inputs.solution_values[q](d);
            computed_quantities[q](dim+4) += polar_basis_vector[d] * inputs.solution_values[q](d);
            computed_quantities[q](dim+5) += azimuthal_basis_vector[d] * inputs.solution_values[q](d);
        }
    }
}

}  // namespace BuoyantFluid

// explicit instantiation
template class BuoyantFluid::PostProcessor<2>;
template class BuoyantFluid::PostProcessor<3>;
