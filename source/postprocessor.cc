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
    switch (dim)
    {
        case 2:
            solution_names.emplace_back("vorticity");
            break;
        case 3:
            for (unsigned int d=0; d<3; ++d)
                solution_names.emplace_back("vorticity");
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
    return update_values|update_gradients|update_quadrature_points;
}

template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostProcessor<dim>::get_data_component_interpretation() const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>

    // velocity
    component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    // pressure
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // temperature
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // mpi partition
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // radial velocity
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    switch (dim)
    {
        case 2:
            // azimuthal velocity
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            break;
        case 3:
            // azimuthal velocity
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            // polar velocity
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            break;
        default:
            Assert(false, ExcDimensionMismatch2(dim, 2, 3));
            break;
    }

    // vorticity
    switch (dim)
    {
        case 2:
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            break;
        case 3:
            component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
            break;
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
    Assert(computed_quantities[0].size() == 2 * dim + 4,
           ExcDimensionMismatch(computed_quantities[0].size(),
                                2 * dim + 4));
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
        double velocity_magnitude = 0;
        computed_quantities[q](dim+3) = 0;
        computed_quantities[q](dim+4) = 0;
        for (unsigned int d=0; d<dim; ++d)
        {
            computed_quantities[q](dim+3) += radial_basis_vector[d] * inputs.solution_values[q](d);
            computed_quantities[q](dim+4) += azimuthal_basis_vector[d] * inputs.solution_values[q](d);
            velocity_magnitude += inputs.solution_values[q](d) * inputs.solution_values[q](d);
        }
        velocity_magnitude = sqrt(velocity_magnitude);

        Assert(computed_quantities[q](dim+3) <= velocity_magnitude + 1e-12,
               ExcLowerRangeType<double>(computed_quantities[q](dim+3),
                                         velocity_magnitude));
        Assert(computed_quantities[q](dim+4) <= velocity_magnitude + 1e-12,
               ExcLowerRangeType<double>(computed_quantities[q](dim+4),
                                         velocity_magnitude));
        // vorticity
        computed_quantities[q](2*dim+3) = inputs.solution_gradients[q][1][0]
                                      - inputs.solution_gradients[q][0][1];
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
    Assert(computed_quantities[0].size() == 3 * dim + 3,
           ExcDimensionMismatch(computed_quantities[0].size(),
                                3 * dim + 3));
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

        // compute projection of spherical components
        double velocity_magnitude = 0;
        computed_quantities[q](dim+3) = 0;
        computed_quantities[q](dim+4) = 0;
        computed_quantities[q](dim+5) = 0;
        for (unsigned int d=0; d<dim; ++d)
        {
            computed_quantities[q](dim+3) += radial_basis_vector[d] * inputs.solution_values[q](d);
            computed_quantities[q](dim+4) += polar_basis_vector[d] * inputs.solution_values[q](d);
            computed_quantities[q](dim+5) += azimuthal_basis_vector[d] * inputs.solution_values[q](d);
            velocity_magnitude += inputs.solution_values[q](d) * inputs.solution_values[q](d);
        }
        velocity_magnitude = sqrt(velocity_magnitude);

        Assert(computed_quantities[q](dim+3) <= velocity_magnitude,
               ExcLowerRangeType<double>(computed_quantities[q](dim+3),
                                         velocity_magnitude));
        Assert(computed_quantities[q](dim+4) <= velocity_magnitude,
               ExcLowerRangeType<double>(computed_quantities[q](dim+4),
                                         velocity_magnitude));
        Assert(computed_quantities[q](dim+5) <= velocity_magnitude,
               ExcLowerRangeType<double>(computed_quantities[q](dim+5),
                                         velocity_magnitude));

        // vorticity
        computed_quantities[q](dim+6) = inputs.solution_gradients[q][2][1]
                                      - inputs.solution_gradients[q][1][2];
        computed_quantities[q](dim+7) = inputs.solution_gradients[q][0][2]
                                      - inputs.solution_gradients[q][2][0];
        computed_quantities[q](dim+8) = inputs.solution_gradients[q][1][0]
                                      - inputs.solution_gradients[q][0][1];
    }
}

}  // namespace BuoyantFluid

// explicit instantiation
template class BuoyantFluid::PostProcessor<2>;
template class BuoyantFluid::PostProcessor<3>;
