/*
 * postprocessor.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <postprocessor.h>

#include <geometric_utilities.h>

namespace BuoyantFluid {

template<int dim>
PostProcessor<dim>::PostProcessor
(const unsigned int partition,
 const bool         magnetism,
 const OutputFlags  output_flags)
:
partition(partition),
magnetism(magnetism),
output_flags(output_flags)
{
    Assert(
       !((output_flags & output_magnetic_helicity)|
         (output_flags & output_coriolis_force)|
         (output_flags & output_lorentz_force) |
         (output_flags & output_buoyancy_force)|
         (output_flags & output_magnetic_induction)),
         ExcNotImplemented());
}

template<int dim>
unsigned int PostProcessor<dim>::n_data_components() const
{
    unsigned int n_components = 0;

    if (output_flags & output_values)
    {
        // velocity
        n_components += dim;
        // pressure
        n_components += 1;
        // temperature
        n_components += 1;

        if (magnetism)
        {
            // magnetic_field
            n_components += dim;
            // magnetic_pressure
            n_components += 1;
        }
    }

    // MPI partition
    if (output_flags & output_mpi_partition)
       n_components += 1;

    // spherical spherical components of velocity field
    if (output_flags & output_spherical_components)
    {
        // velocity
        n_components += dim;
        // magnetic field
        if (magnetism)
            n_components += dim;
    }
    // vorticity
    if (output_flags & output_velocity_curl)
    {
        switch (dim)
        {
            case 2:
                n_components += 1;
                break;
            case 3:
                n_components += dim;
                break;
            default:
                Assert(false, ExcImpossibleInDim(dim));
                break;
        }
    }
    // gradients of scalar fields
    if (output_flags & output_scalar_gradients)
    {
        // temperature gradient
        n_components += dim;
        // pressure gradient
        n_components += dim;
        // magnetic pressure gradient
        if (magnetism)
            n_components += dim;
    }
    // spherical components of gradients of scalar fields
    if ((output_flags & output_scalar_gradients) &&
        (output_flags & output_spherical_components))
    {
        // pressure gradient
        n_components += dim;

        // temperature gradient
        n_components += dim;

        // magnetic pressure gradient
        if (magnetism)
            n_components += dim;
    }
    // curl of magnetic field
    if (output_flags & output_magnetic_curl)
        if (magnetism)
            switch (dim)
            {
                case 2:
                    n_components += 1;
                    break;
                case 3:
                    n_components += dim;
                    break;
                default:
                    Assert(false, ExcDimensionMismatch2(dim, 2, 3));
                    break;
            }

    return n_components;
}

template<int dim>
std::vector<std::string> PostProcessor<dim>::get_names() const
{
    unsigned int n_components = 0;

    std::vector<std::string> solution_names;

    if (output_flags & output_values)
    {
        for (unsigned int d=0; d<dim; ++d)
            solution_names.emplace_back("velocity");
        n_components += dim;

        solution_names.emplace_back("pressure");
        n_components += 1;

        solution_names.emplace_back("temperature");
        n_components += 1;

        if (magnetism)
        {
            for (unsigned int d=0; d<dim; ++d)
                solution_names.emplace_back("magnetic_field");
            n_components += dim;

            solution_names.emplace_back("magnetic_pressure");
            n_components += 1;
        }
    }

    // MPI partition
    if (output_flags & output_mpi_partition)
    {
        solution_names.emplace_back("partition");
        n_components += 1;
    }

    // spherical spherical components of velocity field
    if (output_flags & output_spherical_components)
    {
        solution_names.emplace_back("velocity_radial");
        switch (dim)
        {
            case 2:
                solution_names.emplace_back("velocity_azimuthal");
                break;
            case 3:
                solution_names.emplace_back("velocity_polar");
                solution_names.emplace_back("velocity_azimuthal");
                break;
            default:
                Assert(false, ExcImpossibleInDim(dim));
                break;
        }
        n_components += dim;

        if (magnetism)
        {
            solution_names.emplace_back("magnetic_field_radial");
            switch (dim)
            {
                case 2:
                    solution_names.emplace_back("magnetic_field_azimuthal");
                    break;
                case 3:
                    solution_names.emplace_back("magnetic_field_polar");
                    solution_names.emplace_back("magnetic_field_azimuthal");
                    break;
                default:
                    Assert(false, ExcDimensionMismatch2(dim, 2, 3));
                    break;
            }
            n_components += dim;
        }
    }
    // vorticity
    if (output_flags & output_velocity_curl)
    {
        switch (dim)
        {
            case 2:
                solution_names.emplace_back("vorticity");
                n_components += 1;
                break;
            case 3:
                for (unsigned int d=0; d<dim; ++d)
                    solution_names.emplace_back("vorticity");
                n_components += dim;
                break;
            default:
                Assert(false, ExcImpossibleInDim(dim));
                break;
        }
    }
    // gradients of scalar fields
    if (output_flags & output_scalar_gradients)
    {
        for (unsigned int d=0; d<dim; ++d)
            solution_names.emplace_back("temperature_gradient");
        n_components += dim;

        for (unsigned int d=0; d<dim; ++d)
            solution_names.emplace_back("pressure_gradient");
        n_components += dim;

        if (magnetism)
        {
            for (unsigned int d=0; d<dim; ++d)
                solution_names.emplace_back("magnetic_pressure_gradient");
            n_components += dim;
        }
    }
    // spherical components of gradients of scalar fields
    if ((output_flags & output_scalar_gradients) &&
        (output_flags & output_spherical_components))
    {
        solution_names.emplace_back("pressure_gradient_radial");
        switch (dim)
        {
            case 2:
                solution_names.emplace_back("pressure_gradient_azimuthal");
                break;
            case 3:
                solution_names.emplace_back("pressure_gradient_polar");
                solution_names.emplace_back("pressure_gradient_azimuthal");
                break;
            default:
                Assert(false, ExcImpossibleInDim(dim));
                break;
        }
        n_components += dim;

        solution_names.emplace_back("temperature_gradient_radial");
        switch (dim)
        {
            case 2:
                solution_names.emplace_back("temperature_gradient_azimuthal");
                break;
            case 3:
                solution_names.emplace_back("temperature_gradient_polar");
                solution_names.emplace_back("temperature_gradient_azimuthal");
                break;
            default:
                Assert(false, ExcImpossibleInDim(dim));
                break;
        }
        n_components += dim;

        if (magnetism)
        {
            solution_names.emplace_back("magnetic_pressure_gradient_radial");
            switch (dim)
            {
                case 2:
                    solution_names.emplace_back("magnetic_pressure_gradient_azimuthal");
                    break;
                case 3:
                    solution_names.emplace_back("magnetic_pressure_gradient_polar");
                    solution_names.emplace_back("magnetic_pressure_gradient_azimuthal");
                    break;
                default:
                    Assert(false, ExcImpossibleInDim(dim));
                    break;
            }
            n_components += dim;
        }

    }
    // curl of magnetic field
    if (output_flags & output_magnetic_curl)
        if (magnetism)
        {
            switch (dim)
            {
                case 2:
                    solution_names.emplace_back("curl_magnetic_field");
                    n_components += 1;
                    break;
                case 3:
                    for (unsigned int d=0; d<dim; ++d)
                        solution_names.emplace_back("curl_magnetic_field");
                    n_components += dim;
                    break;
                default:
                    Assert(false, ExcDimensionMismatch2(dim, 2, 3));
                    break;
            }
        }

    AssertDimension(n_data_components(), n_components);
    AssertDimension(solution_names.size(), n_components);

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
    component_interpretation;

    if (output_flags & output_values)
    {
        // velocity
        for (unsigned int d=0; d<dim; ++d)
            component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
        // pressure
        component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // temperature
        component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        if (magnetism)
        {
            // magnetic field
            for (unsigned int d=0; d<dim; ++d)
                component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
            // magnetic pressure
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        }
    }

    // MPI partition
    if (output_flags & output_mpi_partition)
        component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    if (output_flags & output_spherical_components)
    {
        // spherical spherical components of velocity field
        for (unsigned int d=0; d<dim; ++d)
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // spherical spherical components of magnetic field
        if (magnetism)
            for (unsigned int d=0; d<dim; ++d)
                component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }
    // vorticity
    if (output_flags & output_velocity_curl)
        switch (dim)
        {
            case 2:
                component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
                break;
            case 3:
                for (unsigned int d=0; d<dim; ++d)
                    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
                break;
            default:
                Assert(false, ExcImpossibleInDim(dim));
                break;
        }
    // gradients of scalar fields
    if (output_flags & output_scalar_gradients)
    {
        for (unsigned int d=0; d<dim; ++d)
            component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
        for (unsigned int d=0; d<dim; ++d)
            component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

        if (magnetism)
            for (unsigned int d=0; d<dim; ++d)
                component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    }
    // spherical components of gradients of scalar fields
    if ((output_flags & output_scalar_gradients) &&
        (output_flags & output_spherical_components))
    {
        // pressure gradient
        for (unsigned int d=0; d<dim; ++d)
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // temperature gradient
        for (unsigned int d=0; d<dim; ++d)
            component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // magnetic pressure gradient
        if (magnetism)
            for (unsigned int d=0; d<dim; ++d)
                component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }
    // curl of magnetic field
    if (output_flags & output_magnetic_curl)
        if (magnetism)
            switch (dim)
            {
            case 2:
                component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
                break;
            case 3:
                for (unsigned int d=0; d<dim; ++d)
                    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
                break;
            default:
                Assert(false, ExcDimensionMismatch2(dim, 2, 3));
                break;
            }

    AssertDimension(component_interpretation.size(),n_data_components());

    return component_interpretation;
}

template<int dim>
void PostProcessor<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<Vector<double>>                &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(computed_quantities.size() == n_quadrature_points,
           ExcDimensionMismatch(computed_quantities.size(),
                                n_quadrature_points));
    Assert(computed_quantities[0].size() == n_data_components(),
           ExcDimensionMismatch(computed_quantities[0].size(),
                                n_data_components()));
    Assert(inputs.solution_values[0].size() == (magnetism? 2*dim+3: dim + 2),
           ExcDimensionMismatch(inputs.solution_values[0].size(),
                                (magnetism? 2*dim+3: dim + 2)));
    Assert(inputs.solution_gradients[0].size() == (magnetism? 2*dim+3: dim + 2),
           ExcDimensionMismatch(inputs.solution_gradients[0].size(),
                                (magnetism? 2*dim+3: dim + 2)));

    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {

        const std::array<double,dim> scoord
        = GeometricUtilities::Coordinates::to_spherical(inputs.evaluation_points[q]);

        std::array<Tensor<1,dim>,dim> spherical_basis_vectors
        = CoordinateTransformation::spherical_basis(scoord);

        unsigned int quantity_index = 0;

        if (output_flags & output_values)
        {
            // velocity
            for (unsigned int d=0; d<dim; ++d)
                computed_quantities[q](quantity_index+d) = inputs.solution_values[q](d);
            quantity_index += dim;
            // pressure
            computed_quantities[q](quantity_index) = inputs.solution_values[q](dim);
            quantity_index += 1;
            // temperature
            computed_quantities[q](quantity_index) = inputs.solution_values[q](dim+1);
            quantity_index += 1;

            if (magnetism)
            {
                // magnetic field
                for (unsigned int d=0; d<dim; ++d)
                    computed_quantities[q](quantity_index+d) = inputs.solution_values[q](dim+2+d);
                quantity_index += dim;
                // magnetic pseudo pressure
                computed_quantities[q](quantity_index) = inputs.solution_values[q](2*dim+2);
                quantity_index += 1;
            }
        }

        // MPI partition
        if (output_flags & output_mpi_partition)
        {
            computed_quantities[q](quantity_index) = partition;
            quantity_index += 1;
        }


        if (output_flags & output_spherical_components)
        {
            // spherical spherical components of velocity field
            evaluate_component_projection(inputs.solution_values[q],
                                          0,
                                          spherical_basis_vectors,
                                          computed_quantities[q],
                                          quantity_index);
            // spherical components of magnetic field
            if (magnetism)
                evaluate_component_projection(inputs.solution_values[q],
                                              dim+2,
                                              spherical_basis_vectors,
                                              computed_quantities[q],
                                              quantity_index);

        }
        // vorticity
        if (output_flags & output_velocity_curl)
            evaluate_curl_component(inputs.solution_gradients[q],
                                    0,
                                    computed_quantities[q],
                                    quantity_index);
        // gradients of scalar fields
        if (output_flags & output_scalar_gradients)
        {
            // pressure gradient
            for (unsigned int d=0; d<dim; ++d)
                computed_quantities[q](quantity_index) = inputs.solution_gradients[q][dim][d];
            quantity_index += dim;
            // temperature gradient
            for (unsigned int d=0; d<dim; ++d)
                computed_quantities[q](quantity_index) = inputs.solution_gradients[q][dim+1][d];
            quantity_index += dim;
            // magnetic pressure gradient
            if (magnetism)
            {
                for (unsigned int d=0; d<dim; ++d)
                    computed_quantities[q](quantity_index) = inputs.solution_gradients[q][2*dim+2][d];
                quantity_index += dim;
            }
        }
        // spherical components of gradients of scalar fields
        if ((output_flags & output_scalar_gradients) &&
            (output_flags & output_spherical_components))
        {
            // spherical components of pressure gradient
            evaluate_gradient_projection(inputs.solution_gradients[q],
                                         dim,
                                         spherical_basis_vectors,
                                         computed_quantities[q],
                                         quantity_index);
            // spherical components of temperature gradient
            evaluate_gradient_projection(inputs.solution_gradients[q],
                                         dim+1,
                                         spherical_basis_vectors,
                                         computed_quantities[q],
                                         quantity_index);
            // spherical components of magnetic pressure gradient
            if (magnetism)
                evaluate_gradient_projection(inputs.solution_gradients[q],
                                             2*dim+2,
                                             spherical_basis_vectors,
                                             computed_quantities[q],
                                             quantity_index);
        }
        // curl of magnetic field
        if ((output_flags & output_magnetic_curl) && magnetism)
            evaluate_curl_component(inputs.solution_gradients[q],
                                    2*dim+2,
                                    computed_quantities[q],
                                    quantity_index);

        AssertDimension(quantity_index, n_data_components());
    }
}

template<int dim>
void PostProcessor<dim>::evaluate_curl_component
(const std::vector<Tensor<1,dim>>   &solution_gradients,
 const unsigned int                  first_vector_component,
 Vector<double>                     &computed_quantities,
 unsigned int                       &first_quantity_component) const
{
    switch (dim)
    {
        case 2:
            computed_quantities(first_quantity_component)
            = solution_gradients[first_vector_component+1][0]
            - solution_gradients[first_vector_component][1];
            first_quantity_component += 1;
            break;
        case 3:
            computed_quantities(first_quantity_component)
            = solution_gradients[first_vector_component+2][first_vector_component+1]
            - solution_gradients[first_vector_component+1][first_vector_component+2];
            computed_quantities(first_quantity_component+1)
            = solution_gradients[first_vector_component][first_vector_component+2]
            - solution_gradients[first_vector_component+2][first_vector_component];
            computed_quantities(first_quantity_component+2)
            = solution_gradients[first_vector_component+1][first_vector_component]
            - solution_gradients[first_vector_component][first_vector_component+1];
            first_quantity_component += dim;
            break;
        default:
            Assert(false, ExcImpossibleInDim(dim));
            break;
    }
}

template<int dim>
void PostProcessor<dim>::evaluate_component_projection
(const Vector<double>                &solution_values,
 const unsigned int                   first_vector_component,
 const std::array<Tensor<1,dim>,dim> &spherical_basis,
 Vector<double>                      &computed_quantities,
 unsigned int                        &first_quantity_index) const
{
    Assert(solution_values.size() >= first_vector_component + dim,
           ExcLowerRange(solution_values.size(),
                         first_vector_component + dim));

    Assert(computed_quantities.size() >= first_quantity_index + dim,
           ExcLowerRange(computed_quantities.size(),
                         first_quantity_index + dim));

    Tensor<1,dim>   component;
    for (unsigned int d=0; d<dim; ++d)
        component[d] = solution_values(first_vector_component+d);

    const double component_magnitude = component.norm();

    for (unsigned int d=0; d<dim; ++d)
    {
        computed_quantities(first_quantity_index+d) = spherical_basis[d] * component;

        Assert(std::abs(computed_quantities(first_quantity_index+d)) 
               <= component_magnitude + projection_tolerance,
               ExcLowerRangeType<double>(computed_quantities(first_quantity_index+d),
                                         component_magnitude));
    }
    first_quantity_index += dim;
}

template<int dim>
void PostProcessor<dim>::evaluate_gradient_projection
(const std::vector<Tensor<1,dim>>    &solution_gradients,
 const unsigned int                   solution_component,
 const std::array<Tensor<1,dim>,dim> &spherical_basis,
 Vector<double>                      &computed_quantities,
 unsigned int                        &first_quantity_index) const
{
    Assert(solution_gradients.size() >= solution_component,
           ExcLowerRange(solution_gradients.size(),
                         solution_component + dim));

    Assert(computed_quantities.size() >= first_quantity_index + dim,
           ExcLowerRange(computed_quantities.size(),
                         first_quantity_index + dim));

    const double component_magnitude = solution_gradients[solution_component].norm();

    for (unsigned int c=0; c<dim; ++c)
    {
        computed_quantities(first_quantity_index+c)
        = spherical_basis[c] * solution_gradients[solution_component];

        Assert(std::abs(computed_quantities(first_quantity_index+c))
               <= component_magnitude + projection_tolerance,
               ExcLowerRangeType<double>(std::abs(computed_quantities(first_quantity_index+c)),
                       component_magnitude));
    }
    first_quantity_index += dim;
}
}  // namespace BuoyantFluid

// explicit instantiation
template class BuoyantFluid::PostProcessor<2>;
template class BuoyantFluid::PostProcessor<3>;
