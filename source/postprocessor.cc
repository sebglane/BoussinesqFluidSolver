/*
 * postprocessor.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

//#include "buoyant_fluid_solver.h"
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

    return solution_names;
}

template<int dim>
UpdateFlags PostProcessor<dim>::get_needed_update_flags() const
{
    return update_values;
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

    return component_interpretation;
}

template <int dim>
void PostProcessor<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>>                &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcDimensionMismatch(computed_quantities.size(),
                                n_quadrature_points));
    Assert(inputs.solution_values[0].size() == dim + 3,
           ExcDimensionMismatch(inputs.solution_values[0].size(),
                                dim + 3));
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
    }
}

}  // namespace BuoyantFluid

// explicit instantiation
template class BuoyantFluid::PostProcessor<2>;
template class BuoyantFluid::PostProcessor<3>;
