/*
 * postprocessing.templates.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_POSTPROCESSOR_TEMPLATES_H_
#define INCLUDE_POSTPROCESSOR_TEMPLATES_H_

#include "postprocessor.h"

namespace BuoyantFluid {

template<int dim>
std::vector<std::string> PostProcessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    solution_names.push_back("temperature");

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

    return component_interpretation;
}

template <int dim>
void PostProcessor<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double> >               &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert(computed_quantities.size() == n_quadrature_points,
            ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim+2,
            ExcInternalError());
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d) = inputs.solution_values[q](d);
        const double pressure = inputs.solution_values[q](dim);
        computed_quantities[q](dim) = pressure;
        const double temperature = inputs.solution_values[q](dim+1);
        computed_quantities[q](dim+1) = temperature;
    }
}

}  // namespace BuoyantFluid

#endif /* INCLUDE_POSTPROCESSOR_TEMPLATES_H_ */
