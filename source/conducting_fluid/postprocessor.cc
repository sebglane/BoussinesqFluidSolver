/*
 * postprocessor.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

//#include "buoyant_fluid_solver.h"
#include <postprocessor.h>

namespace ConductingFluid {

template<int dim>
std::vector<std::string> PostProcessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(dim, "vector_potential");
    solution_names.push_back("scalar_potential");
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("potential_curl");
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("potential_gradient");

    return solution_names;
}

template<int dim>
UpdateFlags PostProcessor<dim>::get_needed_update_flags() const
{
    return update_values|update_gradients;
}

template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostProcessor<dim>::get_data_component_interpretation() const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

    return component_interpretation;
}

template <>
void PostProcessor<3>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<3> &inputs,
        std::vector<Vector<double>>              &computed_quantities) const
{
    const unsigned int dim = 3;

    const unsigned int n_quadrature_points = inputs.solution_values.size();

    AssertDimension(computed_quantities.size(),
                    n_quadrature_points);
    AssertDimension(inputs.solution_values[0].size(),
                    dim + 1);

    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        AssertDimension(computed_quantities[q].size(), 3*dim+1);
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d] = inputs.solution_values[q][d];
        const double phi = inputs.solution_values[q][dim];
        computed_quantities[q][dim] = phi;

        computed_quantities[q][dim+1] =  inputs.solution_gradients[q][2][1]
                                        -inputs.solution_gradients[q][1][2];
        computed_quantities[q][dim+2] =  inputs.solution_gradients[q][0][2]
                                        -inputs.solution_gradients[q][2][0];
        computed_quantities[q][dim+3] =  inputs.solution_gradients[q][1][0]
                                        -inputs.solution_gradients[q][0][1];
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][2*dim+d] = inputs.solution_gradients[q][dim][d];
    }
}

}  // namespace ConductingFluid

// explicit instantiation
template class ConductingFluid::PostProcessor<3>;
