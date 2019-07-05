/*
 * postprocessor.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_POSTPROCESSOR_H_
#define INCLUDE_POSTPROCESSOR_H_

#include <deal.II/numerics/data_postprocessor.h>

#include <exceptions.h>

namespace BuoyantFluid {

using namespace dealii;

using namespace GeometryExceptions;

enum OutputFlags
{
    output_values = 0x0001,
    output_scalar_gradients = 0x0002,
    output_mpi_partition = 0x0004,
    output_spherical_components = 0x0008,
    output_velocity_curl = 0x0010,
    output_magnetic_curl = 0x0020,
    output_magnetic_helicity = 0x0040,
    output_coriolis_force = 0x0080,
    output_lorentz_force = 0x0100,
    output_buoyancy_force = 0x0200,
    output_magnetic_induction = 0x0400,
    output_default = output_values
    /*
     * bit integers
     *
     * 0x0001
     * 0x0002
     * 0x0008
     * 0x0010
     * 0x0020
     * 0x0040
     * 0x0080
     * 0x0100
     * 0x0200
     * 0x0400
     * 0x0800
     * 0x1000
     * 0x2000
     * 0x4000
     * 0x10000
     * 0x100000
     * 0x200000
     * 0x400000
     * 0x800000
     * 0x1000000
     */
};

inline OutputFlags operator|
(OutputFlags    lhs,
 OutputFlags    rhs)
{
    return static_cast<OutputFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline OutputFlags& operator|=
(OutputFlags    &lhs,
 OutputFlags    rhs)
{
    return lhs = lhs | rhs;
}

template<int dim>
class PostProcessor : public DataPostprocessor<dim>
{
public:
    PostProcessor(const unsigned partition,
                  const bool     magnetism,
                  const OutputFlags output_flags = output_values);

    virtual void evaluate_vector_field
    (const DataPostprocessorInputs::Vector<dim> &inputs,
     std::vector<Vector<double>>                &computed_quantities) const;

    virtual std::vector<std::string> get_names() const;

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const;

    virtual UpdateFlags get_needed_update_flags() const;

private:
    unsigned int n_data_components() const;

    void evaluate_curl_component
    (const std::vector<Tensor<1,dim>>   &solution_gradients,
     const unsigned int                  first_vector_component,
     Vector<double>                     &computed_quantities,
     unsigned int                       &first_quantity_component) const;

    void evaluate_component_projection
    (const Vector<double>               &solution_values,
     const unsigned int                  first_vector_component,
     const Tensor<1,dim>                 spherical_basis[dim],
     Vector<double>                     &computed_quantities,
     unsigned int                       &first_quantity_component) const;

    void evaluate_gradient_projection
    (const std::vector<Tensor<1,dim>>   &solution_gradients,
     const unsigned int                  vector_component,
     const Tensor<1,dim>                 spherical_basis[dim],
     Vector<double>                     &computed_quantities,
     unsigned int                       &first_quantity_component) const;


    void compute_spherical_basis_vectors
    (const Point<dim>   &evaluation_point,
     Tensor<1,dim>       spherical_basis[dim]) const;

    const unsigned int  partition;
    const bool          magnetism;
    const OutputFlags   output_flags;

    const double        projection_tolerance = 1e-12;
};

}

#endif /* INCLUDE_POSTPROCESSOR_H_ */
