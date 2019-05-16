/*
 * assembly_data.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_DATA_H_
#define INCLUDE_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include "initial_values.h"

#include <vector>

namespace TemperatureAssembly {

using namespace dealii;

namespace Scratch {

template<int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim>  &temperature_fe,
                  const Mapping<dim>        &mapping,
                  const Quadrature<dim>     &temperature_quadrature,
                  const UpdateFlags          temperature_update_flags,
                  const FiniteElement<dim>  &stokes_fe,
                  const UpdateFlags          stokes_update_flags,
                  const std::vector<double> &alpha,
                  const std::vector<double> &beta,
                  const std::vector<double> &gamma);

    RightHandSide(const RightHandSide<dim> &scratch);

    FEValues<dim>               temperature_fe_values;
    std::vector<double>         phi_temperature;
    std::vector<Tensor<1,dim>>  grad_phi_temperature;
    std::vector<double>         old_temperature_values;
    std::vector<double>         old_old_temperature_values;
    std::vector<Tensor<1,dim>>  old_temperature_gradients;
    std::vector<Tensor<1,dim>>  old_old_temperature_gradients;

    FEValues<dim>               stokes_fe_values;
    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;

    const std::vector<double>   alpha;
    const std::vector<double>   beta;
    const std::vector<double>   gamma;

    const unsigned int          dofs_per_cell;
    const unsigned int          n_q_points;

    const FEValuesExtractors::Vector    velocity;
};

template<int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &temperature_fe,
           const Mapping<dim>       &mapping,
           const Quadrature<dim>    &temperature_quadrature,
           const UpdateFlags         temperature_update_flags);

    Matrix(const Matrix<dim>  &scratch);

    FEValues<dim>               fe_values;

    std::vector<double>         phi;
    std::vector<Tensor<1,dim>>  grad_phi;
};

}  // namespace Scratch

namespace CopyData {

template <int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim>    &temperature_fe);
    RightHandSide(const RightHandSide<dim>    &data);

    Vector<double>                          local_rhs;
    FullMatrix<double>                      matrix_for_bc;
    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &temperature_fe);
    Matrix(const Matrix<dim>        &data);

    FullMatrix<double>      local_mass_matrix;
    FullMatrix<double>      local_laplace_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

}  // namespace CopyData

}  // namespace TemperatureAssembly


namespace NavierStokesAssembly {

using namespace dealii;

namespace Scratch {

template<int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &stokes_fe,
           const Mapping<dim>       &mapping,
           const Quadrature<dim>    &stokes_quadrature,
           const UpdateFlags        stokes_update_flags);

    Matrix(const Matrix<dim>  &scratch);

    FEValues<dim>               stokes_fe_values;

    std::vector<double>         div_phi_velocity;
    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;

    std::vector<double>         phi_pressure;
    std::vector<Tensor<1,dim>>  grad_phi_pressure;
};

template<int dim>
struct ConvectionMatrix
{
    ConvectionMatrix(const FiniteElement<dim>   &stokes_fe,
                     const Mapping<dim>         &mapping,
                     const Quadrature<dim>      &stokes_quadrature,
                     const UpdateFlags          stokes_update_flags);

    ConvectionMatrix(const ConvectionMatrix<dim>  &scratch);

    FEValues<dim>               stokes_fe_values;

    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;

    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;
    std::vector<double>         old_velocity_divergences;
    std::vector<double>         old_old_velocity_divergences;

};


template<int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim>  &stokes_fe,
                  const Mapping<dim>        &mapping,
                  const Quadrature<dim>     &stokes_quadrature,
                  const UpdateFlags          stokes_update_flags,
                  const FiniteElement<dim>  &temperature_fe,
                  const UpdateFlags          temperature_update_flags,
                  const std::vector<double> &alpha,
                  const std::vector<double> &beta,
                  const std::vector<double> &gamma);

    RightHandSide(const RightHandSide<dim>  &scratch);

    FEValues<dim>               stokes_fe_values;

    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;
    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;
    std::vector<Tensor<2,dim>>  old_velocity_gradients;
    std::vector<Tensor<2,dim>>  old_old_velocity_gradients;

    FEValues<dim>               temperature_fe_values;
    std::vector<double>         old_temperature_values;
    std::vector<double>         old_old_temperature_values;

    const std::vector<double>   alpha;
    const std::vector<double>   beta;
    const std::vector<double>   gamma;

    const EquationData::GravityFunction<dim> gravity_function;
    std::vector<Tensor<1,dim>>  gravity_vectors;

    const unsigned int          dofs_per_cell;
    const unsigned int          n_q_points;

    const FEValuesExtractors::Vector    velocity;

};


}  // namespace Scratch

namespace CopyData {

template <int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &navier_stokes_fe);
    Matrix(const Matrix<dim>        &data);

    FullMatrix<double>      local_matrix;
    FullMatrix<double>      local_mass_matrix;
    FullMatrix<double>      local_laplace_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

template <int dim>
struct ConvectionMatrix
{
    ConvectionMatrix(const FiniteElement<dim> &navier_stokes_fe);
    ConvectionMatrix(const ConvectionMatrix<dim>        &data);

    FullMatrix<double>      local_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

template <int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim> &navier_stokes_fe);
    RightHandSide(const RightHandSide<dim> &data);

    Vector<double>          local_rhs;

    std::vector<types::global_dof_index>   local_dof_indices;
};

}  // namespace Copy

}  // namespace NavierStokesAssembly


#endif /* INCLUDE_ASSEMBLY_DATA_H_ */
