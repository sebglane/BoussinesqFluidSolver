/*
 * assemble_interface.cc
 *
 *  Created on: Nov 26, 2018
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include "conducting_fluid_solver.h"
namespace ConductingFluid {

template<>
void ConductingFluidSolver<2>::assemble_magnetic_interface_term(
        const FEFaceValuesBase<2> &/* fluid_fe_face_values */,
        const FEFaceValuesBase<2> &/* vacuum_fe_face_values */,
        std::vector<typename FEValuesViews::Vector<2>::curl_type> &/* int_curl_values */,
        std::vector<double> &/* ext_phi_values */,
        FullMatrix<double> &/* local_interface_matrix */) const
{
    Assert(false, ExcNotImplemented());
}

template<>
void ConductingFluidSolver<3>::assemble_magnetic_interface_term(
        const FEFaceValuesBase<3> &fluid_fe_face_values,
        const FEFaceValuesBase<3> &vacuum_fe_face_values,
        std::vector<typename FEValuesViews::Vector<3>::curl_type>  &int_curl_values,
        std::vector<double>                                        &ext_phi_values,
        FullMatrix<double> &local_interface_matrix) const
{
    const unsigned int dim = 3;

    AssertDimension(fluid_fe_face_values.n_quadrature_points,
                    vacuum_fe_face_values.n_quadrature_points);
    AssertDimension(local_interface_matrix.m(),
                    fluid_fe_face_values.dofs_per_cell);
    AssertDimension(local_interface_matrix.n(),
                    vacuum_fe_face_values.dofs_per_cell);

    const unsigned int n_face_quadrature_points = fluid_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector vector_potential(0);
    const FEValuesExtractors::Scalar scalar_potential(dim);

    const std::vector<Tensor<1,dim>> normal_vectors = fluid_fe_face_values.get_normal_vectors();

    local_interface_matrix = 0;

    for (unsigned int q=0; q<n_face_quadrature_points; ++q)
    {
        for (unsigned int k=0; k<fluid_fe_face_values.dofs_per_cell; ++k)
            int_curl_values[k] = fluid_fe_face_values[vector_potential].curl(k, q);
        for (unsigned int k=0; k<vacuum_fe_face_values.dofs_per_cell; ++k)
            ext_phi_values[k] = vacuum_fe_face_values[scalar_potential].value(k, q);

        for (unsigned int i=0; i<fluid_fe_face_values.dofs_per_cell; ++i)
            for (unsigned int j=0; j<vacuum_fe_face_values.dofs_per_cell; ++j)
                local_interface_matrix(i, j) += equation_coefficients[0] * normal_vectors[q] * int_curl_values[i] *
                                                ext_phi_values[j] * fluid_fe_face_values.JxW(q);
    }
}

template<int dim>
void ConductingFluidSolver<dim>::distribute_magnetic_interface_term(
        const FullMatrix<double> &local_interface_matrix,
        const std::vector<types::global_dof_index> &local_fluid_dof_indices,
        const std::vector<types::global_dof_index> &local_vacuum_dof_indices)
{
    AssertDimension(local_interface_matrix.m(),
                    interior_magnetic_fe.dofs_per_cell);
    AssertDimension(local_interface_matrix.n(),
                    exterior_magnetic_fe.dofs_per_cell);

    // first distribute to (0,1)-block
    magnetic_constraints.distribute_local_to_global(
            local_interface_matrix,
            local_fluid_dof_indices,
            local_vacuum_dof_indices,
            magnetic_matrix);

    // second distribute to (1,0)-block
    FullMatrix<double> local_interface_matrix_transposed;
    local_interface_matrix_transposed.copy_transposed(local_interface_matrix);
    magnetic_constraints.distribute_local_to_global(
            local_interface_matrix_transposed,
            local_vacuum_dof_indices,
            local_fluid_dof_indices,
            magnetic_matrix);
}
}  // namespace ConductingFluid

// explicit instantiation
template
void ConductingFluid::ConductingFluidSolver<2>::distribute_magnetic_interface_term(
        const FullMatrix<double> &local_interface_matrix,
        const std::vector<types::global_dof_index> &local_fluid_dof_indices,
        const std::vector<types::global_dof_index> &local_vacuum_dof_indices);
template
void ConductingFluid::ConductingFluidSolver<3>::distribute_magnetic_interface_term(
        const FullMatrix<double> &local_interface_matrix,
        const std::vector<types::global_dof_index> &local_fluid_dof_indices,
        const std::vector<types::global_dof_index> &local_vacuum_dof_indices);

template
void ConductingFluid::ConductingFluidSolver<3>::assemble_magnetic_interface_term(
        const FEFaceValuesBase<3> &fluid_fe_face_values,
        const FEFaceValuesBase<3> &vacuum_fe_face_values,
        std::vector<typename FEValuesViews::Vector<3>::curl_type>  &int_curl_values,
        std::vector<double> &ext_phi_values,
        FullMatrix<double> &local_interface_matrix) const;
