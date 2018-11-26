/*
 * assemble_system.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include "conducting_fluid_solver.h"

namespace ConductingFluid {

template<int dim>
void ConductingFluidSolver<dim>::assemble_magnetic_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble magnetic system");

    std::cout << "   Assembling magnetic system..." << std::endl;

    magnetic_matrix = 0;
    magnetic_rhs = 0;

    const QGauss<dim> quadrature(magnetic_degree+2);
    hp::QCollection<dim> q_collection;
    q_collection.push_back(quadrature);
    q_collection.push_back(quadrature);

    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values|
                                   update_quadrature_points|
                                   update_JxW_values|
                                   update_gradients);

    const QGauss<dim-1> face_quadrature(magnetic_degree+2);

    FEFaceValues<dim> int_fe_face_values(interior_magnetic_fe,
                                         face_quadrature,
                                         update_values|
                                         update_normal_vectors|
                                         update_quadrature_points|
                                         update_JxW_values|
                                         update_gradients);

    FEFaceValues<dim> ext_fe_face_values(exterior_magnetic_fe,
                                         face_quadrature,
                                         update_values);

    FESubfaceValues<dim> int_fe_subface_values(interior_magnetic_fe,
                                               face_quadrature,
                                               update_values|
                                               update_normal_vectors|
                                               update_quadrature_points|
                                               update_JxW_values|
                                               update_gradients);
    FESubfaceValues<dim> ext_fe_subface_values(exterior_magnetic_fe,
                                               face_quadrature,
                                               update_values);

    const unsigned int int_dofs_per_cell = interior_magnetic_fe.dofs_per_cell;
    const unsigned int ext_dofs_per_cell = exterior_magnetic_fe.dofs_per_cell;

    FullMatrix<double> local_matrix;
    FullMatrix<double> local_interface_matrix(int_dofs_per_cell,
                                              ext_dofs_per_cell);
    FullMatrix<double> local_interface_matrix_transposed(ext_dofs_per_cell,
                                                         int_dofs_per_cell);

    Vector<double> local_interface_rhs(int_dofs_per_cell);
    Vector<double> local_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices(ext_dofs_per_cell);

    std::vector<double>        ext_phi_values(ext_dofs_per_cell);
    std::vector<Tensor<1,dim>> ext_grad_values(ext_dofs_per_cell);

    typedef typename FEValuesViews::Vector<dim>::curl_type curl_type;
    std::vector<Tensor<1,dim>>  int_phi_values(int_dofs_per_cell);
    std::vector<curl_type>      int_curl_values(int_dofs_per_cell);

    std::vector<Tensor<1,dim>>  int_rhs_values(q_collection[0].size());

    const FEValuesExtractors::Vector vector_potential(0);
    const FEValuesExtractors::Scalar scalar_potential(dim);
}
}  // namespace ConductingFluid

template void ConductingFluid::ConductingFluidSolver<2>::assemble_magnetic_system();
template void ConductingFluid::ConductingFluidSolver<3>::assemble_magnetic_system();
