/*
 * assemble_system.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include "conducting_fluid_solver.h"
#include "grid_factory.h"

namespace ConductingFluid {

template<int dim>
void ConductingFluidSolver<dim>::assemble_magnetic_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble magnetic system");

    std::cout << "   Assembling magnetic system..." << std::endl;

    if (rebuild_magnetic_matrices)
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

    FEFaceValues<dim> fluid_fe_face_values(interior_magnetic_fe,
                                           face_quadrature,
                                           update_values|
                                           update_normal_vectors|
                                           update_quadrature_points|
                                           update_JxW_values|
                                           update_gradients);

    FEFaceValues<dim> vacuum_fe_face_values(exterior_magnetic_fe,
                                            face_quadrature,
                                            update_values);

    FESubfaceValues<dim> fluid_fe_subface_values(interior_magnetic_fe,
                                                 face_quadrature,
                                                 update_values|
                                                 update_normal_vectors|
                                                 update_quadrature_points|
                                                 update_JxW_values|
                                                 update_gradients);
    FESubfaceValues<dim> vacuum_fe_subface_values(exterior_magnetic_fe,
                                                  face_quadrature,
                                                  update_values);

    const unsigned int fluid_dofs_per_cell = interior_magnetic_fe.dofs_per_cell;
    const unsigned int vacuum_dofs_per_cell = exterior_magnetic_fe.dofs_per_cell;

    FullMatrix<double> local_matrix;
    FullMatrix<double> local_interface_matrix(fluid_dofs_per_cell,
                                              vacuum_dofs_per_cell);
    Vector<double> local_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices(vacuum_dofs_per_cell);

    std::vector<double>        vacuum_phi_values(vacuum_dofs_per_cell);
    std::vector<Tensor<1,dim>> vacuum_grad_values(vacuum_dofs_per_cell);

    typedef typename FEValuesViews::Vector<dim>::curl_type curl_type;
    std::vector<Tensor<1,dim>>  fluid_phi_values(fluid_dofs_per_cell);
    std::vector<curl_type>      fluid_curl_values(fluid_dofs_per_cell);

    std::vector<Tensor<1,dim>>  old_magnetic_values(q_collection[0].size());
    std::vector<Tensor<1,dim>>  old_old_magnetic_values(q_collection[0].size());

    std::vector<curl_type>      old_magnetic_curls(q_collection[0].size());
    std::vector<curl_type>      old_old_magnetic_curls(q_collection[0].size());

    const FEValuesExtractors::Vector vector_potential(0);
    const FEValuesExtractors::Scalar scalar_potential(dim);

    const std::vector<double> alpha = (timestep_number != 0?
                                       imex_coefficients.alpha(timestep/old_timestep):
                                       std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                       imex_coefficients.gamma(timestep/old_timestep):
                                       std::vector<double>({1.0,0.0,0.0}));

    for (auto cell: magnetic_dof_handler.active_cell_iterators())
    {
        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

        // assemble volume terms
        if (rebuild_magnetic_matrices)
            local_matrix.reinit(dofs_per_cell,
                                dofs_per_cell);
        local_rhs.reinit(dofs_per_cell);

        // fluid domain
        if (cell->material_id() == DomainIdentifiers::MaterialIds::Fluid)
        {
            Assert(dofs_per_cell == fluid_dofs_per_cell,
                   ExcInternalError());
            AssertDimension(old_magnetic_values.size(),
                            fe_values.n_quadrature_points);
            AssertDimension(old_old_magnetic_values.size(),
                            fe_values.n_quadrature_points);

            AssertDimension(old_magnetic_curls.size(),
                            fe_values.n_quadrature_points);
            AssertDimension(old_old_magnetic_curls.size(),
                            fe_values.n_quadrature_points);

            fe_values[vector_potential].get_function_values(old_magnetic_solution,
                                                            old_magnetic_values);
            fe_values[vector_potential].get_function_values(old_old_magnetic_solution,
                                                            old_old_magnetic_values);

            fe_values[vector_potential].get_function_curls(old_magnetic_solution,
                                                           old_magnetic_curls);
            fe_values[vector_potential].get_function_curls(old_old_magnetic_solution,
                                                           old_old_magnetic_curls);

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
                // pre-computation of values
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    fluid_phi_values[k] = fe_values[vector_potential].value(k, q);
                    fluid_curl_values[k] = fe_values[vector_potential].curl(k, q);
                }

                const Tensor<1,dim> time_derivative_magnetic
                    = alpha[1] * old_magnetic_values[q]
                        + alpha[2] * old_old_magnetic_values[q];

                const curl_type linear_term_magnetic
                    = gamma[1] * old_magnetic_curls[q]
                        + gamma[2] * old_old_magnetic_curls[q];

                // symmetric local matrix assembly
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    if (rebuild_magnetic_matrices)
                        for (unsigned int j=0; j<=i; ++j)
                            local_matrix(i, j) += (
                                      (1./ timestep) * fluid_phi_values[i] * fluid_phi_values[j]
                                    + 0.5 * equation_coefficients[0] * fluid_curl_values[i] * fluid_curl_values[j]
                                    ) * fe_values.JxW(q);
                    local_rhs(i) += (
                            - (1./ timestep) * time_derivative_magnetic * fluid_phi_values[i]
                            - equation_coefficients[0] * linear_term_magnetic * fluid_curl_values[i]
                            ) * fe_values.JxW(q);
                }
            }
        }
        // vacuum domain
        else if (cell->material_id() == DomainIdentifiers::MaterialIds::Vacuum &&
                    rebuild_magnetic_matrices)
        {
            Assert(dofs_per_cell == vacuum_dofs_per_cell,
                   ExcInternalError());
            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
                // pre-computation of values
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                    vacuum_grad_values[k] = fe_values[scalar_potential].gradient(k, q);
                // symmetric local matrix assembly
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    for (unsigned int j=0; j<=i; ++j)
                        local_matrix(i,j) += vacuum_grad_values[i] * vacuum_grad_values[j] * fe_values.JxW(q);
            }
        }
        else if (rebuild_magnetic_matrices)
            Assert(false, ExcInternalError());
        // symmetrize local matrix
        if (rebuild_magnetic_matrices)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i, j) = local_matrix(j, i);

        // distribute local matrix to global matrix
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        if (rebuild_magnetic_matrices)
            magnetic_constraints.distribute_local_to_global(
                    local_matrix,
                    local_rhs,
                    local_dof_indices,
                    magnetic_matrix,
                    magnetic_rhs);
        else
            magnetic_constraints.distribute_local_to_global(
                    local_rhs,
                    local_dof_indices,
                    magnetic_rhs);

        // assemble interface term
        if (cell->material_id() == DomainIdentifiers::MaterialIds::Fluid &&
                rebuild_magnetic_matrices)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (!cell->at_boundary(f))
                {
                    // neighbor in vacuum has same refinement level
                    if ((cell->neighbor(f)->level() == cell->level()) &&
                            (cell->neighbor(f)->has_children() == false) &&
                            (cell->neighbor(f)->material_id() == DomainIdentifiers::MaterialIds::Vacuum))
                    {
                        fluid_fe_face_values.reinit(cell, f);
                        vacuum_fe_face_values.reinit(cell->neighbor(f),
                                cell->neighbor_of_neighbor(f));

                        // test functions come from the fluid domain
                        assemble_magnetic_interface_term(fluid_fe_face_values,
                                                         vacuum_fe_face_values,
                                                         fluid_curl_values,
                                                         vacuum_phi_values,
                                                         local_interface_matrix);

                        // get dof indices of vacuum domain
                        cell->neighbor(f)
                            ->get_dof_indices(neighbor_dof_indices);

                        distribute_magnetic_interface_term(local_interface_matrix,
                                                           local_dof_indices,
                                                           neighbor_dof_indices);

                    }
                    // case 2: neighbor in vacuum domain is finer
                    else if ((cell->neighbor(f)->level() == cell->level()) &&
                            (cell->neighbor(f)->has_children() == true))
                    {
                        // loop over children of the neighbor
                        for (unsigned int subface=0; subface<cell->face(f)->n_children(); ++subface)
                            if (cell->neighbor_child_on_subface(f, subface)->material_id() == DomainIdentifiers::MaterialIds::Vacuum)
                            {
                                // FESubFaceValues for fluid domain
                                fluid_fe_subface_values.reinit(cell, f, subface);
                                // FEFaceValues for vacuum domain
                                vacuum_fe_face_values.reinit(cell->neighbor_child_on_subface(f, subface),
                                        cell->neighbor_of_neighbor(f));

                                // projection space is test function space of interior domain
                                assemble_magnetic_interface_term(fluid_fe_face_values,
                                                                 vacuum_fe_face_values,
                                                                 fluid_curl_values,
                                                                 vacuum_phi_values,
                                                                 local_interface_matrix);

                                // get dof indices of exterior domain
                                cell->neighbor_child_on_subface(f, subface)
                                    ->get_dof_indices(neighbor_dof_indices);

                                distribute_magnetic_interface_term(local_interface_matrix,
                                                                   local_dof_indices,
                                                                   neighbor_dof_indices);
                            }
                    }
                    // case 3: neighbor in vacuum domain is coarser
                    else if (cell->neighbor_is_coarser(f) &&
                            (cell->neighbor(f)->material_id() == DomainIdentifiers::MaterialIds::Vacuum))
                    {
                        // FEValues for fluid
                        fluid_fe_face_values.reinit(cell, f);
                        // FESubFaceValues for exterior
                        vacuum_fe_subface_values.reinit(cell->neighbor(f),
                                cell->neighbor_of_coarser_neighbor(f).first,
                                cell->neighbor_of_coarser_neighbor(f).second);

                        // projection space is test function space of interior domain
                        assemble_magnetic_interface_term(fluid_fe_face_values,
                                                         vacuum_fe_face_values,
                                                         fluid_curl_values,
                                                         vacuum_phi_values,
                                                         local_interface_matrix);

                        // get dof indices of exterior domain
                        cell->neighbor(f)
                            ->get_dof_indices(neighbor_dof_indices);

                        distribute_magnetic_interface_term(local_interface_matrix,
                                                           local_dof_indices,
                                                           neighbor_dof_indices);
                    }
                }
    }
    rebuild_magnetic_matrices = true;
}
}  // namespace ConductingFluid

// explicit instantiation
template void ConductingFluid::ConductingFluidSolver<2>::assemble_magnetic_system();
template void ConductingFluid::ConductingFluidSolver<3>::assemble_magnetic_system();
