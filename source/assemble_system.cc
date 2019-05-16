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
void ConductingFluidSolver<dim>::assemble_magnetic_matrices()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble magnetic system");

    if (rebuild_magnetic_matrices)
    {
        magnetic_matrix = 0;
        magnetic_mass_matrix = 0;
        magnetic_curl_matrix = 0;
        magnetic_stabilization_matrix = 0;
    }

    const QGauss<dim> quadrature(magnetic_degree + 1);

    FEValues<dim> fe_values(mapping,
                            magnetic_fe,
                            quadrature,
                            update_values|
                            update_quadrature_points|
                            update_JxW_values|
                            update_gradients);

    const unsigned int dofs_per_cell = magnetic_fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature.size();

    FullMatrix<double> local_matrix(dofs_per_cell,
                                    dofs_per_cell);
    FullMatrix<double> local_mass_matrix(dofs_per_cell,
                                         dofs_per_cell);
    FullMatrix<double> local_curl_matrix(dofs_per_cell,
                                         dofs_per_cell);
    FullMatrix<double> local_stabilization_matrix(dofs_per_cell,
                                                  dofs_per_cell);

    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

    typedef typename FEValuesViews::Vector<dim>::curl_type curl_type;

    std::vector<double>         div_phi_magnetic_field(dofs_per_cell);
    std::vector<Tensor<1,dim>>  phi_magnetic_field(dofs_per_cell);
    std::vector<curl_type>      curl_phi_magnetic_field(dofs_per_cell);

    std::vector<double>         phi_pseudo_pressure(dofs_per_cell);
    std::vector<Tensor<1,dim>>  grad_phi_pseudo_pressure(dofs_per_cell);

    const FEValuesExtractors::Vector magnetic_field(0);
    const FEValuesExtractors::Scalar pseudo_pressure(dim);

    const double    D = (1. - aspect_ratio);

    const std::vector<double>   tau{D * D / equation_coefficients[0],
                                    equation_coefficients[0] / D / D};

    for (auto cell: magnetic_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        const double tau_h_sqr= tau[1] * pow(cell->diameter(), 2);

        // assemble volume terms
        local_matrix = 0;
        local_mass_matrix = 0;
        local_curl_matrix = 0;
        local_stabilization_matrix = 0;

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            // pre-computation of values
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                div_phi_magnetic_field[k] = fe_values[magnetic_field].divergence(k, q);
                phi_magnetic_field[k] = fe_values[magnetic_field].value(k, q);
                curl_phi_magnetic_field[k] = fe_values[magnetic_field].curl(k, q);

                phi_pseudo_pressure[k] = fe_values[pseudo_pressure].value(k, q);
                grad_phi_pseudo_pressure[k] = fe_values[pseudo_pressure].gradient(k, q);
            }

            // symmetric local matrix assembly
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<=i; ++j)
                {
                    local_matrix(i, j) += (
                              grad_phi_pseudo_pressure[i] * phi_magnetic_field[j]
                            + phi_magnetic_field[i] * grad_phi_pseudo_pressure[j]
                            ) * fe_values.JxW(q);
                    local_mass_matrix(i, j) += (
                              phi_magnetic_field[i] * phi_magnetic_field[j]
                            ) * fe_values.JxW(q);
                    local_curl_matrix(i, j) += (
                              curl_phi_magnetic_field[i] * curl_phi_magnetic_field[j]
                            + grad_phi_pseudo_pressure[i] * grad_phi_pseudo_pressure[j]
                            ) * fe_values.JxW(q);
                }
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    local_stabilization_matrix(i, j) += (
                              tau[0] * div_phi_magnetic_field[i] * div_phi_magnetic_field[j]
                            + tau_h_sqr * grad_phi_pseudo_pressure[i] * grad_phi_pseudo_pressure[j]
                            + tau_h_sqr * phi_magnetic_field[i] * grad_phi_pseudo_pressure[j]
                            ) * fe_values.JxW(q);

            }
        }

        // symmetrize local matrix
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=i+1; j<dofs_per_cell; ++j)
            {
                local_matrix(i, j) = local_matrix(j, i);
                local_mass_matrix(i, j) = local_mass_matrix(j, i);
                local_curl_matrix(i, j) = local_curl_matrix(j, i);
            }

        // distribute local matrix to global matrix
        cell->get_dof_indices(local_dof_indices);

        magnetic_constraints.distribute_local_to_global(
                local_matrix,
                local_dof_indices,
                magnetic_matrix);

        magnetic_constraints.distribute_local_to_global(
                local_mass_matrix,
                local_dof_indices,
                magnetic_mass_matrix);

        magnetic_constraints.distribute_local_to_global(
                local_curl_matrix,
                local_dof_indices,
                magnetic_curl_matrix);

        magnetic_constraints.distribute_local_to_global(
                local_stabilization_matrix,
                local_dof_indices,
                magnetic_stabilization_matrix);
    }
    rebuild_magnetic_matrices = false;
}

template<int dim>
void ConductingFluidSolver<dim>::assemble_magnetic_rhs()
{
    std::cout << "      Assembling magnetic rhs..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble magnetic rhs");

    const QGauss<dim>   quadrature(magnetic_degree + 1);

    FEValues<dim> fe_values(mapping,
                            magnetic_fe,
                            quadrature,
                            update_values|
                            update_quadrature_points|
                            update_JxW_values|
                            update_gradients);

    const unsigned int dofs_per_cell = magnetic_fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature.size();

    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

    typedef typename FEValuesViews::Vector<dim>::curl_type curl_type;

    std::vector<Tensor<1,dim>>  phi_magnetic_field(dofs_per_cell);
    std::vector<curl_type>      curl_phi_magnetic_field(dofs_per_cell);

    std::vector<Tensor<1,dim>>  old_magnetic_values(n_q_points);
    std::vector<Tensor<1,dim>>  old_old_magnetic_values(n_q_points);

    std::vector<curl_type>      old_magnetic_curls(n_q_points);
    std::vector<curl_type>      old_old_magnetic_curls(n_q_points);

    const FEValuesExtractors::Vector magnetic_field(0);

    const std::vector<double> alpha = (timestep_number != 0?
                                       imex_coefficients.alpha(timestep/old_timestep):
                                       std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                       imex_coefficients.gamma(timestep/old_timestep):
                                       std::vector<double>({1.0,0.0,0.0}));

    for (auto cell: magnetic_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        local_rhs = 0;

        fe_values[magnetic_field].get_function_values(old_magnetic_solution,
                                                      old_magnetic_values);
        fe_values[magnetic_field].get_function_values(old_old_magnetic_solution,
                                                      old_old_magnetic_values);

        fe_values[magnetic_field].get_function_curls(old_magnetic_solution,
                                                     old_magnetic_curls);
        fe_values[magnetic_field].get_function_curls(old_old_magnetic_solution,
                                                     old_old_magnetic_curls);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            // pre-computation of values
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                phi_magnetic_field[k] = fe_values[magnetic_field].value(k, q);
                curl_phi_magnetic_field[k] = fe_values[magnetic_field].curl(k, q);
            }

            const Tensor<1,dim> time_derivative_magnetic_field
                = alpha[1] / timestep * old_magnetic_values[q]
                    + alpha[2] / timestep * old_old_magnetic_values[q];

            const curl_type linear_term_magnetic_field
                = gamma[1] * old_magnetic_curls[q]
                    + gamma[2] * old_old_magnetic_curls[q];

            // symmetric local matrix assembly
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                local_rhs(i) += (
                        - time_derivative_magnetic_field * phi_magnetic_field[i]
                        - equation_coefficients[0] * linear_term_magnetic_field * curl_phi_magnetic_field[i]
                        ) * fe_values.JxW(q);
        }

        // distribute local matrix to global matrix
        cell->get_dof_indices(local_dof_indices);

        magnetic_constraints.distribute_local_to_global(
                local_rhs,
                local_dof_indices,
                magnetic_rhs);
    }
}


template<int dim>
void ConductingFluidSolver<dim>::assemble_diffusion_system()
{
    std::cout << "      Assembling diffusion system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble diffusion system");

    if (rebuild_magnetic_matrices)
    {
        assemble_magnetic_matrices();
    }

    if (timestep_number == 0)
    // perform an Euler step
    {
        // time stepping coefficients
        const std::vector<double> alpha = std::vector<double>({1.0,-1.0,0.0});
        const std::vector<double> gamma = std::vector<double>({1.0,0.0,0.0});

        // initialize system matrix with mass matrix
        magnetic_matrix.block(0,0).copy_from(magnetic_mass_matrix.block(0,0));
        magnetic_matrix.block(0,0) *= alpha[0] / timestep;

        // add curl-curl term to system matrix
        magnetic_matrix.block(0,0).add(equation_coefficients[0] * gamma[0],
                                       magnetic_curl_matrix.block(0,0));

        // add stabilization term to system matrix
        magnetic_matrix.block(0,0).add(1.0,
                                       magnetic_stabilization_matrix.block(0,0));

        // rebuild the preconditioner of diffusion solve
        rebuild_magnetic_diffusion_preconditioner = true;
    }
    else if (timestep_number == 1 || timestep_modified)
    // perform an IMEX step
    {
        Assert(timestep_number != 0, ExcInternalError());

        // time stepping coefficients
        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        // initialize system matrix with mass matrix
        magnetic_matrix.block(0,0).copy_from(magnetic_mass_matrix.block(0,0));
        magnetic_matrix.block(0,0) *= alpha[0] / timestep;

        // add curl-curl term to system matrix
        magnetic_matrix.block(0,0).add(equation_coefficients[1] * gamma[0],
                                       magnetic_curl_matrix.block(0,0));

        // add stabilization term to system matrix
        magnetic_matrix.block(0,0).add(1.0,
                                       magnetic_stabilization_matrix.block(0,0));

        // rebuild the preconditioner of diffusion solve
        rebuild_magnetic_diffusion_preconditioner = true;
    }

    // add pressure gradient to right-hand side
    magnetic_matrix.block(0,1).vmult(magnetic_rhs.block(0),
                                     old_magnetic_solution.block(1));
    magnetic_rhs.block(0) *= -1.0;

    // assemble other terms
    assemble_magnetic_rhs();

}

template<int dim>
void ConductingFluidSolver<dim>::assemble_projection_system()
{
    std::cout << "      Assembling projection system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble projection system");

    if (rebuild_magnetic_matrices)
    {
        assemble_magnetic_matrices();
    }

    // compute stiffness matrix with PSPG term
    if (timestep_number == 0 || timestep_modified)
    {
        //
        magnetic_matrix.block(1,1).copy_from(magnetic_curl_matrix.block(1,1));

        magnetic_matrix.block(1,1).add(1. / timestep,
                                       magnetic_stabilization_matrix.block(1,1));
    }

    // compute right-hand side vector, initialize with: a0 / dt * ( B, grad(s))
    magnetic_matrix.block(1,0).vmult(magnetic_rhs.block(1),
                                     magnetic_solution.block(0));

    const std::vector<double> alpha = (timestep_number != 0?
                                       imex_coefficients.alpha(timestep/old_timestep):
                                       std::vector<double>({1.0,-1.0,0.0}));

    magnetic_rhs.block(1) *= alpha[0] / timestep;

    // add pressure laplacian to right-hand side
    magnetic_curl_matrix.block(1,1).vmult_add(magnetic_rhs.block(1),
                                              old_magnetic_solution.block(1));
}
}  // namespace ConductingFluid

// explicit instantiation
template void ConductingFluid::ConductingFluidSolver<3>::assemble_magnetic_matrices();
template void ConductingFluid::ConductingFluidSolver<3>::assemble_magnetic_rhs();
template void ConductingFluid::ConductingFluidSolver<3>::assemble_diffusion_system();
template void ConductingFluid::ConductingFluidSolver<3>::assemble_projection_system();
