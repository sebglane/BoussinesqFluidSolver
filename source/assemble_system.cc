/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/work_stream.h>

#include "buoyant_fluid_solver.h"


namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::assemble_temperature_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble temperature system");

    std::cout << "   Assembling temperature system..." << std::endl;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    // assemble temperature matrices
    if (rebuild_temperature_matrices)
    {
        temperature_mass_matrix = 0;
        temperature_stiffness_matrix = 0;

        WorkStream::run(
                temperature_dof_handler.begin_active(),
                temperature_dof_handler.end(),
                std::bind(&BuoyantFluidSolver<dim>::local_assemble_temperature_matrix,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_temperature_matrix,
                          this,
                          std::placeholders::_1),
                TemperatureAssembly::Scratch::Matrix<dim>(temperature_fe,
                                                          mapping,
                                                          quadrature_formula),
                TemperatureAssembly::CopyData::Matrix<dim>(temperature_fe));

        const std::vector<double> alpha = (timestep_number != 0?
                                                imex_coefficients.alpha(timestep/old_timestep):
                                                std::vector<double>({1.0,-1.0,0.0}));
        const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0];
        temperature_matrix.add(timestep * gamma[0] * equation_coefficients[3],
                               temperature_stiffness_matrix);

        rebuild_temperature_matrices = false;
        rebuild_temperature_preconditioner = true;
    }
    else if (timestep_number == 1 || timestep_modified)
    {
        Assert(timestep_number != 0, ExcInternalError());

        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0];
        temperature_matrix.add(timestep * gamma[0] * equation_coefficients[3],
                               temperature_stiffness_matrix);

        rebuild_temperature_preconditioner = true;
    }
    // reset all entries
    temperature_rhs = 0;

    // assemble temperature right-hand side
    WorkStream::run(
            temperature_dof_handler.begin_active(),
            temperature_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_temperature_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_temperature_rhs,
                      this,
                      std::placeholders::_1),
            TemperatureAssembly::Scratch::RightHandSide<dim>(temperature_fe,
                                                             mapping,
                                                             quadrature_formula,
                                                             update_values|
                                                             update_gradients|
                                                             update_JxW_values,
                                                             navier_stokes_fe,
                                                             update_values),
            TemperatureAssembly::CopyData::RightHandSide<dim>(temperature_fe));
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_navier_stokes_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble stokes system");

    std::cout << "      Assembling Navier-Stokes system..." << std::endl;

    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);

    if (rebuild_navier_stokes_matrices)
    {
        // reset all entries
        navier_stokes_matrix = 0;
        velocity_laplace_matrix = 0;

        FEValues<dim>   fe_values(mapping,
                                  navier_stokes_fe,
                                  quadrature_formula,
                                  update_values|
                                  update_gradients|
                                  update_JxW_values);

        const unsigned int velocity_dofs_per_cell = navier_stokes_fe.base_element(0).dofs_per_cell;
        Assert(velocity_dofs_per_cell > 0, ExcLowerRange(velocity_dofs_per_cell, 0));

        const unsigned int dofs_per_cell = navier_stokes_fe.dofs_per_cell;
        const unsigned int n_q_points    = fe_values.n_quadrature_points;
        Assert(velocity_dofs_per_cell < dofs_per_cell, ExcInternalError());

        const FEValuesExtractors::Vector    velocity(0);
        const FEValuesExtractors::Scalar    pressure(dim);


        FullMatrix<double>  local_matrix(dofs_per_cell,
                                         dofs_per_cell),
                            local_laplace_matrix(velocity_dofs_per_cell,
                                                 velocity_dofs_per_cell);

        std::vector<double>         div_phi_velocity(dofs_per_cell);
        std::vector<Tensor<1,dim>>  phi_velocity(dofs_per_cell);
        std::vector<Tensor<2,dim>>  grad_phi_velocity(velocity_dofs_per_cell);

        std::vector<double>         phi_pressure(dofs_per_cell);
        std::vector<Tensor<1,dim>>  grad_phi_pressure(dofs_per_cell);

        std::vector<types::global_dof_index>   local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index>   local_velocity_dof_indices(velocity_dofs_per_cell);

        for (auto cell: navier_stokes_dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

            cell->get_dof_indices(local_dof_indices);

            local_matrix = 0;
            local_laplace_matrix = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0, k_velocity=0; k<dofs_per_cell; ++k)
                {
                    phi_velocity[k]     = fe_values[velocity].value(k, q);
                    div_phi_velocity[k] = fe_values[velocity].divergence(k, q);
                    phi_pressure[k]     = fe_values[pressure].value(k, q);
                    grad_phi_pressure[k]= fe_values[pressure].gradient(k, q);
                    if (navier_stokes_fe.system_to_component_index(k).first < dim)
                    {
                        grad_phi_velocity[k_velocity] = fe_values[velocity].gradient(k, q);
                        local_velocity_dof_indices[k_velocity] = local_dof_indices[k];
                        ++k_velocity;
                    }
                }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                    for (unsigned int j=0; j<=i; ++j)
                        local_matrix(i,j)
                            += (
                                  phi_velocity[i] * phi_velocity[j]
                                - phi_pressure[i] * div_phi_velocity[j]
                                - div_phi_velocity[i] * phi_pressure[j]
                                + grad_phi_pressure[i] *grad_phi_pressure[j]
                                ) * fe_values.JxW(q);
                for (unsigned int i=0; i<velocity_dofs_per_cell; ++i)
                    for (unsigned int j=0; j<=i; ++j)
                        local_laplace_matrix(i,j)
                            +=   scalar_product(grad_phi_velocity[i], grad_phi_velocity[j])
                               * fe_values.JxW(q);
            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);

            for (unsigned int i=0; i<velocity_dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<velocity_dofs_per_cell; ++j)
                    local_laplace_matrix(i,j) = local_laplace_matrix(j,i);

            navier_stokes_constraints.distribute_local_to_global(local_matrix,
                                                                 local_dof_indices,
                                                                 navier_stokes_matrix);
            navier_stokes_constraints.distribute_local_to_global(local_laplace_matrix,
                                                                 local_velocity_dof_indices,
                                                                 velocity_laplace_matrix);
        }

        // assemble matrix
        /*
         *
        WorkStream::run(
                navier_stokes_dof_handler.begin_active(),
                navier_stokes_dof_handler.end(),
                std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_matrix,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix,
                          this,
                          std::placeholders::_1),
                NavierStokesAssembly::Scratch::Matrix<dim>(
                        navier_stokes_fe,
                        mapping,
                        quadrature_formula,
                        update_values|
                        update_gradients|
                        update_JxW_values),
                NavierStokesAssembly::CopyData::Matrix<dim>(navier_stokes_fe));
         *
         */

        // copy velocity mass matrix
        velocity_mass_matrix.copy_from(navier_stokes_matrix.block(0,0));

        // time stepping coefficients
        const std::vector<double> alpha = (timestep_number != 0?
                                            imex_coefficients.alpha(timestep/old_timestep):
                                            std::vector<double>({1.0,-1.0,0.0}));
        const std::vector<double> gamma = (timestep_number != 0?
                                            imex_coefficients.gamma(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0,0.0}));
        // correct (0,0)-block of stokes system
        navier_stokes_matrix.block(0,0) *= alpha[0];
        navier_stokes_matrix.block(0,0).add(timestep * equation_coefficients[1] * gamma[0],
                                            velocity_laplace_matrix);

        // rebuild the preconditioner of both preconditioners
        rebuild_diffusion_preconditioner = true;
        rebuild_projection_preconditioner = true;

        // do not rebuild stokes matrices again
        rebuild_navier_stokes_matrices = false;
    }
    else if (timestep_number == 1 || timestep_modified)
    {
        Assert(timestep_number != 0, ExcInternalError());

        // time stepping coefficients
        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        // correct (0,0)-block of navier stokes system
        navier_stokes_matrix.block(0,0).copy_from(velocity_mass_matrix);
        navier_stokes_matrix.block(0,0) *= alpha[0];
        navier_stokes_matrix.block(0,0).add(timestep * equation_coefficients[1] * gamma[0],
                                            velocity_laplace_matrix);

        // rebuild the preconditioner of the velocity block
        rebuild_diffusion_preconditioner = true;
    }
    // reset all entries
    navier_stokes_rhs = 0;

    // assemble right-hand side function
    WorkStream::run(
            navier_stokes_dof_handler.begin_active(),
            navier_stokes_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs,
                      this,
                      std::placeholders::_1),
            NavierStokesAssembly::Scratch::RightHandSide<dim>(
                    navier_stokes_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_quadrature_points|
                    update_JxW_values|
                    update_gradients,
                    temperature_fe,
                    update_values),
            NavierStokesAssembly::CopyData::RightHandSide<dim>(navier_stokes_fe));
}

}  // namespace BuoyantFluid

// explicit instantiation

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_temperature_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_navier_stokes_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_navier_stokes_system();
