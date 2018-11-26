/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

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
                                                             stokes_fe,
                                                             update_values),
            TemperatureAssembly::CopyData::RightHandSide<dim>(temperature_fe));
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_stokes_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble stokes system");

    std::cout << "   Assembling stokes system..." << std::endl;

    const QGauss<dim> quadrature_formula(parameters.velocity_degree + 1);

    if (rebuild_stokes_matrices)
    {
        // reset all entries
        stokes_matrix = 0;
        stokes_laplace_matrix = 0;

            // assemble matrix
            WorkStream::run(
                    stokes_dof_handler.begin_active(),
                    stokes_dof_handler.end(),
                    std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_matrix,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix,
                              this,
                              std::placeholders::_1),
                    StokesAssembly::Scratch::Matrix<dim>(
                            stokes_fe,
                            mapping,
                            quadrature_formula,
                            update_values|
                            update_gradients|
                            update_JxW_values),
                    StokesAssembly::CopyData::Matrix<dim>(stokes_fe));

            // copy velocity mass matrix
            velocity_mass_matrix.reinit(stokes_sparsity_pattern.block(0,0));
            velocity_mass_matrix.copy_from(stokes_matrix.block(0,0));

            // copy pressure mass matrix
            pressure_mass_matrix.reinit(stokes_sparsity_pattern.block(1,1));
            pressure_mass_matrix.copy_from(stokes_matrix.block(1,1));
            stokes_matrix.block(1,1) = 0;

            // time stepping coefficients
            const std::vector<double> alpha = (timestep_number != 0?
                                                imex_coefficients.alpha(timestep/old_timestep):
                                                std::vector<double>({1.0,-1.0,0.0}));
            const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));
            // correct (0,0)-block of stokes system
            stokes_matrix.block(0,0) *= alpha[0];
            stokes_matrix.block(0,0).add(timestep * equation_coefficients[1] * gamma[0],
                                         stokes_laplace_matrix.block(0,0));

            // adjust factors in the pressure matrices
            factor_Kp = alpha[0];
            factor_Mp = timestep * gamma[0] * equation_coefficients[1];

            // rebuilding pressure stiffness matrix preconditioner
            if (parameters.assemble_schur_complement == false)
            {
                Vector<double> tmp1(velocity_mass_matrix.m()), tmp2(tmp1);
                tmp1 = 1.0;
                tmp2 = 0.0;

                velocity_mass_matrix.precondition_Jacobi(tmp2, tmp1);
                stokes_matrix.block(1,0).mmult(stokes_laplace_matrix.block(1,1),
                                               stokes_matrix.block(0,1),
                                               tmp2,
                                               false);
            }

            preconditioner_Kp = std::shared_ptr<PreconditionerTypeKp>
            (new PreconditionerTypeKp());

            PreconditionerTypeKp::AdditionalData preconditioner_Kp_data;
            preconditioner_Kp->initialize(stokes_laplace_matrix.block(1,1),
                                          preconditioner_Kp_data);

            // rebuilding pressure mass matrix preconditioner
            preconditioner_Mp = std::shared_ptr<PreconditionerTypeMp>(new PreconditionerTypeMp());
            PreconditionerTypeMp::AdditionalData preconditioner_Mp_data;
            preconditioner_Mp_data.relaxation = 0.75;

            preconditioner_Mp->initialize(pressure_mass_matrix,
                                          preconditioner_Mp_data);

            // rebuild the preconditioner of the velocity block
            rebuild_stokes_preconditioner = true;

        // do not rebuild stokes matrices
        rebuild_stokes_matrices = false;
    }
    else if (timestep_number == 1 || timestep_modified)
    {
        Assert(timestep_number != 0, ExcInternalError());

        // time stepping coefficients
        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        // correct (0,0)-block of stokes system
        stokes_matrix.block(0,0).copy_from(velocity_mass_matrix);
        stokes_matrix.block(0,0) *= alpha[0];
        stokes_matrix.block(0,0).add(timestep * equation_coefficients[1] * gamma[0],
                                     stokes_laplace_matrix.block(0,0));

        // adjust factors in the pressure matrices
        factor_Kp = alpha[0];
        factor_Mp = timestep * gamma[0] * equation_coefficients[1];

        // rebuild the preconditioner of the velocity block
        rebuild_stokes_preconditioner = true;
    }
    // reset all entries
    stokes_rhs = 0;

    // assemble right-hand side function
    WorkStream::run(
            stokes_dof_handler.begin_active(),
            stokes_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs,
                      this,
                      std::placeholders::_1),
            StokesAssembly::Scratch::RightHandSide<dim>(
                    stokes_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_quadrature_points|
                    update_JxW_values|
                    update_gradients,
                    temperature_fe,
                    update_values),
            StokesAssembly::CopyData::RightHandSide<dim>(stokes_fe));
}

}  // namespace BuoyantFluid

// explicit instantiation

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_temperature_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_stokes_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_stokes_system();
