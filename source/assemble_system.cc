/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/matrix_tools.h>

#include "buoyant_fluid_solver.h"


namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::assemble_temperature_system()
{
    if (parameters.verbose)
        std::cout << "   Assembling temperature system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble temperature system");

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 1);

    // assemble temperature matrices
    if (rebuild_temperature_matrices)
    {
        temperature_mass_matrix = 0;
        temperature_stiffness_matrix = 0;

        MatrixCreator::create_mass_matrix(mapping,
                                          temperature_dof_handler,
                                          quadrature_formula,
                                          temperature_mass_matrix,
                                          (const Function<dim> *const)nullptr,
                                          temperature_constraints);
        MatrixCreator::create_laplace_matrix(mapping,
                                             temperature_dof_handler,
                                             quadrature_formula,
                                             temperature_stiffness_matrix,
                                             (const Function<dim> *const)nullptr,
                                             temperature_constraints);

        const std::vector<double> alpha = (timestep_number != 0?
                                                imex_coefficients.alpha(timestep/old_timestep):
                                                std::vector<double>({1.0,-1.0,0.0}));
        const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0] / timestep ;
        temperature_matrix.add(gamma[0] * equation_coefficients[3],
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
        temperature_matrix *= (alpha[0] / timestep);
        temperature_matrix.add(gamma[0] * equation_coefficients[3],
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
                                                             velocity_fe,
                                                             update_values),
            TemperatureAssembly::CopyData::RightHandSide<dim>(temperature_fe));
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_velocity_system()
{
    if (parameters.verbose)
        std::cout << "      Assembling velocity system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble velocity system");

    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);

    if (rebuild_velocity_matrices)
    {
        // reset all entries
        velocity_matrix = 0;
        velocity_laplace_matrix = 0;

        MatrixCreator::create_laplace_matrix(mapping,
                                             velocity_dof_handler,
                                             quadrature_formula,
                                             velocity_laplace_matrix,
                                             (const Function<dim> *const)nullptr,
                                             velocity_constraints);
        MatrixCreator::create_mass_matrix(mapping,
                                          velocity_dof_handler,
                                          quadrature_formula,
                                          velocity_mass_matrix,
                                          (const Function<dim> *const)nullptr,
                                          velocity_constraints);

        // time stepping coefficients
        const std::vector<double> alpha = (timestep_number != 0?
                                            imex_coefficients.alpha(timestep/old_timestep):
                                            std::vector<double>({1.0,-1.0,0.0}));
        const std::vector<double> gamma = (timestep_number != 0?
                                            imex_coefficients.gamma(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0,0.0}));
        // correct (0,0)-block of stokes system
        velocity_matrix.copy_from(velocity_mass_matrix);
        velocity_matrix *= alpha[0] / timestep;
        velocity_matrix.add(equation_coefficients[1] * gamma[0],
                            velocity_laplace_matrix);

        // rebuild the preconditioner of both preconditioners
        rebuild_diffusion_preconditioner = true;

        // do not rebuild stokes matrices again
        rebuild_velocity_matrices = false;
    }
    else if (timestep_number == 1 || timestep_modified)
    {
        Assert(timestep_number != 0, ExcInternalError());

        // time stepping coefficients
        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        // correct (0,0)-block of navier stokes system
        velocity_matrix.copy_from(velocity_mass_matrix);
        velocity_matrix *= alpha[0] / timestep;
        velocity_matrix.add(equation_coefficients[1] * gamma[0],
                            velocity_laplace_matrix);

        // rebuild the preconditioner of the velocity block
        rebuild_diffusion_preconditioner = true;
    }
    // reset all entries
    velocity_rhs = 0;

    // assemble right-hand side function
    WorkStream::run(
            velocity_dof_handler.begin_active(),
            velocity_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_velocity_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_velocity_rhs,
                      this,
                      std::placeholders::_1),
            NavierStokesAssembly::Scratch::RightHandSide<dim>(
                    velocity_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_quadrature_points|
                    update_JxW_values|
                    update_gradients,
                    pressure_fe,
                    update_values,
                    temperature_fe,
                    update_values),
            NavierStokesAssembly::CopyData::RightHandSide<dim>(velocity_fe));
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_pressure_system()
{
    if (parameters.verbose)
        std::cout << "      Assembling pressure system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble pressure system");

    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);

    if (rebuild_pressure_matrices)
    {
        // reset all entries
        pressure_laplace_matrix = 0;
        pressure_mass_matrix = 0;
        MatrixCreator::create_laplace_matrix(mapping,
                                             pressure_dof_handler,
                                             quadrature_formula,
                                             pressure_laplace_matrix,
                                             (const Function<dim> *const)nullptr,
                                             pressure_constraints);
        MatrixCreator::create_mass_matrix(mapping,
                                          pressure_dof_handler,
                                          quadrature_formula,
                                          pressure_mass_matrix,
                                          (const Function<dim> *const)nullptr,
                                          pressure_constraints);

        // rebuild the preconditioner of both preconditioners
        rebuild_projection_preconditioner = true;
        rebuild_pressure_mass_preconditioner = true;

        // do not rebuild pressure matrix again
        rebuild_pressure_matrices = false;
    }
    // reset all entries
    pressure_rhs = 0;

    // assemble right-hand side function
    WorkStream::run(
            pressure_dof_handler.begin_active(),
            pressure_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_pressure_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_pressure_rhs,
                      this,
                      std::placeholders::_1),
            PressureAssembly::Scratch::RightHandSide<dim>(
                    pressure_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_JxW_values,
                    velocity_fe,
                    update_gradients),
            PressureAssembly::CopyData::RightHandSide<dim>(pressure_fe));
}

}  // namespace BuoyantFluid

// explicit instantiation

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_temperature_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_velocity_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_velocity_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_pressure_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_pressure_system();
