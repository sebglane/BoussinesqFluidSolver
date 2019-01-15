/*
 * solve_navier_stokes.cc
 *
 *  Created on: Jan 11, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"

namespace BuoyantFluid {


template<int dim>
void BuoyantFluidSolver<dim>::navier_stokes_step()
{
    std::cout << "   Navier-Stokes step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_velocity_system();

    // rebuild preconditioner for diffusion step
    build_diffusion_preconditioner();

    // solve projection step
    solve_diffusion_system();

    // assemble right-hand side (and system if necessary)
    assemble_pressure_system();

    // rebuild preconditioner for projection step
    build_projection_preconditioner();

    // solve projection system
    solve_projection_system();
}

template<int dim>
void BuoyantFluidSolver<dim>::build_diffusion_preconditioner()
{
    if (!rebuild_diffusion_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build diffusion preconditioner");

    preconditioner_diffusion.reset(new PreconditionerTypeDiffusion());

    PreconditionerTypeDiffusion::AdditionalData     data;
    preconditioner_diffusion->initialize(velocity_matrix,
                                         data);

    rebuild_diffusion_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_projection_preconditioner()
{
    if (!rebuild_projection_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build projection preconditioner");

    preconditioner_projection.reset(new PreconditionerTypeProjection());

    PreconditionerTypeProjection::AdditionalData     data;
    data.extra_off_diagonals = 60;
    data.strengthen_diagonal = 0.01;

    preconditioner_projection->initialize(pressure_laplace_matrix,
                                          data);

    rebuild_projection_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_pressure_mass_preconditioner()
{
    if (!rebuild_pressure_mass_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build projection pressure mass");

    preconditioner_pressure_mass.reset(new PreconditionerTypePressureMass());

    PreconditionerTypePressureMass::AdditionalData     data;

    preconditioner_projection->initialize(pressure_mass_matrix,
                                          data);

    rebuild_pressure_mass_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::solve_diffusion_system()
{
    std::cout << "      Solving diffusion system..." << std::endl;
    TimerOutput::Scope  timer_section(computing_timer, "diffusion solve");

    // solve linear system
    SolverControl   solver_control(30, 1e-6 * velocity_rhs.l2_norm());

    SolverCG<Vector<double>>  cg(solver_control);

    velocity_constraints.set_zero(velocity_solution);

    cg.solve(velocity_matrix,
             velocity_solution,
             velocity_rhs,
             *preconditioner_diffusion);

    velocity_constraints.distribute(velocity_solution);

    // write info message
    std::cout << "      "
            << solver_control.last_step()
            << " CG iterations for diffusion step"
            << std::endl;
}


template<int dim>
void BuoyantFluidSolver<dim>::solve_projection_system()
{
    std::cout << "      Solving projection system..." << std::endl;
    TimerOutput::Scope  timer_section(computing_timer, "projection solve");

    // solve linear system for phi_pressure
    SolverControl   solver_control(300, 1e-6 * pressure_rhs.l2_norm());

    SolverCG<>      cg(solver_control);

    pressure_constraints.set_zero(phi_solution);

    cg.solve(pressure_laplace_matrix,
             phi_solution,
             pressure_rhs,
             *preconditioner_projection);

    pressure_constraints.distribute(phi_solution);
    // write info message
    std::cout << "      "
            << solver_control.last_step()
            << " CG iterations for projection step"
            << std::endl;

    {
        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        phi_solution *= -alpha[0] / timestep;
    }
    // update pressure
    pressure_solution = old_pressure_solution;
    pressure_solution += phi_solution;

    if (parameters.projection_scheme == PressureUpdateType::IrrotationalForm)
    {
        Vector<double>  phi_irrotational(pressure_solution.size());
        {
            // solve linear system for phi_irrotational
            SolverControl   solver_control(30, 1e-9 * pressure_rhs.l2_norm());

            SolverCG<>      cg(solver_control);

            pressure_constraints.set_zero(phi_irrotational);

            cg.solve(pressure_mass_matrix,
                     phi_irrotational,
                     pressure_rhs,
                     *preconditioner_pressure_mass);

            pressure_constraints.distribute(phi_irrotational);
        }

        pressure_solution += equation_coefficients[1] * phi_irrotational;
    }

    const double mean_value = VectorTools::compute_mean_value(mapping,
                                                              pressure_dof_handler,
                                                              QGauss<dim>(parameters.velocity_degree),
                                                              pressure_solution,
                                                              0);
    pressure_solution.add(-mean_value);
}
}  // namespace BuoyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::navier_stokes_step();
template void BuoyantFluid::BuoyantFluidSolver<3>::navier_stokes_step();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_diffusion_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_diffusion_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_projection_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_projection_system();
