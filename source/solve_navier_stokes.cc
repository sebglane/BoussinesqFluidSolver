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
    assemble_navier_stokes_system();

    // rebuild preconditioner for diffusion step
    build_diffusion_preconditioner();

    // solve projection step
    solve_diffusion_system();

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
    preconditioner_diffusion->initialize(navier_stokes_matrix.block(0,0),
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
    data.relaxation = 0.6;

    preconditioner_projection->initialize(navier_stokes_matrix.block(1,1),
                                          data);

    rebuild_projection_preconditioner = false;
}


template<int dim>
void BuoyantFluidSolver<dim>::solve_diffusion_system()
{
    std::cout << "      Solving diffusion system..." << std::endl;
    TimerOutput::Scope  timer_section(computing_timer, "diffusion solve");

    // substract pressure gradient from right-hand side
    Vector<double>  system_rhs(navier_stokes_rhs.block(0));
    {
        Vector<double>  extrapolated_pressure(old_navier_stokes_solution.block(1));

        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        extrapolated_pressure.add(-alpha[1] / alpha[0], old_phi_pressure,
                                  -alpha[2] / alpha[0], old_old_phi_pressure);

        extrapolated_pressure *= -1.0;

        navier_stokes_matrix.block(0,1).vmult_add(system_rhs,
                                                  extrapolated_pressure);
    }

    // solve linear system
    SolverControl   solver_control(300, 1e-6 * system_rhs.l2_norm());

    SolverCG<Vector<double>>  cg(solver_control);

    navier_stokes_constraints.set_zero(navier_stokes_solution);

    cg.solve(navier_stokes_matrix.block(0,0),
             navier_stokes_solution.block(0),
             system_rhs,
             *preconditioner_diffusion);

    navier_stokes_constraints.distribute(navier_stokes_solution);

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

    // construct right-hand from velocity solution
    Vector<double>  system_rhs(navier_stokes_rhs.block(1));
    navier_stokes_matrix.block(1,0).vmult_add(system_rhs,
                                              navier_stokes_solution.block(0));

    // solve linear system for phi_pressure
    SolverControl   solver_control(60, 1e-6 * system_rhs.l2_norm());

    SolverCG<>      cg(solver_control);

    navier_stokes_constraints.set_zero(navier_stokes_solution);

    cg.solve(navier_stokes_matrix.block(1,1),
             navier_stokes_solution.block(1),
             system_rhs,
             *preconditioner_projection);

    navier_stokes_constraints.distribute(navier_stokes_solution);
    // write info message
    std::cout << "      "
            << solver_control.last_step()
            << " CG iterations for projection step"
            << std::endl;

    {
        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        navier_stokes_solution.block(1) *= alpha[0] / timestep;
    }
    // copy solution to phi_pressure
    phi_pressure = navier_stokes_solution.block(1);

    // update pressure
    navier_stokes_solution.block(1) += old_navier_stokes_solution.block(1);

    if (parameters.projection_scheme == PressureUpdateType::IrrotationalForm)
    {
        Vector<double>  velocity_divergence(navier_stokes_rhs.block(1));

        navier_stokes_matrix.block(1,0).vmult_add(velocity_divergence,
                                                  navier_stokes_solution.block(0));

        navier_stokes_solution.block(1) -= velocity_divergence;
    }

    const double mean_value = VectorTools::compute_mean_value(mapping,
                                                              navier_stokes_dof_handler,
                                                              QGauss<dim>(parameters.velocity_degree),
                                                              navier_stokes_solution,
                                                              dim);
    navier_stokes_solution.block(1).add(-mean_value);
}
}  // namespace BuoyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::navier_stokes_step();
template void BuoyantFluid::BuoyantFluidSolver<3>::navier_stokes_step();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_diffusion_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_diffusion_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_projection_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_projection_system();
