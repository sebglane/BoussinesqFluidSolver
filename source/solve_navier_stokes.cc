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
    if (parameters.verbose)
        std::cout << "   Navier-Stokes step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_diffusion_system();

    // rebuild preconditioner for diffusion step
    build_diffusion_preconditioner();

    // solve projection step
    solve_diffusion_system();

    // rebuild preconditioner for projection step
    build_projection_preconditioner();

    // assemble right-hand side (and system if necessary)
    assemble_projection_system();

    // solve projection system
    solve_projection_system();
}

template<int dim>
void BuoyantFluidSolver<dim>::build_diffusion_preconditioner()
{
    if (!rebuild_diffusion_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner diffusion");

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

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner projection");

    preconditioner_projection.reset(new PreconditionerTypeProjection());

    PreconditionerTypeProjection::AdditionalData     data;
    data.strengthen_diagonal = 0.1;
    data.extra_off_diagonals = 60;

    preconditioner_projection->initialize(navier_stokes_laplace_matrix.block(1,1),
                                          data);

    rebuild_projection_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::solve_diffusion_system()
{
    if (parameters.verbose)
        std::cout << "      Solving diffusion system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve diffusion");

    // solve linear system
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * navier_stokes_solution.block(0).l2_norm(),
                                            parameters.abs_tol));;

    SolverCG<Vector<double>>  cg(solver_control);


    navier_stokes_constraints.set_zero(navier_stokes_solution);

    try
    {
        cg.solve(navier_stokes_matrix.block(0,0),
                 navier_stokes_solution.block(0),
                 navier_stokes_rhs.block(0),
                 *preconditioner_diffusion);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception in diffusion solve: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Unknown exception diffusion solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    navier_stokes_constraints.distribute(navier_stokes_solution);

    // write info message
    if (parameters.verbose)
        std::cout << "      "
                << solver_control.last_step()
                << " CG iterations for diffusion step"
                << std::endl;
}


template<int dim>
void BuoyantFluidSolver<dim>::solve_projection_system()
{
    if (parameters.verbose)
        std::cout << "      Solving projection system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for phi_pressure
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * navier_stokes_rhs.block(1).l2_norm(),
                                   parameters.abs_tol));

    SolverCG<>      cg(solver_control);

    navier_stokes_constraints.set_zero(navier_stokes_solution);

    try
    {
        cg.solve(navier_stokes_laplace_matrix.block(1,1),
                 navier_stokes_solution.block(1),
                 navier_stokes_rhs.block(1),
                 *preconditioner_projection);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception in projection solve: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Unknown exception projection solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    navier_stokes_constraints.distribute(navier_stokes_solution);

    // write info message
    if (parameters.verbose)
        std::cout << "      "
                << solver_control.last_step()
                << " CG iterations for projection step"
                << std::endl;

    {
        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        navier_stokes_solution.block(1) *= -alpha[0] / timestep;
    }
    // copy solution to phi_pressure
    phi_pressure.block(1) = navier_stokes_solution.block(1);

    // update pressure
    navier_stokes_solution.block(1) = old_navier_stokes_solution.block(1);
    navier_stokes_solution.block(1) += phi_pressure.block(1);

    const double mean_value = VectorTools::compute_mean_value(mapping,
                                                              navier_stokes_dof_handler,
                                                              QGauss<dim>(parameters.velocity_degree - 1),
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

template void BuoyantFluid::BuoyantFluidSolver<2>::build_diffusion_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_diffusion_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_projection_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_projection_preconditioner();

