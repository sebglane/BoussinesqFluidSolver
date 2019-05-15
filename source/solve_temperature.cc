/*
 * solve_temperature.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include "buoyant_fluid_solver.h"

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::temperature_step()
{
    if (parameters.verbose)
        pcout << "   Temperature step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_temperature_system();

    // rebuild preconditioner for diffusion step
    build_temperature_preconditioner();

    // solve projection step
    solve_temperature_system();
}

template<int dim>
void BuoyantFluidSolver<dim>::build_temperature_preconditioner()
{
    if (!rebuild_temperature_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner temperature");

    preconditioner_temperature.reset(new LA::PreconditionSSOR());

    LA::PreconditionSSOR::AdditionalData     data;
    data.omega = 0.6;

    preconditioner_temperature->initialize(temperature_matrix,
                                 data);

    rebuild_temperature_preconditioner = false;
}


template <int dim>
void BuoyantFluidSolver<dim>::solve_temperature_system()
{
    if (parameters.verbose)
        pcout << "      Solving temperature system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve temperature");

    LA::Vector  distributed_solution(temperature_rhs);
    distributed_solution = temperature_solution;

    SolverControl solver_control(parameters.n_max_iter,
                                 std::max(parameters.rel_tol * temperature_rhs.l2_norm(),
                                          parameters.abs_tol));

    // solve linear system
    try
    {
        LA::SolverCG    cg(solver_control);

        cg.solve(temperature_matrix,
                 distributed_solution,
                 temperature_rhs,
                 *preconditioner_temperature);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception in temperature solve: " << std::endl
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
        std::cerr << "Unknown exception in temperature solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    temperature_constraints.distribute(distributed_solution);
    temperature_solution = distributed_solution;

    // write info message
    if (parameters.verbose)
        pcout << "      "
              << solver_control.last_step()
              << " CG iterations for temperature"
              << std::endl;
}
}  // namespace BouyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::temperature_step();
template void BuoyantFluid::BuoyantFluidSolver<3>::temperature_step();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_temperature_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_temperature_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_temperature_system();
