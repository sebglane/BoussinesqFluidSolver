/*
 * solve_temperature.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <adsolic/convection_diffusion_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>


namespace adsolic {

template<int dim>
void ConvectionDiffusionSolver<dim>::convection_diffusion_step()
{
    if (parameters.verbose)
        pcout << "   Temperature step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_system();

    // rebuild preconditioner for diffusion step
    build_preconditioner();

    // solve projection step
    solve_linear_system();
}

template<int dim>
void ConvectionDiffusionSolver<dim>::build_preconditioner()
{
    if (!rebuild_preconditioner)
        return;

    computing_timer->enter_subsection("build preconditioner temperature");

    preconditioner.reset(new LA::PreconditionSSOR());

    LA::PreconditionSSOR::AdditionalData     data;
    data.omega = 0.6;

    preconditioner->initialize(system_matrix,
                                           data);

    rebuild_preconditioner = false;

    computing_timer->leave_subsection();
}


template <int dim>
void ConvectionDiffusionSolver<dim>::solve_linear_system()
{
    if (parameters.verbose)
        pcout << "      Solving temperature system..." << std::endl;

    computing_timer->enter_subsection("solve temperature");

    LA::Vector  distributed_solution(rhs);
    distributed_solution = solution;

    SolverControl solver_control(parameters.n_max_iter,
                                 std::max(parameters.rel_tol * rhs.l2_norm(),
                                          parameters.abs_tol));

    // solve linear system
    try
    {
        LA::SolverCG    cg(solver_control);

        cg.solve(system_matrix,
                 distributed_solution,
                 rhs,
                 *preconditioner);
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
    constraints.distribute(distributed_solution);
    solution = distributed_solution;

    computing_timer->leave_subsection();

    // write info message
    if (parameters.verbose)
        pcout << "      "
              << solver_control.last_step()
              << " CG iterations for temperature"
              << std::endl;
}

// explicit instantiation
template void ConvectionDiffusionSolver<2>::convection_diffusion_step();
template void ConvectionDiffusionSolver<3>::convection_diffusion_step();

template void ConvectionDiffusionSolver<2>::build_preconditioner();
template void ConvectionDiffusionSolver<3>::build_preconditioner();

template void ConvectionDiffusionSolver<2>::solve_linear_system();
template void ConvectionDiffusionSolver<3>::solve_linear_system();

}  // namespace BouyantFluid
