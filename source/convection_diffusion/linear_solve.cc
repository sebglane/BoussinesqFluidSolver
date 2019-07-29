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
void ConvectionDiffusionSolver<dim>::build_preconditioner()
{
    if (!rebuild_preconditioner)
        return;

    TimerOutput::Scope(*(this->computing_timer),"Convect.-Diff. Build preconditioner.");

    preconditioner.reset(new LA::PreconditionSSOR());

    LA::PreconditionSSOR::AdditionalData     data;
    data.omega = 0.6;

    preconditioner->initialize(system_matrix,
                               data);

    rebuild_preconditioner = false;
}


template <int dim>
void ConvectionDiffusionSolver<dim>::solve_linear_system()
{
    if (parameters.verbose)
        this->pcout << "      Solving temperature system..." << std::endl;

    Assert(rebuild_preconditioner == false,
           ExcMessage("Cannot solve_linear_system if flag to build precondition"
                      " is true"));

    TimerOutput::Scope(*(this->computing_timer),"Convect.-Diff. Linear solve.");

    LA::Vector  distributed_solution(this->rhs);
    distributed_solution = this->solution;

    SolverControl solver_control(parameters.n_max_iter,
                                 std::max(parameters.rel_tol * this->rhs.l2_norm(),
                                          parameters.abs_tol));

    // solve linear system
    try
    {
        LA::SolverCG    cg(solver_control);

        cg.solve(system_matrix,
                 distributed_solution,
                 this->rhs,
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
    this->solution = distributed_solution;

    // write info message
    if (parameters.verbose)
        this->pcout << "      "
                    << solver_control.last_step()
                    << " CG iterations for temperature"
                    << std::endl;
}

// explicit instantiation
template void ConvectionDiffusionSolver<2>::build_preconditioner();
template void ConvectionDiffusionSolver<3>::build_preconditioner();

template void ConvectionDiffusionSolver<2>::solve_linear_system();
template void ConvectionDiffusionSolver<3>::solve_linear_system();

}  // namespace BouyantFluid

