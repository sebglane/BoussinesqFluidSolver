/*
 * solve_magnetic.cc
 *
 *  Created on: Apr 26, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>

#include "conducting_fluid_solver.h"

namespace ConductingFluid {

template<int dim>
void ConductingFluidSolver<dim>::magnetic_step()
{
    std::cout << "   Magnetic step..." << std::endl;

    if (timestep_number == 0)
    {
        assemble_magnetic_matrices();
        // rebuild the preconditioner of diffusion solve
        rebuild_magnetic_diffusion_preconditioner = true;
    }
    else
    {
        Assert(timestep_number != 0, ExcInternalError());
        assemble_magnetic_matrices();
    }

    assemble_magnetic_rhs();

    SparseDirectUMFPACK     direct_solver;
    direct_solver.initialize(magnetic_matrix);

    BlockVector<double> tmp_magnetic_vector(magnetic_rhs);

    direct_solver.solve(tmp_magnetic_vector);

    magnetic_solution = tmp_magnetic_vector;

    magnetic_constraints.distribute(magnetic_solution);

    /*
     *

    // assemble right-hand side (and system if necessary)
    assemble_diffusion_system();

    // solve projection step
    solve_diffusion_system();

    // assemble right-hand side (and system if necessary)
    assemble_projection_system();

    // solve projection system
    solve_projection_system();

     *
     */
}

/*
 *

template<int dim>
void ConductingFluidSolver<dim>::solve_diffusion_system()
{
    std::cout << "      Solving diffusion system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve diffusion");

    // solve linear system
    SolverControl   solver_control(100,
                                   std::max(1e-6 * magnetic_rhs.block(0).l2_norm(),
                                            1e-9));

    SparseILU<double>   preconditioner;
    preconditioner.initialize(magnetic_matrix.block(0,0));

    try
    {
        SolverCG<Vector<double>>    cg(solver_control);

        cg.solve(magnetic_matrix.block(0,0),
                 magnetic_solution.block(0),
                 magnetic_rhs.block(0),
                 preconditioner);
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
    magnetic_constraints.distribute(magnetic_solution);

    // write info message
    std::cout << "      "
              << solver_control.last_step()
              << " CG iterations for diffusion step"
              << std::endl;
}

template<int dim>
void ConductingFluidSolver<dim>::solve_projection_system()
{
    std::cout << "      Solving projection system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for pressure update
    SolverControl   solver_control(100,
                                   std::max(1e-6 * magnetic_rhs.block(1).l2_norm(),
                                   1e-9));

    SparseILU<double>   preconditioner;
    preconditioner.initialize(magnetic_curl_matrix.block(1,1));



    SolverCG<Vector<double>>    cg(solver_control);

    magnetic_constraints.set_zero(phi_pseudo_pressure);

    try
    {
        cg.solve(magnetic_curl_matrix.block(1,1),
                 phi_pseudo_pressure.block(1),
                 magnetic_rhs.block(1),
                 preconditioner);
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
    magnetic_constraints.distribute(phi_pseudo_pressure);

    // write info message
    std::cout << "      "
              << solver_control.last_step()
              << " CG iterations for projection step"
              << std::endl;


    {
        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        phi_pseudo_pressure.block(1) *= alpha[0] / timestep;
    }

    // update pseudo pressure
    magnetic_solution.block(1) = old_magnetic_solution.block(1);
    magnetic_solution.block(1) += phi_pseudo_pressure.block(1);
}

 *
 */

}  // namespace ConductingFluid

// explicit instantiation
template void ConductingFluid::ConductingFluidSolver<3>::magnetic_step();

/*
 *
template void ConductingFluid::ConductingFluidSolver<3>::solve_diffusion_system();

template void ConductingFluid::ConductingFluidSolver<3>::solve_projection_system();
 *
 */
