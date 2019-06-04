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

#include <magnetic_diffusion_solver.h>

namespace ConductingFluid {

template<int dim>
void MagneticDiffusionSolver<dim>::magnetic_step()
{
    std::cout << "   Magnetic step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_diffusion_system();

    // solve projection step
    solve_diffusion_system();

    // assemble right-hand side (and system if necessary)
    assemble_projection_system();

    // solve projection system
    solve_projection_system();
}

template<int dim>
void MagneticDiffusionSolver<dim>::solve_diffusion_system()
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
void MagneticDiffusionSolver<dim>::solve_projection_system()
{
    std::cout << "      Solving projection system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for new pressure
    try
    {
        SolverControl   solver_control(100,
                                       std::max(1e-6 * magnetic_rhs.block(1).l2_norm(),
                                       1e-9));

        SparseILU<double>   preconditioner;
        preconditioner.initialize(magnetic_curl_matrix.block(1,1));


        SolverCG<Vector<double>>    cg(solver_control);

        magnetic_constraints.set_zero(magnetic_solution);

        cg.solve(magnetic_matrix.block(1,1),
                 magnetic_solution.block(1),
                 magnetic_rhs.block(1),
                 preconditioner);

        magnetic_constraints.distribute(magnetic_solution);

        // write info message
        std::cout << "      "
                  << solver_control.last_step()
                  << " CG iterations for pressure projection step"
                  << std::endl;
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
    // solve linear system for new magnetic field
    try
    {
        // initialize right-hand side with pressure gradient
        magnetic_matrix.block(0,1).vmult(magnetic_rhs.block(0),
                                         magnetic_solution.block(1));
        // add stabilization term
        magnetic_stabilization_matrix.block(0,1).vmult_add(magnetic_rhs.block(0),
                                                           magnetic_solution.block(1));
        // correct sign
        magnetic_rhs.block(0) *= -1.0;

        // add old pressure gradient
        magnetic_matrix.block(0,1).vmult_add(magnetic_rhs.block(0),
                                             old_magnetic_solution.block(1));

        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        // scale right-hand side with time stepping coefficient
        magnetic_rhs.block(0) *= timestep / alpha[0];

        SolverControl   solver_control(100,
                                       std::max(1e-6 * magnetic_rhs.block(0).l2_norm(),
                                       1e-9));

        PreconditionJacobi<SparseMatrix<double>>   preconditioner;
        preconditioner.initialize(magnetic_mass_matrix.block(0,0),
                                  PreconditionJacobi<SparseMatrix<double>>::AdditionalData(.6));

        SolverCG<Vector<double>>    cg(solver_control);

        Vector<double>  magnetic_update(magnetic_solution.block(0).size());

        cg.solve(magnetic_mass_matrix.block(0,0),
                 magnetic_update,
                 magnetic_rhs.block(0),
                 preconditioner);

        magnetic_solution.block(0).add(1., magnetic_update);

        // write info message
        std::cout << "      "
                  << solver_control.last_step()
                  << " CG iterations for magnetic field projection step"
                  << std::endl;
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
}

}  // namespace ConductingFluid

// explicit instantiation
template void ConductingFluid::MagneticDiffusionSolver<3>::magnetic_step();
template void ConductingFluid::MagneticDiffusionSolver<3>::solve_diffusion_system();
template void ConductingFluid::MagneticDiffusionSolver<3>::solve_projection_system();
