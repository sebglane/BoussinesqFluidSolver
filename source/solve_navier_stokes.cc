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
    assemble_velocity_system();

    // rebuild preconditioner for diffusion step
    build_diffusion_preconditioner();

    // solve projection step
    solve_diffusion_system();

    // assemble right-hand side (and system if necessary)
    assemble_pressure_system();

    // rebuild preconditioner for projection step
    build_projection_preconditioner();
    build_pressure_mass_preconditioner();

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
    preconditioner_diffusion->initialize(velocity_matrix,
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

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner projection");

    preconditioner_pressure_mass.reset(new PreconditionerTypePressureMass());

    PreconditionerTypePressureMass::AdditionalData     data;

    preconditioner_pressure_mass->initialize(pressure_mass_matrix,
                                             data);

    rebuild_pressure_mass_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::solve_diffusion_system()
{
    if(parameters.verbose)
        std::cout << "      Solving diffusion system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve diffusion");

    // solve linear system
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * velocity_rhs.l2_norm(),
                                            parameters.abs_tol));;

    SolverCG<Vector<double>>  cg(solver_control);

    velocity_constraints.set_zero(velocity_solution);

    try
    {
        cg.solve(velocity_matrix,
                 velocity_solution,
                 velocity_rhs,
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
    velocity_constraints.distribute(velocity_solution);

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
    if(parameters.verbose)
        std::cout << "      Solving projection system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for phi_pressure
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * pressure_rhs.l2_norm(),
                                            parameters.abs_tol));

    SolverCG<>      cg(solver_control);

    pressure_constraints.set_zero(phi_solution);

    try
    {
        cg.solve(pressure_laplace_matrix,
                 phi_solution,
                 pressure_rhs,
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

    pressure_constraints.distribute(phi_solution);
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
            SolverControl   solver_control(parameters.n_max_iter,
                                           std::max(parameters.rel_tol * pressure_rhs.l2_norm(),
                                                    parameters.abs_tol));

            SolverCG<>      cg(solver_control);

            pressure_constraints.set_zero(phi_irrotational);

            try
            {
                cg.solve(pressure_mass_matrix,
                         phi_irrotational,
                         pressure_rhs,
                         *preconditioner_pressure_mass);
            }
            catch (std::exception &exc)
            {
                std::cerr << std::endl << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
                std::cerr << "Exception in pressure mass matrix solve: " << std::endl
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
                std::cerr << "Unknown exception pressure mass matrix solve!" << std::endl
                        << "Aborting!" << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
                std::abort();
            }

            pressure_constraints.distribute(phi_irrotational);

            if (parameters.verbose)
                std::cout << "      "
                        << solver_control.last_step()
                        << " CG iterations for pressure mass solve"
                        << std::endl;
        }
        phi_irrotational *= -1.0;

        pressure_solution.add(equation_coefficients[1], phi_irrotational);
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
