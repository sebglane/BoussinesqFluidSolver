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
#include "initial_values.h"

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
    build_pressure_mass_preconditioner();

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
void BuoyantFluidSolver<dim>::build_pressure_mass_preconditioner()
{
    if (!rebuild_pressure_mass_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner projection");

    preconditioner_pressure_mass.reset(new PreconditionerTypePressureMass());

    PreconditionerTypePressureMass::AdditionalData     data;

    preconditioner_pressure_mass->initialize(navier_stokes_mass_matrix.block(1,1),
                                             data);

    rebuild_pressure_mass_preconditioner = false;
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

    // solve linear system for pressure update
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

        navier_stokes_solution.block(1) *= alpha[0] / timestep;
    }
    // copy solution to phi_pressure
    phi_pressure.block(1) = navier_stokes_solution.block(1);

    if (parameters.projection_scheme == PressureUpdateType::StandardForm)
    {
        // update pressure
        navier_stokes_solution.block(1) = old_navier_stokes_solution.block(1);
        navier_stokes_solution.block(1) += phi_pressure.block(1);
    }
    else if (parameters.projection_scheme == PressureUpdateType::IrrotationalForm
                && timestep_number > 1)
    {
        // solve linear system for irrotational update
        SolverControl   solver_control(parameters.n_max_iter,
                                       std::max(parameters.rel_tol * navier_stokes_rhs.block(1).l2_norm(),
                                                parameters.abs_tol));

        SolverCG<>      cg(solver_control);

        navier_stokes_constraints.set_zero(navier_stokes_solution);

        try
        {
            cg.solve(navier_stokes_mass_matrix.block(1,1),
                     navier_stokes_solution.block(1),
                     navier_stokes_rhs.block(1),
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

        navier_stokes_constraints.distribute(navier_stokes_solution);

        if (parameters.verbose)
            std::cout << "      "
                      << solver_control.last_step()
                      << " CG iterations for pressure mass matrix solve"
                      << std::endl;

        navier_stokes_solution.block(1) *= -equation_coefficients[1];

        // update pressure
        navier_stokes_solution.block(1) += old_navier_stokes_solution.block(1);
        navier_stokes_solution.block(1) += phi_pressure.block(1);
    }

    const double mean_value = VectorTools::compute_mean_value(mapping,
                                                              navier_stokes_dof_handler,
                                                              QGauss<dim>(parameters.velocity_degree - 1),
                                                              navier_stokes_solution,
                                                              dim);
    navier_stokes_solution.block(1).add(-mean_value);
}

template<int dim>
void BuoyantFluidSolver<dim>::compute_initial_pressure()
{
    if (parameters.verbose)
        std::cout << "Computing initial pressure..." << std::endl;

    assemble_navier_stokes_matrices();
    build_projection_preconditioner();

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    const QGauss<dim>   quadrature(parameters.velocity_degree);

    FEValues<dim>   stokes_fe_values(mapping,
                                     navier_stokes_fe,
                                     quadrature,
                                     update_gradients|
                                     update_quadrature_points|
                                     update_JxW_values);

    FEValues<dim>   temperature_fe_values(mapping,
                                          temperature_fe,
                                          quadrature,
                                          update_values);

    const unsigned int dofs_per_cell = navier_stokes_fe.dofs_per_cell;
    const unsigned int n_q_points    = stokes_fe_values.n_quadrature_points;

    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);
    Vector<double>                          local_rhs(dofs_per_cell);

    std::vector<Tensor<1,dim>>  phi_pressure_gradients(dofs_per_cell);

    std::vector<double>  temperature_values(n_q_points);

    for (auto cell: navier_stokes_dof_handler.active_cell_iterators())
    {
        local_rhs = 0;

        cell->get_dof_indices(local_dof_indices);

        stokes_fe_values.reinit(cell);

        typename DoFHandler<dim>::active_cell_iterator
        temperature_cell(&triangulation,
                         cell->level(),
                         cell->index(),
                         &temperature_dof_handler);

        temperature_fe_values.reinit(temperature_cell);

        temperature_fe_values.get_function_values(temperature_solution,
                                                  temperature_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
                phi_pressure_gradients[k] = stokes_fe_values[pressure].gradient(k, q);

            const Tensor<1,dim> gravity_vector = EquationData::GravityVector<dim>().value(stokes_fe_values.quadrature_point(q));

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                local_rhs(i)
                    -= equation_coefficients[2] * temperature_values[q] * gravity_vector
                       * phi_pressure_gradients[i] * stokes_fe_values.JxW(q);
        }

        navier_stokes_constraints.distribute_local_to_global(local_rhs,
                                                             local_dof_indices,
                                                             navier_stokes_rhs);
    }

    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for pressure update
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * navier_stokes_rhs.block(1).l2_norm(),
                                   parameters.abs_tol));

    SolverCG<>      cg(solver_control);

    navier_stokes_constraints.set_zero(old_navier_stokes_solution);
    try
    {
        cg.solve(navier_stokes_laplace_matrix.block(1,1),
                 old_navier_stokes_solution.block(1),
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
    navier_stokes_constraints.distribute(old_navier_stokes_solution);

    // write info message
    if (parameters.verbose)
        std::cout << "   "
                << solver_control.last_step()
                << " CG iterations for projection step"
                << std::endl;

    const double mean_value = VectorTools::compute_mean_value(mapping,
                                                              navier_stokes_dof_handler,
                                                              QGauss<dim>(parameters.velocity_degree - 1),
                                                              old_navier_stokes_solution,
                                                              dim);
    old_navier_stokes_solution.block(1).add(-mean_value);
    navier_stokes_solution.block(1) = old_navier_stokes_solution.block(1);
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

template void BuoyantFluid::BuoyantFluidSolver<2>::compute_initial_pressure();
template void BuoyantFluid::BuoyantFluidSolver<3>::compute_initial_pressure();
