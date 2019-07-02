/*
 * solve_navier_stokes.cc
 *
 *  Created on: Jan 11, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"

namespace BuoyantFluid {


template<int dim>
void BuoyantFluidSolver<dim>::navier_stokes_step()
{
    if (parameters.verbose)
        pcout << "   Navier-Stokes step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_diffusion_system();

    // rebuild preconditioner for diffusion step
    build_diffusion_preconditioner();

    // solve projection step
    solve_diffusion_system();

    // assemble right-hand side (and system if necessary)
    assemble_projection_system();

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

    if (parameters.verbose)
        pcout << "      Building diffusion preconditioner..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner diffusion");

    if (parameters.convective_scheme == ConvectiveDiscretizationType::LinearImplicit)
    {
        preconditioner_asymmetric_diffusion.reset(new LA::PreconditionAMG());

        LA::PreconditionAMG::AdditionalData     data;
        data.higher_order_elements = true;
        data.elliptic = true;

        preconditioner_asymmetric_diffusion
        ->initialize(navier_stokes_matrix.block(0,0),
                     data);
    }
    else
    {
        preconditioner_symmetric_diffusion.reset(new LA::PreconditionAMG());

        LA::PreconditionAMG::AdditionalData    data;
        data.higher_order_elements = true;
        data.elliptic = true;
        data.aggregation_threshold = 0.02;
        data.smoother_sweeps = 2;

        preconditioner_symmetric_diffusion
        ->initialize(navier_stokes_matrix.block(0,0),
                     data);
    }
    rebuild_diffusion_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_projection_preconditioner()
{
    if (!rebuild_projection_preconditioner)
        return;

    if (parameters.verbose)
        pcout << "      Building projection preconditioner..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner projection");

    preconditioner_projection.reset(new LA::PreconditionAMG());

    LA::PreconditionAMG::AdditionalData     data;

    preconditioner_projection->initialize(navier_stokes_laplace_matrix.block(1,1),
                                          data);

    rebuild_projection_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_pressure_mass_preconditioner()
{
    if (!rebuild_pressure_mass_preconditioner &&
            parameters.projection_scheme != PressureUpdateType::IrrotationalForm)
        return;

    if (parameters.verbose)
        pcout << "      Building pressure mass matrix preconditioner..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "build preconditioner projection");

    preconditioner_pressure_mass.reset(new LA::PreconditionJacobi());

    LA::PreconditionJacobi::AdditionalData  data;

    preconditioner_pressure_mass->initialize(navier_stokes_mass_matrix.block(1,1),
                                             data);

    rebuild_pressure_mass_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::solve_diffusion_system()
{
    if (parameters.verbose)
        pcout << "      Solving diffusion system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve diffusion");

    // solve linear system
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * navier_stokes_rhs.block(0).l2_norm(),
                                            parameters.abs_tol));;

    LA::BlockVector     distributed_solution(navier_stokes_rhs);
    distributed_solution = navier_stokes_solution;

    try
    {
        if (parameters.convective_scheme == ConvectiveDiscretizationType::LinearImplicit)
        {
            LA::SolverGMRES gmres(solver_control);

            gmres.solve(navier_stokes_matrix.block(0,0),
                        distributed_solution.block(0),
                        navier_stokes_rhs.block(0),
                        *preconditioner_asymmetric_diffusion);
        }
        else
        {
            LA::SolverCG    cg(solver_control);

            cg.solve(navier_stokes_matrix.block(0,0),
                     distributed_solution.block(0),
                     navier_stokes_rhs.block(0),
                     *preconditioner_symmetric_diffusion);
        }
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
        std::cerr << "Unknown exception in diffusion solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    navier_stokes_constraints.distribute(distributed_solution);

    navier_stokes_solution = distributed_solution;

    // write info message
    if (parameters.verbose)
        pcout << "      "
                << solver_control.last_step()
                << " CG iterations for diffusion step"
                << std::endl;
}


template<int dim>
void BuoyantFluidSolver<dim>::solve_projection_system()
{
    if (parameters.verbose)
        pcout << "      Solving projection system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for pressure update
    SolverControl   solver_control(parameters.n_max_iter,
                                   std::max(parameters.rel_tol * navier_stokes_rhs.block(1).l2_norm(),
                                   parameters.abs_tol));

    LA::SolverCG    cg(solver_control);

    LA::BlockVector distributed_solution_vector(navier_stokes_rhs);
    LA::Vector      &distributed_velocity_vector = distributed_solution_vector.block(0);
    LA::Vector      &distributed_pressure_vector = distributed_solution_vector.block(1);
    distributed_velocity_vector = 0.0;

    try
    {
        cg.solve(navier_stokes_laplace_matrix.block(1,1),
                 distributed_pressure_vector,
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
        std::cerr << "Unknown exception in projection solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }

    // write info message
    if (parameters.verbose)
        pcout << "      "
                << solver_control.last_step()
                << " CG iterations for projection step"
                << std::endl;

    /*
     * step 1: set non-distributed pressure update to preliminary solution in
     *         order to apply the constraints and to compute mean value
     */
    stokes_pressure_constraints.distribute(distributed_solution_vector);
    phi_pressure.block(1) = distributed_pressure_vector;

    /*
     * step 2: substract mean value from distributed solution
     */
    {
        const double mean_value = VectorTools::compute_mean_value(mapping,
                                                                  navier_stokes_dof_handler,
                                                                  QGauss<dim>(parameters.velocity_degree - 1),
                                                                  phi_pressure,
                                                                  dim);
        distributed_pressure_vector.add(-mean_value);
    }

    /*
     * step 3: scale distributed solution with timestep
     */
    {

        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        distributed_pressure_vector *= alpha[0] / timestep;

    }

    /*
     * step 4: set non-distributed pressure update to correct solution
     */
    phi_pressure.block(1) = distributed_pressure_vector;

    /*
     * step 5: update the pressure
     */
    if (parameters.projection_scheme == PressureUpdateType::StandardForm)
    {
        // standard update pressure: p^n = p^{n-1} + \phi^n
        navier_stokes_solution.block(1) = old_navier_stokes_solution.block(1);
        navier_stokes_solution.block(1) += phi_pressure.block(1);
    }
    else if (parameters.projection_scheme == PressureUpdateType::IrrotationalForm)
    {
        /*
         * step 5a: compute divergence of the velocity solution and solve the
         *          linear system for the irrotational update
         */
        distributed_velocity_vector = navier_stokes_solution.block(0);

        // set right-hand side vector to divergence of the velocity
        navier_stokes_matrix.block(1,0).vmult(navier_stokes_rhs.block(1),
                                              distributed_velocity_vector);
        navier_stokes_rhs.compress(VectorOperation::add);

        // solve linear system for irrotational update
        SolverControl   solver_control(parameters.n_max_iter,
                                       std::max(parameters.rel_tol * navier_stokes_rhs.block(1).l2_norm(),
                                                parameters.abs_tol));

        LA::SolverCG    cg(solver_control);

        try
        {
            cg.solve(navier_stokes_mass_matrix.block(1,1),
                     distributed_pressure_vector,
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
            std::cerr << "Unknown exception in pressure mass matrix solve!" << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
            std::abort();
        }

        if (parameters.verbose)
            pcout << "      "
                      << solver_control.last_step()
                      << " CG iterations for pressure mass matrix solve"
                      << std::endl;

        /*
         * step 5b: apply the constraints to the preliminary irrotational
         *          pressure update
         */
        navier_stokes_constraints.distribute(distributed_solution_vector);

        /*
         * step 5c: irrotational pressure update:
         *          p^n = p^{n-1} + \phi^n - \gamma_0 * C_1 * \psi,
         *          where \psi is the irrotational update, i.e. the solution
         *          to (\psi, p) = (div(v^n), p)
         */
        // scale the irrotational update
        const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));
        distributed_pressure_vector *= -gamma[0] * equation_coefficients[0];

        navier_stokes_solution.block(1) = old_navier_stokes_solution.block(1);

        navier_stokes_solution.block(1) += phi_pressure.block(1);

        LA::Vector irrotational_pressure_vector(phi_pressure.block(1));
        irrotational_pressure_vector = distributed_pressure_vector;
        navier_stokes_solution.block(1) += irrotational_pressure_vector;
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::compute_initial_pressure()
{
    if (parameters.verbose)
        pcout << "Computing initial pressure..." << std::endl;

    assemble_navier_stokes_matrices();

    build_projection_preconditioner();

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
        if (cell->is_locally_owned())
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

        temperature_fe_values.get_function_values(old_temperature_solution,
                                                  temperature_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
                phi_pressure_gradients[k] = stokes_fe_values[pressure].gradient(k, q);

            const Tensor<1,dim> gravity_vector = EquationData::GravityFunction<dim>().value(stokes_fe_values.quadrature_point(q));

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                local_rhs(i)
                    -= equation_coefficients[2] * temperature_values[q]
                       * gravity_vector * phi_pressure_gradients[i] * stokes_fe_values.JxW(q);

        }
        stokes_pressure_constraints.distribute_local_to_global(local_rhs,
                                                               local_dof_indices,
                                                               navier_stokes_rhs);
    }

    navier_stokes_rhs.compress(VectorOperation::add);


    TimerOutput::Scope  timer_section(computing_timer, "solve projection");

    // solve linear system for pressure update
    SolverControl   solver_control(navier_stokes_laplace_matrix.block(1,1).m(),
                                   std::max(parameters.rel_tol * navier_stokes_rhs.block(1).l2_norm(),
                                   parameters.abs_tol));

    LA::SolverCG    cg(solver_control);

    LA::BlockVector distributed_solution(navier_stokes_rhs);

    try
    {
        cg.solve(navier_stokes_laplace_matrix.block(1,1),
                 distributed_solution.block(1),
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
    stokes_pressure_constraints.distribute(distributed_solution);

    // write info message
    if (parameters.verbose)
        pcout << "   "
                << solver_control.last_step()
                << " CG iterations for projection step"
                << std::endl;

    old_navier_stokes_solution = distributed_solution;
    const double mean_value = VectorTools::compute_mean_value(mapping,
                                                              navier_stokes_dof_handler,
                                                              QGauss<dim>(parameters.velocity_degree - 1),
                                                              old_navier_stokes_solution,
                                                              dim);
    distributed_solution.block(1).add(-mean_value);
    old_navier_stokes_solution.block(1) = distributed_solution.block(1);
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
