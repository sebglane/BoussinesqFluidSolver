/*
 * solve_magnetic.cc
 *
 *  Created on: Jun 28, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_control.h>

#include <buoyant_fluid_solver.h>

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::project_magnetic_field()
{
    assemble_magnetic_matrices();

    // assemble right-hand side vector
    QGauss<dim> quadrature(parameters.magnetic_degree + 2);

    FEValues<dim>   magnetic_fe_values(mapping,
                                       magnetic_fe,
                                       quadrature,
                                       update_values|
                                       update_quadrature_points|
                                       update_JxW_values);
    const unsigned int dofs_per_cell = magnetic_fe_values.dofs_per_cell,
                       n_q_points = magnetic_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    magnetic_field(0);

    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

    Vector<double>                          local_rhs(dofs_per_cell);
    FullMatrix<double>                      local_matrix_for_bc(dofs_per_cell);

    std::vector<Tensor<1,dim>>              magnetic_field_values(n_q_points);

    LA::BlockVector     distributed_solution_vector(magnetic_rhs),
                        distributed_rhs_vector(magnetic_rhs);
    LA::Vector          &solution_vector = distributed_solution_vector.block(0);
    LA::Vector          &rhs_vector = distributed_rhs_vector.block(0);

    const EquationData::MagneticFieldInitialValues<dim>
    initial_magnetic_field(parameters.aspect_ratio,
                           1.0);

    for (auto cell: magnetic_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            magnetic_fe_values.reinit(cell);

            cell->get_dof_indices(local_dof_indices);

            initial_magnetic_field.value_list(magnetic_fe_values.get_quadrature_points(),
                                              magnetic_field_values);

            local_rhs = 0;
            local_matrix_for_bc = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    local_rhs(i) += magnetic_field_values[q] * magnetic_fe_values[magnetic_field].value(i,q)
                                  * magnetic_fe_values.JxW(q);
                    if (magnetic_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                            local_matrix_for_bc(j,i) += magnetic_fe_values[magnetic_field].value(i,q)
                                                      * magnetic_fe_values[magnetic_field].value(j,q)
                                                      * magnetic_fe_values.JxW(q);
                    }
                }
            magnetic_constraints.distribute_local_to_global(local_rhs,
                                                            local_dof_indices,
                                                            distributed_rhs_vector,
                                                            local_matrix_for_bc);
        }
    distributed_rhs_vector.compress(VectorOperation::add);

    // solve linear system
    SolverControl   solver_control(rhs_vector.size(),
                                   1e-12 * rhs_vector.l2_norm());

    TrilinosWrappers::PreconditionJacobi::AdditionalData    data;
    data.omega = 1.3;

    TrilinosWrappers::PreconditionJacobi    preconditioner_mass;
    preconditioner_mass.initialize(magnetic_mass_matrix.block(0,0), data);

    try
    {
        LA::SolverCG    cg(solver_control);

        cg.solve(magnetic_mass_matrix.block(0,0),
                 solution_vector,
                 rhs_vector,
                 preconditioner_mass);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception in magnetic mass matrix solve: " << std::endl
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
        std::cerr << "Unknown exception in magnetic mass matrix solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    magnetic_constraints.distribute(distributed_solution_vector);

    // assign initial condition to magnetic field
    magnetic_solution.block(0) = solution_vector;
    old_magnetic_solution.block(0) = solution_vector;
}

template<int dim>
void BuoyantFluidSolver<dim>::magnetic_step()
{
    if (parameters.verbose)
        pcout << "   Magnetic step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_magnetic_diffusion_system();

    // rebuild preconditioner for diffusion step
    build_magnetic_diffusion_preconditioner();

    // solve projection step
    solve_magnetic_diffusion_system();

    // assemble right-hand side (and system if necessary)
    assemble_magnetic_projection_system();

    // rebuild magnetic preconditioners for projection step
    build_magnetic_projection_preconditioner();
    build_magnetic_pressure_mass_preconditioner();

    // solve projection system
    solve_magnetic_projection_system();
}

template<int dim>
void BuoyantFluidSolver<dim>::build_magnetic_diffusion_preconditioner()
{
    if (!rebuild_magnetic_diffusion_preconditioner)
        return;

    if (parameters.verbose)
        pcout << "      Building magnetic diffusion preconditioner..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "build magnetic diffusion preconditioner ");

    preconditioner_magnetic_diffusion.reset(new LA::PreconditionAMG());

    LA::PreconditionAMG::AdditionalData     data;
    data.higher_order_elements = true;
    data.elliptic = true;

    preconditioner_magnetic_diffusion
    ->initialize(magnetic_matrix.block(0,0),
                 data);

    rebuild_magnetic_diffusion_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_magnetic_projection_preconditioner()
{
    if (!rebuild_magnetic_projection_preconditioner)
        return;

    if (parameters.verbose)
        pcout << "      Building magnetic projection preconditioner..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "build magnetic projection preconditioner ");

    preconditioner_magnetic_projection.reset(new LA::PreconditionAMG());

    LA::PreconditionAMG::AdditionalData     data;

    preconditioner_magnetic_projection->initialize(magnetic_laplace_matrix.block(1,1),
                                                   data);

    rebuild_magnetic_projection_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_magnetic_pressure_mass_preconditioner()
{
    if (!rebuild_magnetic_pressure_mass_preconditioner &&
            parameters.projection_scheme != PressureUpdateType::IrrotationalForm)
        return;

    if (parameters.verbose)
        pcout << "      Building magnetic pressure mass matrix preconditioner..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "build magnetic projection preconditioner");

    preconditioner_magnetic_pressure_mass.reset(new LA::PreconditionJacobi());

    LA::PreconditionJacobi::AdditionalData  data;

    preconditioner_magnetic_pressure_mass->initialize(magnetic_mass_matrix.block(1,1),
                                                      data);

    rebuild_magnetic_pressure_mass_preconditioner = false;
}


template<int dim>
void BuoyantFluidSolver<dim>::solve_magnetic_diffusion_system()
{
    if (parameters.verbose)
        pcout << "      Solving magnetic diffusion system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve diffusion");

    // solve linear system
    SolverControl   solver_control(parameters.max_iter_magnetic,
                                   std::max(parameters.rel_tol * magnetic_rhs.block(0).l2_norm(),
                                            parameters.abs_tol));;

    LA::BlockVector     distributed_solution(magnetic_rhs);
    distributed_solution = magnetic_solution;

    try
    {
        LA::SolverGMRES gmres(solver_control);

        gmres.solve(magnetic_matrix.block(0,0),
                    distributed_solution.block(0),
                    magnetic_rhs.block(0),
                    *preconditioner_magnetic_diffusion);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception in magnetic diffusion solve: " << std::endl
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
        std::cerr << "Unknown exception in magnetic diffusion solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
    magnetic_constraints.distribute(distributed_solution);

    magnetic_solution = distributed_solution;

    // write info message
    if (parameters.verbose)
        pcout << "      "
                << solver_control.last_step()
                << " CG iterations for magnetic diffusion step"
                << std::endl;
}


template<int dim>
void BuoyantFluidSolver<dim>::solve_magnetic_projection_system()
{
    if (parameters.verbose)
        pcout << "      Solving magnetic projection system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "solve magnetic projection");

    // solve linear system for irrotational pseudo pressure update
    SolverControl   solver_control(parameters.max_iter_magnetic,
                                   std::max(parameters.rel_tol * magnetic_rhs.block(1).l2_norm(),
                                   parameters.abs_tol));

    LA::SolverCG    cg(solver_control);

    LA::BlockVector     distributed_solution_vector(magnetic_rhs);
    LA::Vector          &distributed_magnetic_vector = distributed_solution_vector.block(0);
    LA::Vector          &distributed_pseudo_pressure_vector = distributed_solution_vector.block(1);


    distributed_solution_vector = phi_pseudo_pressure;

    try
    {
        cg.solve(magnetic_laplace_matrix.block(1,1),
                 distributed_pseudo_pressure_vector,
                 magnetic_rhs.block(1),
                 *preconditioner_magnetic_projection);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception in magnetic projection solve: " << std::endl
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
        std::cerr << "Unknown exception in magnetic projection solve!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }

    // write info message
    if (parameters.verbose)
        pcout << "      "
                << solver_control.last_step()
                << " CG iterations for magnetic projection step"
                << std::endl;

    /*
     * step 1: set non-distributed pressure update to preliminary solution in
     *         order to apply the constraints and to compute mean value
     */
    magnetic_constraints.distribute(distributed_solution_vector);
    phi_pseudo_pressure.block(1) = distributed_pseudo_pressure_vector;

    /*
     * step 2: scale distributed solution with timestep
     */
    {
        const std::vector<double> alpha = (timestep_number != 0?
                                           imex_coefficients.alpha(timestep/old_timestep):
                                           std::vector<double>({1.0,-1.0,0.0}));

        distributed_solution_vector.block(1) *= alpha[0] / timestep;
    }

    /*
     * step 3: set non-distributed pseudo pressure update to correct solution
     */
    phi_pseudo_pressure.block(1) = distributed_solution_vector.block(1);

    /*
     * step 5: update the pseudo pressure
     */
    if (parameters.magnetic_projection_scheme == PressureUpdateType::StandardForm)
    {
        // standard update pressure: p^n = p^{n-1} + \phi^n
        magnetic_solution.block(1) = old_magnetic_solution.block(1);
        magnetic_solution.block(1) += phi_pressure.block(1);
    }
    else if (parameters.magnetic_projection_scheme == PressureUpdateType::IrrotationalForm)
    {
        /*
         * step 5a: compute divergence of the velocity solution and solve the
         *          linear system for the irrotational update
         */
        distributed_magnetic_vector = magnetic_solution.block(0);

        // set right-hand side vector to divergence of the magnetic field
        magnetic_matrix.block(0,1).vmult(magnetic_rhs.block(1),
                                         distributed_solution_vector.block(0));
        magnetic_rhs.compress(VectorOperation::add);

        // solve linear system for irrotational update
        SolverControl   solver_control(parameters.max_iter_magnetic,
                                       std::max(parameters.rel_tol * magnetic_rhs.block(1).l2_norm(),
                                                parameters.abs_tol));

        LA::SolverCG    cg(solver_control);

        try
        {
            cg.solve(magnetic_mass_matrix.block(1,1),
                     distributed_pseudo_pressure_vector,
                     magnetic_rhs.block(1),
                     *preconditioner_magnetic_pressure_mass);
        }
        catch (std::exception &exc)
        {
            std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
            std::cerr << "Exception in magnetic pressure mass matrix solve: " << std::endl
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
            std::cerr << "Unknown exception in magnetic pressure mass matrix solve!" << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
            std::abort();
        }

        if (parameters.verbose)
            pcout << "      "
                      << solver_control.last_step()
                      << " CG iterations for magnetic pressure mass matrix solve"
                      << std::endl;

        /*
         * step 5b: apply the constraints to the preliminary irrotational pseudo
         *          pressure update
         */
        magnetic_constraints.distribute(distributed_solution_vector);

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
        distributed_pseudo_pressure_vector *= -gamma[0] * equation_coefficients[5];

        magnetic_solution.block(1) = old_magnetic_solution.block(1);

        magnetic_solution.block(1) += phi_pseudo_pressure.block(1);

        LA::Vector  irrotational_pseudo_pressure_vector(phi_pseudo_pressure.block(1));
        irrotational_pseudo_pressure_vector = distributed_pseudo_pressure_vector;
        magnetic_solution.block(1) += irrotational_pseudo_pressure_vector;
    }
}
}  // namespace BuoyantFluid

template void BuoyantFluid::BuoyantFluidSolver<2>::project_magnetic_field();
template void BuoyantFluid::BuoyantFluidSolver<3>::project_magnetic_field();

template void BuoyantFluid::BuoyantFluidSolver<2>::magnetic_step();
template void BuoyantFluid::BuoyantFluidSolver<3>::magnetic_step();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_magnetic_diffusion_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_magnetic_diffusion_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_magnetic_projection_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_magnetic_projection_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_magnetic_pressure_mass_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_magnetic_pressure_mass_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_magnetic_diffusion_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_magnetic_diffusion_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_magnetic_projection_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_magnetic_projection_system();
