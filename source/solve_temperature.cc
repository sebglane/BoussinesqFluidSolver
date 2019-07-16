/*
 * solve_temperature.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <buoyant_fluid_solver.h>

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::project_temperature_field()
{
    if (parameters.geometry == GeometryType::Cavity)
        return;

    assemble_temperature_system();

    // assemble right-hand side vector
    QGauss<dim> quadrature(parameters.temperature_degree + 4);

    FEValues<dim>   temperature_fe_values(mapping,
                                          temperature_fe,
                                          quadrature,
                                          update_values|
                                          update_quadrature_points|
                                          update_JxW_values);
    const unsigned int dofs_per_cell = temperature_fe_values.dofs_per_cell,
                       n_q_points = temperature_fe_values.n_quadrature_points;

    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

    Vector<double>                          local_rhs(dofs_per_cell);
    FullMatrix<double>                      local_matrix_for_bc(dofs_per_cell);

    std::vector<double>                     temperature_values(n_q_points);

    IndexSet    temperature_mass_matrix_partioning(temperature_mass_matrix.n());
    temperature_mass_matrix_partioning.add_range(temperature_mass_matrix.local_range().first,
                                                 temperature_mass_matrix.local_range().second);

    LA::Vector  rhs_vector(temperature_mass_matrix_partioning),
                solution_vector(temperature_mass_matrix_partioning);

    const EquationData::TemperatureInitialValues<dim>
    initial_temperature(parameters.aspect_ratio,
                        1.0,
                        parameters.temperature_perturbation);

    for (auto cell: temperature_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            temperature_fe_values.reinit(cell);

            cell->get_dof_indices(local_dof_indices);

            initial_temperature.value_list(temperature_fe_values.get_quadrature_points(),
                                           temperature_values);

            local_rhs = 0;
            local_matrix_for_bc = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    local_rhs(i) += temperature_values[q] * temperature_fe_values.shape_value(i,q)
                                  * temperature_fe_values.JxW(q);
                    if (temperature_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                            local_matrix_for_bc(j,i) += temperature_fe_values.shape_value(i,q)
                                                      * temperature_fe_values.shape_value(j,q)
                                                      * temperature_fe_values.JxW(q);
                    }
                }
            temperature_constraints.distribute_local_to_global(local_rhs,
                                                               local_dof_indices,
                                                               rhs_vector,
                                                               local_matrix_for_bc);
        }
    rhs_vector.compress(VectorOperation::add);

    // solve linear system
    SolverControl   solver_control(rhs_vector.size(),
                                   1e-12 * rhs_vector.l2_norm());

    TrilinosWrappers::PreconditionJacobi::AdditionalData    data;
    data.omega = 1.3;

    TrilinosWrappers::PreconditionJacobi    preconditioner_mass;
    preconditioner_mass.initialize(temperature_mass_matrix, data);

    try
    {
        LA::SolverCG    cg(solver_control);

        cg.solve(temperature_mass_matrix,
                 solution_vector,
                 rhs_vector,
                 preconditioner_mass);
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
    temperature_constraints.distribute(solution_vector);

    // assign initial condition to temperature field
    temperature_solution = solution_vector;
    old_temperature_solution = solution_vector;
}

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

    preconditioner_temperature.reset(new LA::PreconditionJacobi());

    LA::PreconditionJacobi::AdditionalData     data;

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

    SolverControl solver_control(parameters.max_iter_temperature,
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
template void BuoyantFluid::BuoyantFluidSolver<2>::project_temperature_field();
template void BuoyantFluid::BuoyantFluidSolver<3>::project_temperature_field();

template void BuoyantFluid::BuoyantFluidSolver<2>::temperature_step();
template void BuoyantFluid::BuoyantFluidSolver<3>::temperature_step();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_temperature_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_temperature_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_temperature_system();
