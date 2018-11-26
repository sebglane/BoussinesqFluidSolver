/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"
#include "postprocessor.h"
#include "preconditioning.h"

namespace BuoyantFluid {

template<int dim>
BuoyantFluidSolver<dim>::BuoyantFluidSolver(Parameters &parameters_)
:
parameters(parameters_),
imex_coefficients(parameters.imex_scheme),
triangulation(),
mapping(4),
// temperature part
temperature_fe(parameters.temperature_degree),
temperature_dof_handler(triangulation),
// stokes part
stokes_fe(FE_Q<dim>(parameters.velocity_degree), dim,
          FE_Q<dim>(parameters.velocity_degree - 1), 1),
stokes_dof_handler(triangulation),
// coefficients
equation_coefficients{(parameters.rotation ? 2.0/parameters.Ek: 0.0),
                      (parameters.rotation ? 1.0 : std::sqrt(parameters.Pr/ parameters.Ra) ),
                      (parameters.rotation ? parameters.Ra / parameters.Pr  : 1.0 ),
                      (parameters.rotation ? 1.0 / parameters.Pr : 1.0 / std::sqrt(parameters.Ra * parameters.Pr) )},
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
// time stepping
timestep(parameters.initial_timestep),
old_timestep(parameters.initial_timestep)
{
    std::cout << "Boussinesq solver by S. Glane\n"
              << "This program solves the Navier-Stokes system with thermal convection.\n"
              << "The stable Taylor-Hood (P2-P1) element and an approximative Schur complement solver is used.\n\n"
              << "The governing equations are\n\n"
              << "\t-- Incompressibility constraint:\n\t\t div(v) = 0,\n\n"
              << "\t-- Navier-Stokes equation:\n\t\tdv/dt + v . grad(v) + C1 Omega .times. v\n"
              << "\t\t\t\t= - grad(p) + C2 div(grad(v)) - C3 T g,\n\n"
              << "\t-- Heat conduction equation:\n\t\tdT/dt + v . grad(T) = C4 div(grad(T)).\n\n"
              << "The coefficients C1 to C4 depend on the normalization as follows.\n\n";

    // generate a nice table of the equation coefficients
    std::cout << "\n\n"
              << "+-------------------+----------+---------------+----------+-------------------+\n"
              << "|       case        |    C1    |      C2       |    C3    |        C4         |\n"
              << "+-------------------+----------+---------------+----------+-------------------+\n"
              << "| Non-rotating case |    0     | sqrt(Pr / Ra) |    1     | 1 / sqrt(Ra * Pr) |\n"
              << "| Rotating case     |  2 / Ek  |      1        |  Ra / Pr | 1 /  Pr           |\n"
              << "+-------------------+----------+---------------+----------+-------------------+\n";

    std::cout << std::endl << "You have chosen ";

    std::stringstream ss;
    ss << "+----------+----------+----------+----------+----------+----------+----------+\n"
       << "|    Ek    |    Ra    |    Pr    |    C1    |    C2    |    C3    |    C4    |\n"
       << "+----------+----------+----------+----------+----------+----------+----------+\n";

    if (parameters.rotation)
    {
        rotation_vector[dim-1] = 1.0;

        std::cout << "the rotating case with the following parameters: "
                  << std::endl;
        ss << "| ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ek;
        ss << " | ";
    }
    else
    {
        std::cout << "the non-rotating case with the following parameters: "
                  << std::endl;
        ss << "|     0    | ";
    }

    ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ra;
    ss << " | ";
    ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pr;
    ss << " | ";


    for (unsigned int n=0; n<4; ++n)
    {
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << equation_coefficients[n];
        ss << " | ";
    }

    ss << "\n+----------+----------+----------+----------+----------+----------+----------+\n";

    std::cout << std::endl << ss.str() << std::endl;

    std::cout << std::endl << std::fixed << std::flush;
}






template<int dim>
void BuoyantFluidSolver<dim>::build_temperature_preconditioner()
{
    if (!rebuild_temperature_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build temperature preconditioner");

    preconditioner_T.reset(new PreconditionerTypeT());

    PreconditionerTypeT::AdditionalData     data;
    data.relaxation = 0.6;

    preconditioner_T->initialize(temperature_matrix,
                                 data);

    rebuild_temperature_preconditioner = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::build_stokes_preconditioner()
{
    if (!rebuild_stokes_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build stokes preconditioner");

    Assert(!rebuild_stokes_matrices, ExcInternalError());

    std::cout << "   Building stokes preconditioner..." << std::endl;

    preconditioner_A = std::shared_ptr<PreconditionerTypeA>
                       (new PreconditionerTypeA());

    std::vector<std::vector<bool>>  constant_modes;
    FEValuesExtractors::Vector      velocity_components(0);
    DoFTools::extract_constant_modes(stokes_dof_handler,
                                     stokes_fe.component_mask(velocity_components),
                                     constant_modes);

    PreconditionerTypeA::AdditionalData preconditioner_A_data;
    preconditioner_A_data.constant_modes = constant_modes;
    preconditioner_A_data.elliptic = true;
    preconditioner_A_data.higher_order_elements = true;
    preconditioner_A_data.smoother_sweeps = 2;
    preconditioner_A_data.aggregation_threshold = 0.02;

    preconditioner_A->initialize(stokes_matrix.block(0,0),
                                 preconditioner_A_data);

    rebuild_stokes_preconditioner = false;
}


template<int dim>
std::pair<double, double> BuoyantFluidSolver<dim>::compute_rms_values() const
{
    const QGauss<dim> quadrature_formula(parameters.velocity_degree + 1);

    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> stokes_fe_values(mapping,
                                   stokes_fe,
                                   quadrature_formula,
                                   update_values|update_JxW_values);

    FEValues<dim> temperature_fe_values(mapping,
                                        temperature_fe,
                                        quadrature_formula,
                                        update_values);

    std::vector<double>         temperature_values(n_q_points);
    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double rms_velocity = 0;
    double rms_temperature = 0;
    double volume = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    temperature_cell = temperature_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();

    for (; cell != endc; ++cell, ++temperature_cell)
    {
        stokes_fe_values.reinit(cell);
        temperature_fe_values.reinit(temperature_cell);

        temperature_fe_values.get_function_values(temperature_solution,
                                                  temperature_values);
        stokes_fe_values[velocities].get_function_values(stokes_solution,
                                                         velocity_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            rms_velocity += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
            rms_temperature += temperature_values[q] * temperature_values[q] * stokes_fe_values.JxW(q);
            volume += stokes_fe_values.JxW(q);
        }
    }

    rms_velocity /= volume;
    AssertIsFinite(rms_velocity);
    Assert(rms_velocity >= 0, ExcLowerRangeType<double>(rms_velocity, 0));

    rms_temperature /= volume;
    AssertIsFinite(rms_temperature);
    Assert(rms_temperature >= 0, ExcLowerRangeType<double>(rms_temperature, 0));

    return std::pair<double,double>(std::sqrt(rms_velocity), std::sqrt(rms_temperature));
}

template <int dim>
double BuoyantFluidSolver<dim>::compute_cfl_number() const
{
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping,
                            stokes_fe,
                            quadrature_formula,
                            update_values);

    std::vector<Tensor<1,dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double max_cfl = 0;

    for (auto cell : stokes_dof_handler.active_cell_iterators())
    {
        fe_values.reinit (cell);
        fe_values[velocities].get_function_values(stokes_solution,
                                                  velocity_values);
        double max_cell_velocity = 0;
        for (unsigned int q=0; q<n_q_points; ++q)
            max_cell_velocity = std::max(max_cell_velocity,
                                         velocity_values[q].norm());
        max_cfl = std::max(max_cfl,
                           max_cell_velocity / cell->diameter());
    }
    return max_cfl * timestep;
}

template<int dim>
void BuoyantFluidSolver<dim>::update_timestep(const double current_cfl_number)
{
    TimerOutput::Scope  timer_section(computing_timer, "update time step");

    std::cout << "   Updating time step..." << std::endl;

    old_timestep = timestep;
    timestep_modified = false;

    if (current_cfl_number > parameters.cfl_max || current_cfl_number < parameters.cfl_min)
    {
        timestep = 0.5 * (parameters.cfl_min + parameters.cfl_max)
                        * old_timestep / current_cfl_number;
        if (timestep == old_timestep)
            return;
        else if (timestep > parameters.max_timestep
                && old_timestep == parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
            return;
        }
        else if (timestep > parameters.max_timestep
                && old_timestep != parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
            timestep_modified = true;
        }
        else if (timestep < parameters.min_timestep)
        {
            ExcLowerRangeType<double>(timestep, parameters.min_timestep);
        }
        else if (timestep < parameters.max_timestep)
        {
            timestep_modified = true;
        }
    }
    if (timestep_modified)
        std::cout << "      time step changed from "
                  << std::setw(6) << std::setprecision(2) << std::scientific << old_timestep
                  << " to "
                  << std::setw(6) << std::setprecision(2) << std::scientific << timestep
                  << std::endl;
}

template<int dim>
void BuoyantFluidSolver<dim>::output_results() const
{
    std::cout << "   Output results..." << std::endl;

    // create joint finite element
    const FESystem<dim> joint_fe(stokes_fe, 1,
                                 temperature_fe, 1);

    // create joint dof handler
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    Assert(joint_dof_handler.n_dofs() ==
           stokes_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
           ExcInternalError());

    // create joint solution
    Vector<double>      joint_solution;
    joint_solution.reinit(joint_dof_handler.n_dofs());

    {
        std::vector<types::global_dof_index> local_joint_dof_indices(joint_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_stokes_dof_indices(stokes_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_temperature_dof_indices(temperature_fe.dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
        joint_cell       = joint_dof_handler.begin_active(),
        joint_endc       = joint_dof_handler.end(),
        stokes_cell      = stokes_dof_handler.begin_active(),
        temperature_cell = temperature_dof_handler.begin_active();
        for (; joint_cell!=joint_endc; ++joint_cell, ++stokes_cell, ++temperature_cell)
        {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            stokes_cell->get_dof_indices(local_stokes_dof_indices);
            temperature_cell->get_dof_indices(local_temperature_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    Assert (joint_fe.system_to_base_index(i).second < local_stokes_dof_indices.size(),
                            ExcInternalError());
                    joint_solution(local_joint_dof_indices[i])
                    = stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else
                {
                    Assert (joint_fe.system_to_base_index(i).first.first == 1,
                            ExcInternalError());
                    Assert (joint_fe.system_to_base_index(i).second < local_temperature_dof_indices.size(),
                            ExcInternalError());
                    joint_solution(local_joint_dof_indices[i])
                    = temperature_solution(local_temperature_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
        }
    }

    // create post processor
    PostProcessor<dim>   postprocessor;

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    data_out.add_data_vector(joint_solution, postprocessor);
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(timestep_number, 5) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

template<int dim>
void BuoyantFluidSolver<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    // error estimation based on temperature
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(temperature_dof_handler,
                                       QGauss<dim-1>(parameters.temperature_degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       temperature_solution,
                                       estimated_error_per_cell);
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.7, 0.3);

    // clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > parameters.n_max_levels)
        for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_max_levels))
            cell->clear_refine_flag();


    // preparing temperature solution transfer
    std::vector<Vector<double>> x_temperature(3);
    x_temperature[0] = temperature_solution;
    x_temperature[1] = old_temperature_solution;
    x_temperature[2] = old_old_temperature_solution;
    SolutionTransfer<dim,Vector<double>> temperature_transfer(temperature_dof_handler);

    // preparing temperature stokes transfer
    std::vector<BlockVector<double>> x_stokes(3);
    x_stokes[0] = stokes_solution;
    x_stokes[1] = old_stokes_solution;
    x_stokes[2] = old_old_stokes_solution;
    SolutionTransfer<dim,BlockVector<double>> stokes_transfer(stokes_dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    temperature_transfer.prepare_for_coarsening_and_refinement(x_temperature);
    stokes_transfer.prepare_for_coarsening_and_refinement(x_stokes);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of temperature solution
    {
        std::vector<Vector<double>> tmp_temperature(3);
        tmp_temperature[0].reinit(temperature_solution);
        tmp_temperature[1].reinit(temperature_solution);
        tmp_temperature[2].reinit(temperature_solution);
        temperature_transfer.interpolate(x_temperature, tmp_temperature);

        temperature_solution = tmp_temperature[0];
        old_temperature_solution = tmp_temperature[1];
        old_old_temperature_solution = tmp_temperature[2];

        temperature_constraints.distribute(temperature_solution);
        temperature_constraints.distribute(old_temperature_solution);
        temperature_constraints.distribute(old_old_temperature_solution);
    }
    // transfer of stokes solution
    {
        std::vector<BlockVector<double>>    tmp_stokes(3);
        tmp_stokes[0].reinit(stokes_solution);
        tmp_stokes[1].reinit(stokes_solution);
        tmp_stokes[2].reinit(stokes_solution);
        stokes_transfer.interpolate(x_stokes, tmp_stokes);

        stokes_solution = tmp_stokes[0];
        old_stokes_solution = tmp_stokes[1];
        old_old_stokes_solution = tmp_stokes[2];

        stokes_constraints.distribute(stokes_solution);
        stokes_constraints.distribute(old_stokes_solution);
        stokes_constraints.distribute(old_old_stokes_solution);
    }
    // set rebuild flags
    rebuild_stokes_matrices = true;
    rebuild_temperature_matrices = true;
}


template <int dim>
void BuoyantFluidSolver<dim>::solve()
{
    {
        std::cout << "   Solving temperature system..." << std::endl;
        TimerOutput::Scope  timer_section(computing_timer, "temperature solve");

        temperature_constraints.set_zero(temperature_solution);

        SolverControl solver_control(temperature_matrix.m(),
                1e-12 * temperature_rhs.l2_norm());

        SolverCG<>   cg(solver_control);
        cg.solve(temperature_matrix,
                 temperature_solution,
                 temperature_rhs,
                 *preconditioner_T);

        temperature_constraints.distribute(temperature_solution);

        std::cout << "      "
                << solver_control.last_step()
                << " CG iterations for temperature"
                << std::endl;
    }
    {
        std::cout << "   Solving stokes system..." << std::endl;

        TimerOutput::Scope  timer_section(computing_timer, "stokes solve");

        stokes_constraints.set_zero(stokes_solution);

        PrimitiveVectorMemory<BlockVector<double>> vector_memory;

        SolverControl solver_control(parameters.n_max_iter,
                                     std::max(parameters.abs_tol,
                                              parameters.rel_tol * stokes_rhs.l2_norm()));

        SolverFGMRES<BlockVector<double>>
        solver(solver_control,
               vector_memory,
               SolverFGMRES<BlockVector<double>>::AdditionalData(30, true));

        const Preconditioning::BlockSchurPreconditioner
        <PreconditionerTypeA, PreconditionerTypeMp, PreconditionerTypeKp>
        preconditioner(stokes_matrix,
                       pressure_mass_matrix,
                       stokes_laplace_matrix.block(1,1),
                       *preconditioner_A,
                       *preconditioner_Kp,
                       factor_Kp,
                       *preconditioner_Mp,
                       factor_Mp,
                       false);

        solver.solve(stokes_matrix,
                     stokes_solution,
                     stokes_rhs,
                     preconditioner);

        std::cout << "      "
                  << solver_control.last_step()
                  << " GMRES iterations for stokes system, "
                  << " (A: " << preconditioner.n_iterations_A()
                  << ", Kp: " << preconditioner.n_iterations_Kp()
                  << ", Mp: " << preconditioner.n_iterations_Mp()
                  << ")"
                  << std::endl;

        stokes_constraints.distribute(stokes_solution);
    }
}


template<int dim>
void BuoyantFluidSolver<dim>::run()
{
    make_grid();

    setup_dofs();

    const EquationData::TemperatureInitialValues<dim>
    initial_temperature(parameters.aspect_ratio,
                        1.0,
                        0.5,
                        -0.5);

    VectorTools::interpolate(mapping,
                             temperature_dof_handler,
                             initial_temperature,
                             old_temperature_solution);

    temperature_constraints.distribute(old_temperature_solution);

    temperature_solution = old_temperature_solution;

    output_results();

    double time = 0;
    double cfl_number = 0;

    do
    {
        std::cout << "step: " << Utilities::int_to_string(timestep_number, 8) << ", "
                  << "time: " << time << ", "
                  << "time step: " << timestep
                  << std::endl;

        assemble_stokes_system();
        build_stokes_preconditioner();

        assemble_temperature_system();
        build_temperature_preconditioner();

        solve();
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute rms values");

            const std::pair<double,double> rms_values = compute_rms_values();

            std::cout << "   velocity rms value: "
                      << rms_values.first
                      << std::endl
                      << "   temperature rms value: "
                      << rms_values.second
                      << std::endl;
        }
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute cfl number");

            cfl_number = compute_cfl_number();

            std::cout << "   current cfl number: "
                      << cfl_number
                      << std::endl;
        }
        if (timestep_number % parameters.output_frequency == 0
                && timestep_number != 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }
        // mesh refinement
        if ((timestep_number > 0)
                && (timestep_number % parameters.refinement_frequency == 0))
            refine_mesh();
        // adjust time step
        if (parameters.adaptive_timestep && timestep_number > 1)
            update_timestep(cfl_number);

        // copy temperature solution
        old_old_temperature_solution = old_temperature_solution;
        old_temperature_solution = temperature_solution;

        // extrapolate temperature solution
        temperature_solution.sadd(1. + timestep / old_timestep,
                                  timestep / old_timestep,
                                  old_old_temperature_solution);

        // extrapolate stokes solution
        old_old_stokes_solution = old_stokes_solution;
        old_stokes_solution = stokes_solution;

        // extrapolate stokes solution
        stokes_solution.sadd(1. + timestep / old_timestep,
                             timestep / old_timestep,
                             old_old_stokes_solution);
        // advance in time
        time += timestep;
        ++timestep_number;

    } while (timestep_number <= parameters.n_steps);

    if (parameters.n_steps % parameters.output_frequency != 0)
        output_results();

    std::cout << std::fixed;

    computing_timer.print_summary();
    computing_timer.reset();

    std::cout << std::endl;
}

}  // namespace BouyantFluid

// explicit instantiation
template class BuoyantFluid::BuoyantFluidSolver<2>;
template class BuoyantFluid::BuoyantFluidSolver<3>;
