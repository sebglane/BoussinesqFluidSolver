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
navier_stokes_fe(FE_Q<dim>(parameters.velocity_degree), dim,
                 FE_Q<dim>(parameters.velocity_degree - 1), 1),
navier_stokes_dof_handler(triangulation),
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
    x_stokes[0] = navier_stokes_solution;
    x_stokes[1] = old_navier_stokes_solution;
    x_stokes[2] = old_old_navier_stokes_solution;
    SolutionTransfer<dim,BlockVector<double>> stokes_transfer(navier_stokes_dof_handler);

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
        tmp_stokes[0].reinit(navier_stokes_solution);
        tmp_stokes[1].reinit(navier_stokes_solution);
        tmp_stokes[2].reinit(navier_stokes_solution);
        stokes_transfer.interpolate(x_stokes, tmp_stokes);

        navier_stokes_solution = tmp_stokes[0];
        old_navier_stokes_solution = tmp_stokes[1];
        old_old_navier_stokes_solution = tmp_stokes[2];

        navier_stokes_constraints.distribute(navier_stokes_solution);
        navier_stokes_constraints.distribute(old_navier_stokes_solution);
        navier_stokes_constraints.distribute(old_old_navier_stokes_solution);
    }
    // set rebuild flags
    rebuild_navier_stokes_matrices = true;
    rebuild_temperature_matrices = true;
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

        assemble_navier_stokes_system();
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
        old_old_navier_stokes_solution = old_navier_stokes_solution;
        old_navier_stokes_solution = navier_stokes_solution;

        // extrapolate stokes solution
        navier_stokes_solution.sadd(1. + timestep / old_timestep,
                             timestep / old_timestep,
                             old_old_navier_stokes_solution);
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
