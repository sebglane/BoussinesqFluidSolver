/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"
#include "postprocessor.h"

namespace BuoyantFluid {

template<int dim>
BuoyantFluidSolver<dim>::BuoyantFluidSolver(Parameters &parameters_)
:
mpi_communicator(MPI_COMM_WORLD),
parameters(parameters_),
imex_coefficients(parameters.imex_scheme),
triangulation(mpi_communicator),
mapping(4),
// temperature part
temperature_fe(parameters.temperature_degree),
temperature_dof_handler(triangulation),
// stokes part
navier_stokes_fe(FESystem<dim>(FE_Q<dim>(parameters.velocity_degree), dim), 1,
                 FE_Q<dim>(parameters.velocity_degree - 1), 1),
navier_stokes_dof_handler(triangulation),
// coefficients
equation_coefficients{(parameters.rotation ? 2.0/parameters.Ek: 0.0),
                      (parameters.rotation ? 1.0 : std::sqrt(parameters.Pr/ parameters.Ra) ),
                      (parameters.rotation ? parameters.Ra / parameters.Pr  : 1.0 ),
                      (parameters.rotation ? 1.0 / parameters.Pr : 1.0 / std::sqrt(parameters.Ra * parameters.Pr) )},
// parallel output
pcout(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
// monitor
computing_timer(mpi_communicator, pcout,
                TimerOutput::summary, TimerOutput::wall_times),
// time stepping
timestep(parameters.initial_timestep),
old_timestep(parameters.initial_timestep),
// benchmarking
phi_benchmark(-2.*numbers::PI)
{
    pcout << "Boussinesq solver written by Sebastian Glane\n"
          << "This program solves the Navier-Stokes system with thermal convection.\n"
          << "The stable Taylor-Hood (P2-P1) element and a pressure projection scheme is applied.\n"
          << "For time discretization an adaptive IMEX time stepping is used.\n\n"
          << "The governing equations are\n\n"
          << "\t-- Incompressibility constraint:\n\t\t div(v) = 0,\n\n"
          << "\t-- Navier-Stokes equation:\n\t\tdv/dt + v . grad(v) + C1 Omega .times. v\n"
          << "\t\t\t\t= - grad(p) + C2 div(grad(v)) - C3 T g,\n\n"
          << "\t-- Heat conduction equation:\n\t\tdT/dt + v . grad(T) = C4 div(grad(T)).\n\n"
          << "The coefficients C1 to C4 depend on the normalization as follows.\n\n";

    // generate a nice table of the equation coefficients
    pcout << "+-------------------+----------+---------------+----------+-------------------+\n"
          << "|       case        |    C1    |      C2       |    C3    |        C4         |\n"
          << "+-------------------+----------+---------------+----------+-------------------+\n"
          << "| Non-rotating case |    0     | sqrt(Pr / Ra) |    1     | 1 / sqrt(Ra * Pr) |\n"
          << "| Rotating case     |  2 / Ek  |      1        |  Ra / Pr | 1 /  Pr           |\n"
          << "+-------------------+----------+---------------+----------+-------------------+\n";

    pcout << std::endl << "You have chosen ";

    std::stringstream ss;
    ss << "+----------+----------+----------+----------+----------+----------+----------+\n"
       << "|    Ek    |    Ra    |    Pr    |    C1    |    C2    |    C3    |    C4    |\n"
       << "+----------+----------+----------+----------+----------+----------+----------+\n";

    if (parameters.rotation)
    {
        rotation_vector[dim-1] = 1.0;

        pcout << "the rotating case with the following parameters: "
              << std::endl;
        ss << "| ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ek;
        ss << " | ";
    }
    else
    {
        pcout << "the non-rotating case with the following parameters: "
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

    pcout << std::endl << ss.str()
          << std::endl << std::fixed << std::flush;

    benchmark_table.declare_column("time step");
    benchmark_table.declare_column("time");
    benchmark_table.declare_column("phi");
    benchmark_table.declare_column("temperature");
    benchmark_table.declare_column("azimuthal velocity");
}

template<int dim>
void BuoyantFluidSolver<dim>::update_timestep(const double current_cfl_number)
{
    TimerOutput::Scope  timer_section(computing_timer, "update time step");

    if (parameters.verbose)
        pcout << "   Updating time step..." << std::endl;

    old_timestep = timestep;
    timestep_modified = false;

    if (current_cfl_number > parameters.cfl_max || current_cfl_number < parameters.cfl_min)
    {
        timestep = 0.5 * (parameters.cfl_min + parameters.cfl_max)
                        * old_timestep / current_cfl_number;
        if (timestep > parameters.max_timestep
                 && old_timestep != parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
            timestep_modified = true;
        }
        else if (timestep > parameters.max_timestep
                 && old_timestep == parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
        }
        else if (timestep < parameters.max_timestep
                 && timestep > parameters.min_timestep)
        {
            timestep_modified = true;
        }
        else if (timestep < parameters.min_timestep)
        {
            Assert(false,
                   ExcLowerRangeType<double>(timestep, parameters.min_timestep));
        }
    }

    if (timestep_modified)
        pcout << (parameters.verbose? "   ": "")
              << "   Time step changed from "
              << std::setw(6) << std::scientific << old_timestep
              << " to "
              << std::setw(6) << std::scientific << timestep
              << std::fixed << std::endl
              << "   New cfl number: " << current_cfl_number / old_timestep * timestep
              << std::endl;
}

template<int dim>
void BuoyantFluidSolver<dim>::refine_mesh()
{
    pcout << "Mesh refinement..." << std::endl;

    parallel::distributed::SolutionTransfer<dim,LA::Vector>
    temperature_transfer(temperature_dof_handler);

    parallel::distributed::SolutionTransfer<dim,LA::BlockVector>
    stokes_transfer(navier_stokes_dof_handler);

    {
    TimerOutput::Scope timer_section(computing_timer, "refine mesh (part 1)");

    // error estimation based on temperature
    Vector<float>   estimated_temperature_error(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(mapping,
                                       temperature_dof_handler,
                                       QGauss<dim-1>(parameters.temperature_degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       temperature_solution,
                                       estimated_temperature_error,
                                       ComponentMask(),
                                       nullptr,
                                       0,
                                       triangulation.locally_owned_subdomain());
    const float max_temperature_error
    = *std::max_element(estimated_temperature_error.begin(),
                        estimated_temperature_error.end());
    AssertIsFinite(max_temperature_error);
    Assert(max_temperature_error > 0., ExcLowerRangeType<double>(max_temperature_error, 0.));

    estimated_temperature_error /= max_temperature_error;

    // error estimation based on velocity
    Vector<float>   estimated_velocity_error(triangulation.n_active_cells());
    const FEValuesExtractors::Vector    velocities(0);
    KellyErrorEstimator<dim>::estimate(mapping,
                                       navier_stokes_dof_handler,
                                       QGauss<dim-1>(parameters.velocity_degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       navier_stokes_solution,
                                       estimated_velocity_error,
                                       navier_stokes_fe.component_mask(velocities),
                                       nullptr,
                                       0,
                                       triangulation.locally_owned_subdomain());
    const float max_velocity_error
    = *std::max_element(estimated_velocity_error.begin(),
                        estimated_velocity_error.end());

    // error combined error estimate
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());
    if (max_velocity_error > 0.)
    {
        AssertIsFinite(max_velocity_error);
        estimated_velocity_error /= max_velocity_error;

        estimated_error_per_cell.add(0.5, estimated_temperature_error);
        estimated_error_per_cell.add(0.5, estimated_velocity_error);
    }
    else
        estimated_error_per_cell = estimated_temperature_error;



    // set refinement flags
    parallel::distributed::
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.3, 0.1);

    // clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > parameters.n_max_levels)
        for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_max_levels))
            cell->clear_refine_flag();

    // clear coarsen flags if level decreases minimum
    for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_global_refinements))
        cell->clear_coarsen_flag();

    // count number of cells to be refined and coarsened
    unsigned int local_cell_counts[2] = {0, 0};
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned() && cell->refine_flag_set())
            local_cell_counts[0] += 1;
        else if (cell->is_locally_owned() && cell->coarsen_flag_set())
            local_cell_counts[1] += 1;

    unsigned int global_cell_counts[2];
    Utilities::MPI::sum(local_cell_counts, mpi_communicator, global_cell_counts);

    pcout << "   Number of cells refined: " << global_cell_counts[0] << std::endl
          << "   Number of cells coarsened: " << global_cell_counts[1] << std::endl;

    // preparing temperature solution transfer
    std::vector<const LA::Vector *> x_temperature(3);
    x_temperature[0] = &temperature_solution;
    x_temperature[1] = &old_temperature_solution;
    x_temperature[2] = &old_old_temperature_solution;

    // preparing stokes solution transfer
    std::vector<const LA::BlockVector *> x_stokes(6);
    x_stokes[0] = &navier_stokes_solution;
    x_stokes[1] = &old_navier_stokes_solution;
    x_stokes[2] = &old_old_navier_stokes_solution;
    x_stokes[3] = &phi_pressure;
    x_stokes[4] = &old_phi_pressure;
    x_stokes[5] = &old_old_phi_pressure;

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    temperature_transfer.prepare_for_coarsening_and_refinement(x_temperature);
    stokes_transfer.prepare_for_coarsening_and_refinement(x_stokes);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();
    }

    std::vector<unsigned int>   locally_active_cells(triangulation.n_global_levels());
    for (unsigned int level=0; level < triangulation.n_levels(); ++level)
        for (auto cell: triangulation.active_cell_iterators())
            if ((unsigned int) cell->level() == level && cell->is_locally_owned())
                locally_active_cells[level] += 1;
    pcout << "   Number of cells (on level): ";
    for (unsigned int level=0; level < triangulation.n_levels(); ++level)
    {
        pcout << Utilities::MPI::sum(locally_active_cells[level], mpi_communicator) << " (" << level << ")" << ", ";
    }
    pcout << std::endl;


    // setup dofs and constraints on refined mesh
    setup_dofs();

    {
        TimerOutput::Scope timer_section(computing_timer, "refine mesh (part 2)");

        // transfer of temperature solution
        {
        LA::Vector  distributed_temperature(temperature_rhs),
                    distributed_old_temperature(temperature_rhs),
                    distributed_old_old_temperature(temperature_rhs);

        std::vector<LA::Vector *> tmp_temperature(3);
        tmp_temperature[0] = &distributed_temperature;
        tmp_temperature[1] = &distributed_old_temperature;
        tmp_temperature[2] = &distributed_old_old_temperature;

        temperature_transfer.interpolate(tmp_temperature);


        temperature_constraints.distribute(distributed_temperature);
        temperature_constraints.distribute(distributed_old_temperature);
        temperature_constraints.distribute(distributed_old_old_temperature);

        temperature_solution            = distributed_temperature;
        old_temperature_solution        = distributed_old_temperature;
        old_old_temperature_solution    = distributed_old_old_temperature;
        }

        // transfer of stokes solution
        {
        LA::BlockVector distributed_solution(navier_stokes_rhs),
                        distributed_old_solution(navier_stokes_rhs),
                        distributed_old_old_solution(navier_stokes_rhs),
                        distributed_phi(navier_stokes_rhs),
                        distributed_old_phi(navier_stokes_rhs),
                        distributed_old_old_phi(navier_stokes_rhs);

        std::vector<LA::BlockVector *>    tmp_stokes(6);
        tmp_stokes[0] = &distributed_solution;
        tmp_stokes[1] = &distributed_old_solution;
        tmp_stokes[2] = &distributed_old_old_solution;
        tmp_stokes[3] = &distributed_phi;
        tmp_stokes[4] = &distributed_old_phi;
        tmp_stokes[5] = &distributed_old_old_phi;

        stokes_transfer.interpolate(tmp_stokes);

        navier_stokes_constraints.distribute(distributed_solution);
        navier_stokes_constraints.distribute(distributed_old_solution);
        navier_stokes_constraints.distribute(distributed_old_old_solution);
        navier_stokes_constraints.distribute(distributed_phi);
        navier_stokes_constraints.distribute(distributed_old_phi);
        navier_stokes_constraints.distribute(distributed_old_old_phi);

        navier_stokes_solution          = distributed_solution;
        old_navier_stokes_solution      = distributed_old_solution;
        old_old_navier_stokes_solution  = distributed_old_old_solution;
        phi_pressure                    = distributed_phi;
        old_phi_pressure                = distributed_old_phi;
        old_old_phi_pressure            = distributed_old_old_phi;
        }
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::run()
{
    make_grid();

    setup_dofs();

    {
        TimerOutput::Scope  timer_section(computing_timer, "compute initialization");

        const EquationData::TemperatureInitialValues<dim>
        initial_temperature(parameters.aspect_ratio,
                            1.0,
                            parameters.temperature_perturbation);

        // initial condition for temperature
        if (parameters.n_initial_refinements == 0)
        {
            LA::Vector  distributed_temperature(temperature_rhs);

            VectorTools::interpolate(mapping,
                                     temperature_dof_handler,
                                     initial_temperature,
                                     distributed_temperature);
            temperature_constraints.distribute(distributed_temperature);

            old_temperature_solution = distributed_temperature;
        }
        else
        {
            unsigned int cnt = 0;
            while (cnt < parameters.n_initial_refinements)
            {
                LA::Vector  distributed_temperature(temperature_rhs);

                VectorTools::interpolate(mapping,
                                         temperature_dof_handler,
                                         initial_temperature,
                                         distributed_temperature);
                temperature_constraints.distribute(distributed_temperature);

                old_temperature_solution = distributed_temperature;

                // copy solution vectors for mesh refinement
                temperature_solution = old_temperature_solution;

                refine_mesh();

                ++cnt;
            }
            pcout << "   Number of cells after "
                  << parameters.n_initial_refinements
                  << " refinements based on the initial condition: "
                  << triangulation.n_global_active_cells()
                  << std::endl;
        }

        // compute consistent initial pressure
        compute_initial_pressure();
        // copy solution vectors for output
        navier_stokes_solution = old_navier_stokes_solution;
    }

    // output of the initial condition
    output_results(true);

    double time = 0;
    double cfl_number = 0;

    do
    {
        pcout << "Step: " << Utilities::int_to_string(timestep_number, 8) << ", "
              << "time: " << time << ", "
              << "time step: " << timestep
              << std::endl;

        // evolve temperature
        temperature_step();

        // evolve velocity and pressure
        navier_stokes_step();

        // compute rms values
        if (timestep_number % parameters.rms_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute rms values");

            const std::pair<double,double> rms_values = compute_rms_values();

            pcout << "   Velocity rms value: "
                  << rms_values.first
                  << std::endl
                  << "   Temperature rms value: "
                  << rms_values.second
                  << std::endl;
        }

        // compute kinetic energy
        if (timestep_number % parameters.energy_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute energies");

            const double kinetic_energy = compute_kinetic_energy();

            pcout << "   Kinetic energy: " << kinetic_energy << std::endl;
        }

        // compute benchmark results
        if (timestep_number >= parameters.benchmark_start &&
                timestep_number % parameters.benchmark_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute benchmark requests");

            update_benchmark_point();

            pcout << "   Benchmark point is at phi: " << phi_benchmark << std::endl;

            std::pair<double,double> benchmark_results
            = compute_benchmark_requests(0.5 * (1. + parameters.aspect_ratio),
                                         0,
                                         phi_benchmark);
            // add values to table
            benchmark_table.add_value("time step", timestep_number);
            benchmark_table.add_value("time", time);
            benchmark_table.add_value("phi", phi_benchmark);
            benchmark_table.add_value("temperature", benchmark_results.first);
            benchmark_table.add_value("azimuthal velocity", benchmark_results.second);
        }

        // compute Courant number
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute cfl number");

            cfl_number = compute_cfl_number();

            if (timestep_number % parameters.cfl_frequency == 0)
                pcout << "   Current cfl number: "
                      << cfl_number
                      << std::endl;
        }

        if (timestep_number % parameters.vtk_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }

        // mesh refinement
        if (timestep_number > 0
                && timestep_number % parameters.refinement_frequency == 0
                && parameters.adaptive_refinement)
            refine_mesh();

        // adjust time step
        if (parameters.adaptive_timestep && timestep_number > parameters.adaptive_timestep_barrier)
            update_timestep(cfl_number);

        // copy temperature solution
        old_old_temperature_solution = old_temperature_solution;
        old_temperature_solution = temperature_solution;

        // extrapolate temperature solution
        {
            LA::Vector  extrapolated_solution(temperature_rhs),
                        old_old_distributed_solution(temperature_rhs);
            extrapolated_solution = old_temperature_solution;
            old_old_distributed_solution = old_old_temperature_solution;

            extrapolated_solution.sadd(1. + timestep / old_timestep,
                                       timestep / old_timestep,
                                       old_old_distributed_solution);

            temperature_solution = extrapolated_solution;
        }

        // copy stokes solution
        old_old_navier_stokes_solution = old_navier_stokes_solution;
        old_navier_stokes_solution = navier_stokes_solution;

        // extrapolate stokes solution
        {
            LA::BlockVector extrapolated_solution(navier_stokes_rhs),
                            old_old_distributed_solution(navier_stokes_rhs);
            extrapolated_solution = old_navier_stokes_solution;
            old_old_distributed_solution = old_old_navier_stokes_solution;

            extrapolated_solution.sadd(1. + timestep / old_timestep,
                                       timestep / old_timestep,
                                       old_old_distributed_solution);

            navier_stokes_solution = extrapolated_solution;
        }

        // copy auxiliary pressure solution
        old_old_phi_pressure = old_phi_pressure;
        old_phi_pressure = phi_pressure;

        // advance in time
        time += timestep;
        ++timestep_number;

    } while (timestep_number < parameters.n_steps && time < parameters.t_final);

    if (parameters.n_steps % parameters.vtk_frequency != 0)
        output_results();

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::ofstream   out_file("benchmark_report.txt");
        benchmark_table.write_text(out_file);
        out_file.close();
    }

    pcout << std::fixed;
}

}  // namespace BouyantFluid

// explicit instantiation
template class BuoyantFluid::BuoyantFluidSolver<2>;
template class BuoyantFluid::BuoyantFluidSolver<3>;
