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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <buoyant_fluid_solver.h>
#include <initial_values.h>
#include <snapshot_information.h>
#include <postprocessor.h>

namespace BuoyantFluid {

template<int dim>
BuoyantFluidSolver<dim>::BuoyantFluidSolver(Parameters &parameters_)
:
mpi_communicator(MPI_COMM_WORLD),
parameters(parameters_),
// parallel output
pcout(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
// monitor
computing_timer(mpi_communicator, pcout,
                TimerOutput::summary, TimerOutput::wall_times),
// coefficients
equation_coefficients{(parameters.rotation ? 2.0 / parameters.Ek: 0.0),
                      (parameters.rotation ? 1.0 : std::sqrt(parameters.Pr/ parameters.Ra) ),
                      (parameters.rotation ? parameters.Ra / parameters.Pr  : 1.0 ),
                      (parameters.rotation ? 1.0 / parameters.Pr : 1.0 / std::sqrt(parameters.Ra * parameters.Pr)),
                      (parameters.magnetism ? 1. / (parameters.Ek * parameters.Pm): 0.0),
                      (parameters.magnetism ? 1. / parameters.Pm: 0.0)},
// triangulation
triangulation(mpi_communicator),
mapping(4),
// temperature part
temperature_fe(parameters.temperature_degree),
temperature_dof_handler(triangulation),
// stokes part
navier_stokes_fe(FESystem<dim>(FE_Q<dim>(parameters.velocity_degree), dim), 1,
                 FE_Q<dim>(parameters.velocity_degree - 1), 1),
navier_stokes_dof_handler(triangulation),
// magnetic part
magnetic_fe(FESystem<dim>(FE_Q<dim>(parameters.magnetic_degree), dim), 1,
            FE_Q<dim>(parameters.magnetic_degree - 1), 1),
magnetic_dof_handler(triangulation),
// magnetic stabilization
tau{parameters.magnetism ?
        (1. - parameters.aspect_ratio) * (1. - parameters.aspect_ratio) / equation_coefficients[5] : 0.,
    parameters.magnetism ?
        equation_coefficients[5] / (1. - parameters.aspect_ratio) / (1. - parameters.aspect_ratio): 0.},
// time stepping
imex_coefficients(parameters.imex_scheme),
timestep(parameters.initial_timestep),
old_timestep(parameters.initial_timestep),
old_alpha_zero(1.0),
// benchmarking
phi_benchmark(-2.*numbers::PI)
{
    pcout << "Boussinesq solver written by Sebastian Glane\n"
          << "This program solves the Navier-Stokes system incl. thermal convection and magnetic induction.\n"
          << "The stable Taylor-Hood (P2-P1) element and a pressure projection scheme is applied.\n"
          << "For time discretization an adaptive IMEX time stepping is used.\n\n"
          << "The governing equations are\n\n"
          << "\t-- Incompressibility constraint:\n\t\t div(v) = 0,\n\n"
          << "\t-- Navier-Stokes equation:\n\t\tdv/dt + v . grad(v) + C1 Omega x v\n"
          << "\t\t\t\t= - grad(p) + C2 div(grad(v)) - C3 T g + C5 curl(B) x B,\n\n"
          << "\t-- Heat conduction equation:\n\t\tdT/dt + v . grad(T) = C4 div(grad(T)).\n\n"
          << "\t-- Magnetic induction equation:\n\t\tdB/dt = curl(v x B) + C6 div(grad(T)).\n\n"
          << "The coefficients C1 to C4 depend on the normalization as follows.\n\n";

    // generate a nice table of the equation coefficients
    pcout << "+-------------------+----------+---------------+----------+-------------------+---------------+--------+\n"
          << "|       case        |    C1    |      C2       |    C3    |        C4         |      C5       |   C6   |\n"
          << "+-------------------+----------+---------------+----------+-------------------+---------------+--------+\n"
          << "| Non-rotating case |    0     | sqrt(Pr / Ra) |    1     | 1 / sqrt(Ra * Pr) |      0        |   0    |\n"
          << "| Rotating case     |  2 / Ek  |      1        |  Ra / Pr | 1 /  Pr           |      0        |   0    |\n"
          << "| Magnetic case     |  2 / Ek  |      1        |  Ra / Pr | 1 /  Pr           | 1 / (Ek * Pm) | 1 / Pm |  \n"
          << "+-------------------+----------+---------------+----------+-------------------+---------------+--------+\n";

    pcout << std::endl << "You have chosen ";

    std::stringstream ss;

    if (parameters.rotation && !parameters.magnetism)
    {
        rotation_vector[dim-1] = 1.0;

        pcout << "the rotating case with the following parameters: "
              << std::endl;

        ss << "+----------+----------+----------+----------+\n"
           << "|    Ek    |    Ra    |    Pr    |    Pm    |\n"
           << "+----------+----------+----------+----------+\n";

        ss << "| ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ek;
        ss << " | ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ra;
        ss << " | ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pr;
        ss << " | ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pm;
        ss << " | ";
        ss << "\n+----------+----------+----------+----------+\n";
    }
    else if (parameters.rotation && parameters.magnetism)
    {
        rotation_vector[dim-1] = 1.0;

        pcout << "the rotating magnetic case with the following parameters: "
              << std::endl;

        ss << "+----------+----------+----------+\n"
           << "|    Ek    |    Ra    |    Pr    |\n"
           << "+----------+----------+----------+\n";

        ss << "| ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ek;
        ss << " | ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ra;
        ss << " | ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pr;
        ss << " |";
        ss << "\n+----------+----------+----------+\n";
    }
    else
    {
        pcout << "the non-rotating non-magnetic case with the following parameters: "
              << std::endl;

        ss << "+----------+----------+\n"
           << "|    Ra    |    Pr    |\n"
           << "+----------+----------+\n";

        ss << "| ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ra;
        ss << " | ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pr;
        ss << " |";
        ss << "\n+----------+----------+\n";
    }

    ss << "\n";
    ss << "+----------+----------+----------+----------+----------+----------+\n"
       << "|    C1    |    C2    |    C3    |    C4    |    C5    |    C6    |\n"
       << "+----------+----------+----------+----------+----------+----------+\n";

    ss << "| ";

    for (unsigned int n=0; n<6; ++n)
    {
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << equation_coefficients[n];
        ss << " | ";
    }

    ss << "\n+----------+----------+----------+----------+----------+----------+\n";

    pcout << std::endl << ss.str()
          << std::endl << std::fixed << std::flush;

    benchmark_table.declare_column("timestep");
    benchmark_table.declare_column("time");
    benchmark_table.declare_column("phi");
    benchmark_table.declare_column("temperature");
    benchmark_table.declare_column("azimuthal velocity");

    global_avg_table.declare_column("timestep");
    global_avg_table.declare_column("time");
    global_avg_table.declare_column("velocity rms");
    global_avg_table.declare_column("kinetic energy");
    global_avg_table.declare_column("temperature avg");

    if (parameters.magnetism)
    {
        if (dim == 3)
            benchmark_table.declare_column("polar magnetic field");

        global_avg_table.declare_column("magnetic rms");
        global_avg_table.declare_column("magnetic energy");
    }

}

template<int dim>
void BuoyantFluidSolver<dim>::update_timestep(const double current_cfl_number)
{
    TimerOutput::Scope  timer_section(computing_timer, "update time step");

    if (parameters.verbose)
        pcout << "   Updating time step..." << std::endl;

    old_alpha_zero = (timestep_number != 0?
                        imex_coefficients.alpha(timestep/old_timestep)[0]:
                        1.0);

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
            return;
        }
        else if (timestep > parameters.max_timestep
                 && old_timestep == parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
            return;
        }
        else if (timestep < parameters.max_timestep
                 && timestep > parameters.min_timestep)
        {
            timestep_modified = true;
            return;
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
void BuoyantFluidSolver<dim>::create_snapshot(const double time)
{
    TimerOutput::Scope  timer(computing_timer, "create snapshot");

    unsigned int mpi_process_id = Utilities::MPI::this_mpi_process(mpi_communicator);

    // save triangulation and solution vectors:
    {
        std::vector<const LA::BlockVector *>    x_navier_stokes(6);
        x_navier_stokes[0] = &navier_stokes_solution;
        x_navier_stokes[1] = &old_navier_stokes_solution;
        x_navier_stokes[2] = &old_old_navier_stokes_solution;
        x_navier_stokes[3] = &phi_pressure;
        x_navier_stokes[4] = &old_phi_pressure;
        x_navier_stokes[5] = &old_old_phi_pressure;

        parallel::distributed::SolutionTransfer<dim, LA::BlockVector>
        navier_stokes_transfer(navier_stokes_dof_handler);

        navier_stokes_transfer.prepare_serialization(x_navier_stokes);

        std::vector<const LA::Vector *>    x_temperature(3);
        x_temperature[0] = &temperature_solution;
        x_temperature[1] = &old_temperature_solution;
        x_temperature[2] = &old_old_temperature_solution;

        parallel::distributed::SolutionTransfer<dim, LA::Vector>
        temperature_transfer(temperature_dof_handler);

        temperature_transfer.prepare_serialization(x_temperature);

        parallel::distributed::SolutionTransfer<dim, LA::BlockVector>
        magnetic_transfer(magnetic_dof_handler);
        if (parameters.magnetism)
        {
            std::vector<const LA::BlockVector *>    x_magnetic(6);
            x_magnetic[0] = &magnetic_solution;
            x_magnetic[1] = &old_magnetic_solution;
            x_magnetic[2] = &old_old_magnetic_solution;
            x_magnetic[3] = &phi_pseudo_pressure;
            x_magnetic[4] = &old_phi_pseudo_pressure;
            x_magnetic[5] = &old_old_phi_pseudo_pressure;

            magnetic_transfer.prepare_serialization(x_magnetic);
        }

        triangulation.save("restart.mesh");
    }

    // save general information
    if (mpi_process_id == 0)
    {
        std::ofstream   os("restart.snapshot.info");

        Snapshot::SnapshotInformation snapshot_info(timestep_number,
                                                    time,
                                                    timestep,
                                                    old_timestep,
                                                    parameters.magnetism);
        if (parameters.magnetism)
            snapshot_info.set_parameters(parameters.Ek,
                                         parameters.Pr,
                                         parameters.Ra,
                                         parameters.Pm);
        else
            snapshot_info.set_parameters(parameters.Ek,
                                         parameters.Pr,
                                         parameters.Ra);

        Snapshot::save(os, snapshot_info);
    }

    pcout << "Snapshot created on "
          << "Step: " << Utilities::int_to_string(timestep_number, 8)
          << std::endl;
}


template<int dim>
void BuoyantFluidSolver<dim>::resume_from_snapshot()
{
    TimerOutput::Scope  timer(computing_timer, "resume from snapshot");

    // first check existence of the two restart files
    {
        const std::string filename = "restart.mesh";
        std::ifstream   in(filename.c_str());
        if (!in)
            AssertThrow(false,
                        ExcMessage(std::string("You are trying to restart a previous computation, "
                                   "but the restart file <")
                                   +
                                   filename
                                   +
                                   "> does not appear to exist!"));
    }
    {
        const std::string   filename = "restart.snapshot.info";
        std::ifstream   in(filename.c_str());
        if (!in)
            AssertThrow(false,
                        ExcMessage(std::string("You are trying to restart a previous computation, "
                                   "but the restart file <")
                                   +
                                   filename
                                   +
                                   "> does not appear to exist!"));
    }

    pcout << "Resuming from snapshot..." << std::endl;

    try
    {
        triangulation.load("restart.mesh");
    }
    catch (...)
    {
        AssertThrow(false, ExcMessage("Cannot open snapshot mesh file or read the triangulation stored there."));
    }

    setup_dofs();

    // resume navier stokes solution
    {
        LA::BlockVector     distributed_navier_stokes(navier_stokes_rhs);
        LA::BlockVector     old_distributed_navier_stokes(navier_stokes_rhs);
        LA::BlockVector     old_old_distributed_navier_stokes(navier_stokes_rhs);
        LA::BlockVector     distributed_phi_pressure(navier_stokes_rhs);
        LA::BlockVector     old_distributed_phi_pressure(navier_stokes_rhs);
        LA::BlockVector     old_old_distributed_phi_pressure(navier_stokes_rhs);

        std::vector<LA::BlockVector *>  x_navier_stokes(6);
        x_navier_stokes[0] = &distributed_navier_stokes;
        x_navier_stokes[1] = &old_distributed_navier_stokes;
        x_navier_stokes[2] = &old_old_distributed_navier_stokes;
        x_navier_stokes[3] = &distributed_phi_pressure;
        x_navier_stokes[4] = &old_distributed_phi_pressure;
        x_navier_stokes[5] = &old_old_distributed_phi_pressure;

        parallel::distributed::SolutionTransfer<dim, LA::BlockVector>
        navier_stokes_transfer(navier_stokes_dof_handler);

        navier_stokes_transfer.deserialize(x_navier_stokes);
        navier_stokes_solution = distributed_navier_stokes;
        old_navier_stokes_solution= old_distributed_navier_stokes;
        old_old_navier_stokes_solution = old_old_distributed_navier_stokes;
        phi_pressure = distributed_phi_pressure;
        old_phi_pressure = old_distributed_phi_pressure;
        old_old_phi_pressure = old_old_distributed_phi_pressure;
    }
    // resume temperature solution
    {
        LA::Vector     distributed_temperature(temperature_rhs);
        LA::Vector     old_distributed_temperature(temperature_rhs);
        LA::Vector     old_old_distributed_temperature(temperature_rhs);

        std::vector<LA::Vector *>   x_temperature(3);
        x_temperature[0] = &distributed_temperature;
        x_temperature[1] = &old_distributed_temperature;
        x_temperature[2] = &old_old_distributed_temperature;

        parallel::distributed::SolutionTransfer<dim, LA::Vector>
        temperature_transfer(temperature_dof_handler);

        temperature_transfer.deserialize(x_temperature);

        temperature_solution = distributed_temperature;
        old_temperature_solution = old_distributed_temperature;
        old_old_temperature_solution= old_old_distributed_temperature;
    }
    // resume magnetic solution
    if (parameters.magnetism)
    {
        LA::BlockVector     distributed_magnetic(magnetic_rhs);
        LA::BlockVector     old_distributed_magnetic(magnetic_rhs);
        LA::BlockVector     old_old_distributed_magnetic(magnetic_rhs);
        LA::BlockVector     distributed_phi_pseudo_pressure(magnetic_rhs);
        LA::BlockVector     old_distributed_phi_pseudo_pressure(magnetic_rhs);
        LA::BlockVector     old_old_distributed_phi_pseudo_pressure(magnetic_rhs);

        std::vector<LA::BlockVector *>  x_magnetic(6);
        x_magnetic[0] = &distributed_magnetic;
        x_magnetic[1] = &old_distributed_magnetic;
        x_magnetic[2] = &old_old_distributed_magnetic;
        x_magnetic[3] = &distributed_phi_pseudo_pressure;
        x_magnetic[4] = &old_distributed_phi_pseudo_pressure;
        x_magnetic[5] = &old_old_distributed_phi_pseudo_pressure;

        parallel::distributed::SolutionTransfer<dim, LA::BlockVector>
        magnetic_transfer(magnetic_dof_handler);

        magnetic_transfer.deserialize(x_magnetic);
        magnetic_solution = distributed_magnetic;
        old_magnetic_solution = old_distributed_magnetic;
        old_old_magnetic_solution = old_old_distributed_magnetic;
        phi_pseudo_pressure= distributed_phi_pseudo_pressure;
        old_phi_pseudo_pressure = old_distributed_phi_pseudo_pressure;
        old_old_phi_pseudo_pressure = old_old_distributed_phi_pseudo_pressure;
    }

    try
    {
        std::ifstream   is("restart.snapshot.info");

        Snapshot::SnapshotInformation   snapshot_info;

        Snapshot::load(is, snapshot_info);

        snapshot_info.print(pcout);

        timestep_number = snapshot_info.timestep_number();
        if (parameters.adaptive_timestep)
            timestep = snapshot_info.timestep();
        old_timestep = snapshot_info.old_timestep();

        Assert(timestep > 0, ExcLowerRangeType<double>(timestep, 0));
        Assert(old_timestep > 0, ExcLowerRangeType<double>(old_timestep, 0));
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception on processing: " << std::endl
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
        std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::abort();
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::refine_mesh()
{
    pcout << "Mesh refinement..." << std::endl;

    parallel::distributed::SolutionTransfer<dim,LA::Vector>
    temperature_transfer(temperature_dof_handler);

    parallel::distributed::SolutionTransfer<dim,LA::BlockVector>
    stokes_transfer(navier_stokes_dof_handler);

    parallel::distributed::SolutionTransfer<dim,LA::BlockVector>
    magnetic_transfer(magnetic_dof_handler);

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
                                                      0.3, 0.3);

    // clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > parameters.n_max_levels)
        for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_max_levels))
            cell->clear_refine_flag();

    // clear coarsen flags if level decreases minimum
    for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_min_levels))
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

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();

    // preparing temperature solution transfer
    std::vector<const LA::Vector *> x_temperature(3);
    x_temperature[0] = &temperature_solution;
    x_temperature[1] = &old_temperature_solution;
    x_temperature[2] = &old_old_temperature_solution;
    temperature_transfer.prepare_for_coarsening_and_refinement(x_temperature);

    // preparing stokes solution transfer
    std::vector<const LA::BlockVector *> x_stokes(6);
    x_stokes[0] = &navier_stokes_solution;
    x_stokes[1] = &old_navier_stokes_solution;
    x_stokes[2] = &old_old_navier_stokes_solution;
    x_stokes[3] = &phi_pressure;
    x_stokes[4] = &old_phi_pressure;
    x_stokes[5] = &old_old_phi_pressure;
    stokes_transfer.prepare_for_coarsening_and_refinement(x_stokes);

    if (parameters.magnetism)
    {
        // preparing temperature solution transfer
        std::vector<const LA::BlockVector *> x_magnetic(6);
        x_magnetic[0] = &magnetic_solution;
        x_magnetic[1] = &old_magnetic_solution;
        x_magnetic[2] = &old_old_magnetic_solution;
        x_magnetic[3] = &phi_pseudo_pressure;
        x_magnetic[4] = &old_phi_pseudo_pressure;
        x_magnetic[5] = &old_old_phi_pseudo_pressure;

        magnetic_transfer.prepare_serialization(x_magnetic);
    }

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();
    }

    std::vector<unsigned int>   locally_active_cells(triangulation.n_global_levels());
    for (unsigned int level=parameters.n_global_refinements; level < triangulation.n_levels(); ++level)
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

        // transfer of magnetic solution
        if (parameters.magnetism)
        {
            LA::BlockVector     distributed_magnetic(magnetic_rhs);
            LA::BlockVector     old_distributed_magnetic(magnetic_rhs);
            LA::BlockVector     old_old_distributed_magnetic(magnetic_rhs);
            LA::BlockVector     distributed_phi_pseudo_pressure(magnetic_rhs);
            LA::BlockVector     old_distributed_phi_pseudo_pressure(magnetic_rhs);
            LA::BlockVector     old_old_distributed_phi_pseudo_pressure(magnetic_rhs);

            std::vector<LA::BlockVector *>  tmp_magnetic(6);
            tmp_magnetic[0] = &distributed_magnetic;
            tmp_magnetic[1] = &old_distributed_magnetic;
            tmp_magnetic[2] = &old_old_distributed_magnetic;
            tmp_magnetic[3] = &distributed_phi_pseudo_pressure;
            tmp_magnetic[4] = &old_distributed_phi_pseudo_pressure;
            tmp_magnetic[5] = &old_old_distributed_phi_pseudo_pressure;

            magnetic_transfer.interpolate(tmp_magnetic);

            magnetic_constraints.distribute(distributed_magnetic);
            magnetic_constraints.distribute(old_distributed_magnetic);
            magnetic_constraints.distribute(old_old_distributed_magnetic);
            magnetic_constraints.distribute(distributed_phi_pseudo_pressure);
            magnetic_constraints.distribute(old_distributed_phi_pseudo_pressure);
            magnetic_constraints.distribute(old_old_distributed_phi_pseudo_pressure);

            magnetic_solution = distributed_magnetic;
            old_magnetic_solution = old_distributed_magnetic;
            old_old_magnetic_solution = old_old_distributed_magnetic;
            phi_pseudo_pressure = distributed_phi_pseudo_pressure;
            old_phi_pseudo_pressure = old_distributed_phi_pseudo_pressure;
            old_old_phi_pseudo_pressure = old_distributed_phi_pseudo_pressure;
        }

    }
}

template<int dim>
void BuoyantFluidSolver<dim>::run()
{
    if (parameters.resume_from_snapshot == false )
    {
    make_grid();

    setup_dofs();

    {
        TimerOutput::Scope  timer_section(computing_timer, "compute initialization");

        // initial conditions
        if (parameters.n_initial_refinements == 0)
        {
            // project initial conditions
            project_temperature_field();

            if (parameters.magnetism)
                project_magnetic_field();
        }
        else
        {
            unsigned int cnt = 0;
            while (cnt < parameters.n_initial_refinements)
            {
                // project initial conditions
                project_temperature_field();

                if (parameters.magnetism)
                    project_magnetic_field();

                // refine the mesh
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
    }
    }
    else
    {
        make_coarse_grid();

        resume_from_snapshot();
    }

    // output of the initial condition
    output_results(true);

    // compute rms values
    {
        TimerOutput::Scope  timer_section(computing_timer, "compute global averages values");

        const std::vector<double> global_avg = compute_global_averages();

        pcout << "   Velocity rms value: "
              << global_avg[0]
              << std::endl
              << "   Temperature globally average value: "
              << global_avg[2]
              << std::endl;
        if (parameters.magnetism)
        {
            AssertDimension(global_avg.size(), 5);

            pcout << "   Magnetic field rms value: "
                  << global_avg[3]
                  << std::endl;
        }

        pcout << "   Kinetic energy: " << global_avg[1] << std::endl;
        if (parameters.magnetism)
        {
            AssertDimension(global_avg.size(), 5);

            pcout << "   Magnetic energy: "
                  << global_avg[4]
                  << std::endl;
        }

        // add values to table
        global_avg_table.add_value("timestep", timestep_number);
        global_avg_table.add_value("time", 0.0);
        global_avg_table.add_value("velocity rms", global_avg[0]);
        global_avg_table.add_value("kinetic energy", global_avg[1]);
        global_avg_table.add_value("temperature avg", global_avg[2]);
        if (parameters.magnetism)
        {
            global_avg_table.add_value("magnetic rms", global_avg[3]);
            global_avg_table.add_value("magnetic energy", global_avg[4]);
        }
    }

    double  time = 0;
    double  cfl_number = 0;

    const unsigned int inital_timestep_number{timestep_number};

    bool    mesh_refined = false;

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

        if (parameters.magnetism)
            magnetic_step();

        // compute rms values
        if (timestep_number % parameters.global_avg_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute global averages values");

            const std::vector<double> global_avg = compute_global_averages();

            pcout << "   Velocity rms value: "
                  << global_avg[0]
                  << std::endl
                  << "   Temperature globally averaged value: "
                  << global_avg[2]
                  << std::endl;
            if (parameters.magnetism)
            {
                AssertDimension(global_avg.size(), 5);

                pcout << "   Magnetic field rms value: "
                      << global_avg[3]
                      << std::endl;
            }

            pcout << "   Kinetic energy: " << global_avg[1] << std::endl;
            if (parameters.magnetism)
            {
                AssertDimension(global_avg.size(), 5);

                pcout << "   Magnetic energy: "
                      << global_avg[4]
                      << std::endl;
            }

            // add values to table
            global_avg_table.add_value("timestep", timestep_number);
            global_avg_table.add_value("time", time);
            global_avg_table.add_value("velocity rms", global_avg[0]);
            global_avg_table.add_value("kinetic energy", global_avg[1]);
            global_avg_table.add_value("temperature avg", global_avg[2]);
            if (parameters.magnetism)
            {
                global_avg_table.add_value("magnetic rms", global_avg[3]);
                global_avg_table.add_value("magnetic energy", global_avg[4]);
            }
        }

        // compute benchmark results
        if (timestep_number >= parameters.benchmark_start &&
                timestep_number % parameters.benchmark_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute benchmark requests");

            update_benchmark_point();

            const double radial_velocity
            = compute_radial_velocity(0.5 * (1. + parameters.aspect_ratio),
                                      phi_benchmark,
                                      numbers::PI / 2.);

            pcout << "   Benchmark point at  phi = " << phi_benchmark
                  << " (x = " <<  0.5 * (1. + parameters.aspect_ratio) * cos(phi_benchmark)
                  << ", y = " <<  0.5 * (1. + parameters.aspect_ratio) * sin(phi_benchmark)
                  << ")"
                  << ", radial velocity =  "
                  << radial_velocity
                  << std::endl;

            std::vector<double> benchmark_results
            = compute_benchmark_requests();

            pcout << "   Benchmark requests:  T = " << benchmark_results[0]
                  << ", v_phi = " << benchmark_results[1];
            if (parameters.magnetism && dim == 3)
                pcout << ", B_theta = " << benchmark_results[2];
            pcout << std::endl;

            // add values to table
            benchmark_table.add_value("timestep", timestep_number);
            benchmark_table.add_value("time", time);
            benchmark_table.add_value("phi", phi_benchmark);
            benchmark_table.add_value("temperature", benchmark_results[0]);
            benchmark_table.add_value("azimuthal velocity", benchmark_results[1]);
            if (parameters.magnetism && dim == 3)
                benchmark_table.add_value("polar magnetic field", benchmark_results[2]);
        }

        // write vtk output
        if (timestep_number % parameters.vtk_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }

        // mesh refinement
        if (timestep_number > 0
                && timestep_number % parameters.refinement_frequency == 0
                && parameters.adaptive_refinement)
        {
            refine_mesh();
            mesh_refined = true;
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
        // adjust time step
        if (parameters.adaptive_timestep)
        {
            if (mesh_refined && cfl_number > parameters.cfl_max)
            {
                update_timestep(cfl_number);
                mesh_refined = false;
            }
            else if ((timestep_number - inital_timestep_number) > parameters.adaptive_timestep_barrier)
                update_timestep(cfl_number);
        }
        /*
         * If computation is resumed from a snapshot and non-adaptive
         * timestepping is done, the old_timestep of the resumed computation
         * is used in the first iteration. The old_timestep needs to be
         * updated after the first newly computed timestep with the equidistant
         * timestep.
         */
        else if (!parameters.adaptive_timestep && parameters.resume_from_snapshot)
        {
            if ((timestep_number - inital_timestep_number) == 0)
            {
                old_timestep = timestep;
                timestep_modified = true;
            }
            else if (timestep_number - inital_timestep_number == 1)
                timestep_modified = false;
        }

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

        // copy navier stokes solution
        old_old_navier_stokes_solution = old_navier_stokes_solution;
        old_navier_stokes_solution = navier_stokes_solution;

        // extrapolate navier stokes solution
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

        /*
         * Advance in time with old_time because the timestep may be updated
         * by the method update_timestep earlier.
         */
        time += old_timestep;

        // write snapshot
        if (timestep_number > 0 &&
            timestep_number % parameters.snapshot_frequency == 0)
        {
            create_snapshot(time);

            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
                std::ofstream   out_file("benchmark_report.txt");
                benchmark_table.write_text(out_file);
                out_file.close();
            }

            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
                std::ofstream   out_file("global_avg_report.txt");
                global_avg_table.write_text(out_file);
                out_file.close();
            }
        }

        // increase timestep number
        ++timestep_number;

    } while (timestep_number < parameters.n_steps + 1 && time < parameters.t_final);

    timestep_number -= 1;

    if (parameters.n_steps % parameters.vtk_frequency != 0)
        output_results();

    if (parameters.n_steps % parameters.snapshot_frequency != 0)
        create_snapshot(time);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::ofstream   out_file("benchmark_report.txt");
        benchmark_table.write_text(out_file);
        out_file.close();
    }

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::ofstream   out_file("global_avg_report.txt");
        global_avg_table.write_text(out_file);
        out_file.close();
    }

    pcout << std::fixed;
}

}  // namespace BouyantFluid

// explicit instantiation
template class BuoyantFluid::BuoyantFluidSolver<2>;
template class BuoyantFluid::BuoyantFluidSolver<3>;
