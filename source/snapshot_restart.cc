/*
 * snapshot_restart.cc
 *
 *  Created on: Jul 17, 2019
 *      Author: sg
 */
#include <deal.II/distributed/solution_transfer.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <buoyant_fluid_solver.h>
#include <snapshot_information.h>

namespace BuoyantFluid
{
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

template void BuoyantFluidSolver<2>::resume_from_snapshot();
template void BuoyantFluidSolver<3>::resume_from_snapshot();

template void BuoyantFluidSolver<2>::create_snapshot(const double );
template void BuoyantFluidSolver<3>::create_snapshot(const double );
}  // namespace BuoyantFluid

