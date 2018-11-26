/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "conducting_fluid_solver.h"
#include "initial_values.h"
#include "postprocessor.h"

namespace ConductingFluid {

template<int dim>
ConductingFluidSolver<dim>::ConductingFluidSolver(
        const double       &timestep,
        const unsigned int &n_steps,
        const unsigned int &output_frequency,
        const unsigned int &refinement_frequency)
:
magnetic_degree(1),
imex_coefficients(TimeStepping::IMEXType::CNAB),
triangulation(),
mapping(4),
// magnetic part
interior_magnetic_fe(FE_Nedelec<dim>(magnetic_degree), 1,
                     FE_Nothing<dim>(), 1),
exterior_magnetic_fe(FE_Nothing<dim>(dim), 1,
                     FE_Q<dim>(magnetic_degree), 1),
magnetic_dof_handler(triangulation),
// coefficients
equation_coefficients{1.0},
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
// time stepping
timestep(timestep),
old_timestep(timestep),
// TODO: goes to parameter file later
n_steps(n_steps),
output_frequency(output_frequency),
refinement_frequency(refinement_frequency)
{
    fe_collection.push_back(interior_magnetic_fe);
    fe_collection.push_back(exterior_magnetic_fe);
}

/*
 *
template<int dim>
void ConductingFluidSolver<dim>::output_results() const
{
    std::cout << "   Output results..." << std::endl;

    // create post processor
    PostProcessor<dim>   postprocessor;

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(magnetic_dof_handler);
    data_out.add_data_vector(magnetic_solution, postprocessor);
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(timestep_number, 5) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

template<int dim>
void ConductingFluidSolver<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    triangulation.set_all_refine_flags();

    // preparing magnetic solution transfer
    std::vector<BlockVector<double>> x_magnetic(3);
    x_magnetic[0] = magnetic_solution;
    x_magnetic[1] = old_magnetic_solution;
    x_magnetic[2] = old_old_magnetic_solution;
    SolutionTransfer<dim,Vector<double>> magnetic_transfer(magnetic_dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    magnetic_transfer.prepare_for_coarsening_and_refinement(x_magnetic);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of magnetic solution
    {
        std::vector<BlockVector<double>>    tmp_magnetic(3);
        tmp_magnetic[0].reinit(magnetic_solution);
        tmp_magnetic[1].reinit(magnetic_solution);
        tmp_magnetic[2].reinit(magnetic_solution);
        magnetic_transfer.interpolate(x_magnetic, tmp_magnetic);

        magnetic_solution = tmp_magnetic[0];
        old_magnetic_solution = tmp_magnetic[1];
        old_old_magnetic_solution = tmp_magnetic[2];

        magnetic_constraints.distribute(magnetic_solution);
        magnetic_constraints.distribute(old_magnetic_solution);
        magnetic_constraints.distribute(old_old_magnetic_solution);
    }
    // set rebuild flags
    rebuild_magnetic_matrices = true;
}
 *
 */

template <int dim>
void ConductingFluidSolver<dim>::solve()
{
    {
        std::cout << "   Solving magnetic system..." << std::endl;

        TimerOutput::Scope  timer_section(computing_timer, "stokes solve");

        magnetic_constraints.set_zero(magnetic_solution);

        PrimitiveVectorMemory<BlockVector<double>> vector_memory;

        SolverControl solver_control(1000,
                                     1e-6 * magnetic_rhs.l2_norm());

        SolverGMRES<BlockVector<double>>
        solver(solver_control,
               vector_memory,
               SolverGMRES<BlockVector<double>>::AdditionalData(30, true));

        PreconditionSSOR<BlockSparseMatrix<double>> preconditioner;
        preconditioner.initialize(magnetic_matrix,
                                  PreconditionSSOR<BlockSparseMatrix<double>>::AdditionalData());

        solver.solve(magnetic_matrix,
                     magnetic_solution,
                     magnetic_rhs,
                     preconditioner);

        std::cout << "      "
                  << solver_control.last_step()
                  << " GMRES iterations for stokes system, "
                  << std::endl;

        magnetic_constraints.distribute(magnetic_solution);
    }
}



template<int dim>
void ConductingFluidSolver<dim>::run()
{
    make_grid();

    setup_dofs();

    assemble_magnetic_system();
    /*
     *
    const EquationData::MagneticInitialValues<dim> initial_potential();

    VectorTools::interpolate(mapping,
                             magnetic_dof_handler,
                             initial_potential,
                             old_magnetic_solution_solution);


    magnetic_constraints.distribute(old_magnetic_solution);

    magnetic_solution = old_magnetic_solution;

    output_results();

    double time = 0;
    double cfl_number = 0;

    do
    {
        std::cout << "step: " << Utilities::int_to_string(timestep_number, 8) << ", "
                  << "time: " << time << ", "
                  << "time step: " << timestep
                  << std::endl;

        assemble_magnetic_system();

        solve();
        if (timestep_number % output_frequency == 0 && timestep_number != 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }
        // mesh refinement
        if ((timestep_number > 0) && (timestep_number % refinement_frequency == 0))
            refine_mesh();

        // extrapolate magnetic solution
        old_old_magnetic_solution = old_magnetic_solution;
        old_magnetic_solution = magnetic_solution;

        // extrapolate magnetic solution
        magnetic_solution.sadd(1. + timestep / old_timestep,
                             timestep / old_timestep,
                             old_old_magnetic_solution);
        // advance in time
        time += timestep;
        ++timestep_number;

    } while (timestep_number <= n_steps);

    if (parameters.n_steps % output_frequency != 0)
        output_results();
     *
     */


    std::cout << std::fixed;

    computing_timer.print_summary();
    computing_timer.reset();

    std::cout << std::endl;
}
}  // namespace ConductingFluid

// explicit instantiation
template class ConductingFluid::ConductingFluidSolver<3>;
