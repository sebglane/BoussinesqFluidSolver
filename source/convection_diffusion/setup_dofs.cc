/*
 * setup.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */
#include <adsolic/convection_diffusion_solver.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/vector_tools.h>


namespace adsolic {

template<int dim>
void ConvectionDiffusionSolver<dim>::setup_dofs()
{
    pcout << "Setup dofs..." << std::endl;

    computing_timer->enter_subsection("setup temperature dofs");

    // temperature part
    locally_owned_dofs.clear();
    locally_relevant_dofs.clear();
    dof_handler.distribute_dofs(fe);

    DoFRenumbering::Cuthill_McKee(dof_handler);

    // extract locally owned and relevant dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

    // temperature constraints
    {
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);

        DoFTools::make_hanging_node_constraints(
                dof_handler,
                constraints);

        /*
         *
        const Functions::ConstantFunction<dim>  icb_temperature(1.0);
        const Functions::ConstantFunction<dim>  cmb_temperature(0.0);

        const std::map<typename types::boundary_id, const Function<dim>*>
        temperature_boundary_values = {{GridFactory::BoundaryIds::ICB, &icb_temperature},
                                       {GridFactory::BoundaryIds::CMB, &cmb_temperature}};

        VectorTools::interpolate_boundary_values(temperature_dof_handler,
                                                 temperature_boundary_values,
                                                 temperature_constraints);
         *
         */

        constraints.close();
    }
    // temperature matrix and vector setup
    setup_matrix(locally_owned_dofs,
                             locally_relevant_dofs);

    solution.reinit(locally_relevant_dofs,
                                triangulation.get_communicator());
    old_solution.reinit(solution);
    old_old_solution.reinit(solution);

    rhs.reinit(locally_owned_dofs,
                           locally_relevant_dofs,
                           triangulation.get_communicator(),
                           true);

    // count dofs
    const types::global_dof_index n_dofs_temperature
    = dof_handler.n_dofs();

    computing_timer->leave_subsection();

    // print info message
    pcout << "   Number of active cells: "
          << triangulation.n_global_active_cells()
          << std::endl
          << "   Number of temperature degrees of freedom: "
          << n_dofs_temperature
          << std::endl;
}

template<int dim>
void ConvectionDiffusionSolver<dim>::setup_matrix
(const IndexSet &locally_owned_dofs,
 const IndexSet &locally_relevant_dofs)
{
    preconditioner.reset();

    system_matrix.clear();
    mass_matrix.clear();
    stiffness_matrix.clear();

    LA::DynamicSparsityPattern  dsp(locally_owned_dofs,
                                    locally_owned_dofs,
                                    locally_relevant_dofs,
                                    triangulation.get_communicator());

    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(triangulation.get_communicator()));

    dsp.compress();

    system_matrix.reinit(dsp);
    mass_matrix.reinit(dsp);
    stiffness_matrix.reinit(dsp);

    rebuild_matrices = true;
}

// explicit instantiation
template void ConvectionDiffusionSolver<2>::setup_dofs();
template void ConvectionDiffusionSolver<3>::setup_dofs();

template void ConvectionDiffusionSolver<2>::setup_matrix
(const IndexSet &,
 const IndexSet &);
template void ConvectionDiffusionSolver<3>::setup_matrix
(const IndexSet &,
 const IndexSet &);

}  // namespace BuoyantFluid


