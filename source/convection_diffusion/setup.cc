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
void ConvectionDiffusionSolver<dim>::set_convection_function
(ConvectionFunction<dim> &function)
{
    convection_function = std::make_shared<ConvectionFunction<dim>>(function);
}

template<int dim>
void ConvectionDiffusionSolver<dim>::setup_dofs()
{
    pcout << "Setup dofs..." << std::endl;

    TimerOutput::Scope(*computing_timer, "setup dofs");

    // temperature part
    locally_owned_dofs.clear();
    locally_relevant_dofs.clear();
    dof_handler.distribute_dofs(fe);

    DoFRenumbering::Cuthill_McKee(dof_handler);

    // extract locally owned and relevant dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

    // constraints
    {
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);

        // constraint matrix for hanging nodes
        DoFTools::make_hanging_node_constraints(
                dof_handler,
                constraints);

        // constraint matrix for periodicity constraints
        for (unsigned int d=0; d<dim; ++d)
            if (boundary_conditions->periodic_bcs[d] !=
                std::pair<types::boundary_id,types::boundary_id>
                (numbers::invalid_boundary_id, numbers::invalid_boundary_id))
            {
                const types::boundary_id first_id = boundary_conditions->periodic_bcs[d].first;
                const types::boundary_id second_id = boundary_conditions->periodic_bcs[d].second;
                AssertThrow(boundary_conditions->dirichlet_bcs.find(first_id) ==
                            boundary_conditions->dirichlet_bcs.end() &&
                            boundary_conditions->dirichlet_bcs.find(second_id) ==
                            boundary_conditions->dirichlet_bcs.end() &&
                            boundary_conditions->neumann_bcs.find(first_id) ==
                            boundary_conditions->neumann_bcs.end() &&
                            boundary_conditions->neumann_bcs.find(second_id) ==
                            boundary_conditions->neumann_bcs.end(),
                            ExcMessage("Cannot mix periodic boundary conditions with "
                                       "other types of boundary conditions on same "
                                       "boundary!"));
                AssertThrow(first_id != second_id,
                            ExcMessage("The two faces for periodic boundary conditions "
                                       "must have different boundary indicators!"));
                DoFTools::make_periodicity_constraints(dof_handler,
                                                       first_id,
                                                       second_id,
                                                       d,
                                                       constraints);
              }


        const Functions::ZeroFunction<dim>  zero_function;
        typename FunctionMap<dim>::type   function_map;

        if (boundary_conditions.get() != 0)
        {
            if (boundary_conditions->dirichlet_bcs.size() != 0)
                for (const auto &it: boundary_conditions->dirichlet_bcs)
                    function_map[it.first] = it.second.get();
            else
            {
                pcout << "   No Dirichlet boundary conditions specified in" << std::endl
                      << "   BC object. Using homogeneous Dirichlet boundary" << std::endl
                      << "   conditions on all boundaries." << std::endl;

                for (const auto &id: triangulation.get_boundary_ids())
                    function_map[id] = &zero_function;
            }
        }
        else
        {
            pcout << "   No BC object passed. Using homogeneous Dirichlet boundary" << std::endl
                  << "   conditions on all boundaries." << std::endl;

            for (const auto &id: triangulation.get_boundary_ids())
                function_map[id] = &zero_function;
        }

        VectorTools::interpolate_boundary_values
        (mapping,
         dof_handler,
         function_map,
         constraints);

        constraints.close();
    }
    // temperature matrix and vector setup
    setup_system_matrix(locally_owned_dofs,
                        locally_relevant_dofs);

    solution.reinit(locally_relevant_dofs,
                                triangulation.get_communicator());
    old_solution.reinit(solution);
    old_old_solution.reinit(solution);

    rhs.reinit(locally_owned_dofs,
                           locally_relevant_dofs,
                           triangulation.get_communicator(),
                           true);
}

template<int dim>
void ConvectionDiffusionSolver<dim>::setup_system_matrix
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

template<int dim>
void ConvectionDiffusionSolver<dim>::setup_problem()
{
    setup_dofs();
}

template<int dim>
void ConvectionDiffusionSolver<dim>::setup_initial_condition
(const Function<dim> &initial_condition)
{
    if (parameters.verbose)
        pcout << "   Setup initial condition..." << std::endl;

    Assert(initial_condition.n_components == 1,
           ExcDimensionMismatch(initial_condition.n_components, 1));

    TimerOutput::Scope(*computing_timer, "setup initial field");

    VectorTools::interpolate(mapping,
                             dof_handler,
                             initial_condition,
                             old_solution);
    constraints.distribute(old_solution);

    // copy initial solution to current solution
    LA::Vector::const_iterator
    old = old_solution.begin(),
    end_old = old_solution.end();

    LA::Vector::iterator
    sol = solution.begin();

    for (; old != end_old; ++old, ++sol)
        *sol = *old;
}
// explicit instantiation
template void ConvectionDiffusionSolver<2>::set_convection_function
(ConvectionFunction<2> &);
template void ConvectionDiffusionSolver<3>::set_convection_function
(ConvectionFunction<3> &);

template void ConvectionDiffusionSolver<2>::setup_dofs();
template void ConvectionDiffusionSolver<3>::setup_dofs();

template void ConvectionDiffusionSolver<2>::setup_system_matrix
(const IndexSet &,
 const IndexSet &);
template void ConvectionDiffusionSolver<3>::setup_system_matrix
(const IndexSet &,
 const IndexSet &);

template void ConvectionDiffusionSolver<2>::setup_problem();
template void ConvectionDiffusionSolver<3>::setup_problem();

template void ConvectionDiffusionSolver<2>::setup_initial_condition(const Function<2> &);
template void ConvectionDiffusionSolver<3>::setup_initial_condition(const Function<3> &);

}  // namespace BuoyantFluid


