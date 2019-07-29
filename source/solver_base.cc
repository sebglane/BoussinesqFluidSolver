/*
 * solver_base.cc
 *
 *  Created on: Jul 27, 2019
 *      Author: sg
 */

#include <adsolic/solver_base.h>

namespace adsolic
{

template<int dim, class VectorType>
SolverBase<dim,VectorType>::SolverBase
(const parallel::distributed::Triangulation<dim>  &triangulation_in,
 const MappingQ<dim>    &mapping_in,
 const IMEXTimeStepping &timestepper_in,
 const std::shared_ptr<TimerOutput> external_timer)
:
triangulation(triangulation_in),
mapping(mapping_in),
timestepper(timestepper_in),
pcout(std::cout,
      Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0),
dof_handler(triangulation)
{
    if (external_timer.get() != 0)
        computing_timer  = external_timer;
    else
        computing_timer.reset(new TimerOutput(pcout,
                                              TimerOutput::summary,
                                              TimerOutput::wall_times));
}

template<int dim, class VectorType>
types::global_dof_index
SolverBase<dim,VectorType>::n_dofs() const
{
    return dof_handler.n_dofs();
}

template<int dim, class VectorType>
const DoFHandler<dim> &
SolverBase<dim,VectorType>::get_dof_handler() const
{
    return dof_handler;
}

template<int dim, class VectorType>
const VectorType &
SolverBase<dim,VectorType>::get_solution() const
{
    return solution;
}

template<int dim, class VectorType>
std::vector<const VectorType*>
SolverBase<dim,VectorType>::get_old_solutions() const
{
    std::vector<const VectorType*> old_solutions;
    old_solutions.push_back(&old_solution);
    old_solutions.push_back(&old_old_solution);

    return old_solutions;
}

template<int dim, class VectorType>
void
SolverBase<dim,VectorType>::extrapolate_solution()
{
    typename VectorType::iterator
    sol = solution.begin(),
    end_sol = solution.end();
    typename VectorType::const_iterator
    old_sol = old_solution.begin(),
    old_old_sol = old_old_solution.begin();

    // extrapolate solution from old states
    for (; sol!=end_sol; ++sol, ++old_sol, ++old_old_sol)
        *sol = timestepper.extrapolate(*old_sol, *old_old_sol);
}

template<int dim, class VectorType>
void
SolverBase<dim,VectorType>::advance_solution()
{
    typename VectorType::const_iterator
    sol = solution.begin(),
    end_sol = solution.end();

    typename VectorType::iterator
    old_sol = old_solution.begin(),
    old_old_sol = old_old_solution.begin();

    // copy solutions
    for (; sol!=end_sol; ++sol, ++old_sol, ++old_old_sol)
    {
        *old_old_sol = *old_sol;
        *old_sol = *sol;
    }
}

// explicit instantiations
template class SolverBase<2,LA::Vector>;
template class SolverBase<3,LA::Vector>;

template class SolverBase<2,LA::BlockVector>;
template class SolverBase<3,LA::BlockVector>;

}  // namespace adsolic


