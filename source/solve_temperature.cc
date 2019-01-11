/*
 * solve_temperature.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include "buoyant_fluid_solver.h"

namespace BuoyantFluid {

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


template <int dim>
void BuoyantFluidSolver<dim>::solve_temperature_system()
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
}  // namespace BouyantFluid




