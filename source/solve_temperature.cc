/*
 * solve_temperature.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include "buoyant_fluid_solver.h"

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::temperature_step()
{
    std::cout << "   Temperature step..." << std::endl;

    // assemble right-hand side (and system if necessary)
    assemble_temperature_system();

    // rebuild preconditioner for diffusion step
    build_temperature_preconditioner();

    // solve projection step
    solve_temperature_system();
}

template<int dim>
void BuoyantFluidSolver<dim>::build_temperature_preconditioner()
{
    if (!rebuild_temperature_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build temperature preconditioner");

    preconditioner_temperature.reset(new PreconditionerTypeTemperature());

    PreconditionerTypeTemperature::AdditionalData     data;
    data.relaxation = 0.6;

    preconditioner_temperature->initialize(temperature_matrix,
                                 data);

    rebuild_temperature_preconditioner = false;
}


template <int dim>
void BuoyantFluidSolver<dim>::solve_temperature_system()
{
    std::cout << "      Solving temperature system..." << std::endl;
    TimerOutput::Scope  timer_section(computing_timer, "temperature solve");

    // solve linear system
    SolverControl solver_control(30, 1e-6 * temperature_rhs.l2_norm());

    SolverCG<>   cg(solver_control);

    temperature_constraints.set_zero(temperature_solution);

    cg.solve(temperature_matrix,
             temperature_solution,
             temperature_rhs,
             *preconditioner_temperature);

    temperature_constraints.distribute(temperature_solution);

    // write info message
    std::cout << "      "
            << solver_control.last_step()
            << " CG iterations for temperature"
            << std::endl;

}
}  // namespace BouyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::temperature_step();
template void BuoyantFluid::BuoyantFluidSolver<3>::temperature_step();

template void BuoyantFluid::BuoyantFluidSolver<2>::build_temperature_preconditioner();
template void BuoyantFluid::BuoyantFluidSolver<3>::build_temperature_preconditioner();

template void BuoyantFluid::BuoyantFluidSolver<2>::solve_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::solve_temperature_system();
