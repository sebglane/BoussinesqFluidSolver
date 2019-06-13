/*
 * make_grid.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */


#include "buoyant_fluid_solver.h"
#include "grid_factory.h"

namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::make_grid()
{
    TimerOutput::Scope timer_section(computing_timer, "make grid");

    pcout << "Making grid..." << std::endl;

    make_coarse_grid();

    // initial global refinements
    if (parameters.n_global_refinements > 0)
    {
        triangulation.refine_global(parameters.n_global_refinements);
        pcout << "   Number of cells after "
              << parameters.n_global_refinements
              << " global refinements: "
              << triangulation.n_global_active_cells()
              << std::endl;
    }

    // initial boundary refinements
    if (parameters.n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<parameters.n_boundary_refinements; ++step)
        {
            for (auto cell: triangulation.active_cell_iterators())
                if (cell->is_locally_owned() && cell->at_boundary())
                    cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
        }
        pcout << "   Number of cells after "
              << parameters.n_boundary_refinements
              << " boundary refinements: "
              << triangulation.n_global_active_cells()
              << std::endl;
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::make_coarse_grid()
{
    GridFactory::SphericalShell<dim> spherical_shell(parameters.aspect_ratio);
    spherical_shell.create_coarse_mesh(triangulation);
}
}  // namespace BuoyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::make_grid();
template void BuoyantFluid::BuoyantFluidSolver<3>::make_grid();

template void BuoyantFluid::BuoyantFluidSolver<2>::make_coarse_grid();
template void BuoyantFluid::BuoyantFluidSolver<3>::make_coarse_grid();
