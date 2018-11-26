/*
 * make_grid.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include "grid_factory.h"
#include "conducting_fluid_solver.h"

namespace ConductingFluid {

template<int dim>
void ConductingFluidSolver<dim>::make_grid()
{
    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    GridFactory::SphericalShell<dim> spherical_shell(0.35,
                                                     true,
                                                     true,
                                                     3.0);
    spherical_shell.create_coarse_mesh(triangulation);

    const unsigned int n_global_refinements = 2;
    // initial global refinements
    if (n_global_refinements > 0)
    {
        triangulation.refine_global(n_global_refinements);
        std::cout << "      Number of cells after "
                  << n_global_refinements
                  << " global refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    const unsigned int n_interface_refinements = 0;
    // initial boundary refinements
    if (n_interface_refinements > 0)
    {
        for (unsigned int step=0; step<n_interface_refinements; ++step)
        {
            for (auto cell: triangulation.active_cell_iterators())
                if (!cell->at_boundary() &&
                        cell->material_id() == DomainIdentifiers::MaterialIds::Fluid)
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                        if (cell->neighbor(f)->material_id()== DomainIdentifiers::MaterialIds::Vacuum)
                            cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells after "
                  << n_interface_refinements
                  << " interface refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }
}
}  // namespace BuoyantFluid

// explicit instantiation
template void ConductingFluid::ConductingFluidSolver<2>::make_grid();
template void ConductingFluid::ConductingFluidSolver<3>::make_grid();
