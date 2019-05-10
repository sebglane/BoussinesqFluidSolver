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

    GridFactory::SphericalShell<dim> spherical_shell(parameters.aspect_ratio);
    spherical_shell.create_coarse_mesh(triangulation);

    // initial global refinements
    if (parameters.n_global_refinements > 0)
    {
        triangulation.refine_global(parameters.n_global_refinements);
        pcout << "   Number of cells after "
              << parameters.n_global_refinements
              << " global refinements: "
              << triangulation.n_active_cells()
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
              << triangulation.n_active_cells()
              << std::endl;
    }

    mark_sector_cells(0.5 * (1.0 + parameters.aspect_ratio), 0.);
}

template<>
void BuoyantFluidSolver<2>::mark_sector_cells(double radius, double /* theta */)
{
    Assert(radius >= 0.0, ExcLowerRangeType<double>(0.0, radius));

    const unsigned int dim = 2;

    Point<dim>  cell_midpoint;
    Point<dim>  test_point;

    for (auto cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell_midpoint = cell->center();

            const double phi = atan2(cell_midpoint[1], cell_midpoint[0]);
            Assert(phi >= -numbers::PI, ExcLowerRangeType<double>(phi, -numbers::PI));
            Assert(phi <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, phi));

            test_point[0] = radius * cos(phi);
            test_point[1] = radius * sin(phi);

            if (cell->point_inside(test_point))
                cell->set_material_id(GridFactory::MaterialIds::BenchmarkSector);
        }
}

template<>
void BuoyantFluidSolver<3>::mark_sector_cells(double radius, double theta)
{
    Assert(radius >= 0.0, ExcLowerRangeType<double>(0.0, radius));

    Assert(theta >= 0., ExcLowerRangeType<double>(theta, 0.));
    Assert(theta <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, theta));

    const unsigned int dim = 3;

    Point<dim>  cell_midpoint;
    Point<dim>  test_point;

    for (auto cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell_midpoint = cell->center();

            const double phi = atan2(cell_midpoint[1], cell_midpoint[0]);
            Assert(phi >= -numbers::PI, ExcLowerRangeType<double>(phi, -numbers::PI));
            Assert(phi <= numbers::PI, ExcLowerRangeType<double>(numbers::PI, phi));

            test_point[0] = radius * cos(phi) * sin(theta);
            test_point[1] = radius * sin(phi) * sin(theta);
            test_point[2] = radius * cos(theta);

            if (cell->point_inside(test_point))
            {
                cell->set_material_id(GridFactory::MaterialIds::BenchmarkSector);
                std::cout << "fffff: " << GridFactory::MaterialIds::BenchmarkSector
                        << std::endl;
            }
        }
}
}  // namespace BuoyantFluid

// explicit instantiation
template void BuoyantFluid::BuoyantFluidSolver<2>::make_grid();
template void BuoyantFluid::BuoyantFluidSolver<3>::make_grid();

template void BuoyantFluid::BuoyantFluidSolver<2>::mark_sector_cells(double, double);
template void BuoyantFluid::BuoyantFluidSolver<3>::mark_sector_cells(double, double);
