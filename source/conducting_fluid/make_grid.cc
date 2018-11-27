/*
 * make_grid.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/grid/grid_out.h>

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

    const Point<dim> center;

    for (auto cell: triangulation.active_cell_iterators())
        if (cell->center().distance(center) < 1.0)
            cell->set_material_id(DomainIdentifiers::MaterialIds::Fluid);

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

    // check normal vectors
    const QMidpoint<dim-1> face_quadrature;

    FEFaceValues<dim> fe_face_values(interior_magnetic_fe,
                                     face_quadrature,
                                     update_normal_vectors|
                                     update_quadrature_points);

    for (auto cell: triangulation.active_cell_iterators())
    {
        if (!cell->at_boundary() &&
                cell->material_id() == DomainIdentifiers::MaterialIds::Fluid)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->neighbor(f)->material_id() == DomainIdentifiers::MaterialIds::Vacuum)
                {
                    std::cout << "   On cell ("
                              << cell->level() << ","
                              << cell->index() << "): ";

                    fe_face_values.reinit(cell, f);

                    const std::vector<Tensor<1,dim>> normal_vectors = fe_face_values.get_normal_vectors();

                    const std::vector<Point<dim>>    q_points = fe_face_values.get_quadrature_points();

                    for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                    {
                        const double tmp = (q_points[q] / q_points[q].norm()) * normal_vectors[q];
                        std::cout << tmp << ", ";
                    }

                    std::cout << std::endl;
                }
    }


}
}  // namespace BuoyantFluid

// explicit instantiation
template void ConductingFluid::ConductingFluidSolver<2>::make_grid();
template void ConductingFluid::ConductingFluidSolver<3>::make_grid();
