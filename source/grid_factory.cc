/*
 * grid_factor.cc
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#include "grid_factory.h"

namespace GridFactory {

template <int dim>
SinusoidalManifold<dim>::SinusoidalManifold(const double wavenumber,
                                            const double amplitude)
:
ChartManifold<dim,dim,dim-1>(),
wavenumber(wavenumber),
amplitude(amplitude)
{}

template <int dim>
std::unique_ptr<Manifold<dim,dim>> SinusoidalManifold<dim>::clone() const
{
  return std::make_unique<SinusoidalManifold<dim>>();
}

template<int dim>
Point<dim-1> SinusoidalManifold<dim>::pull_back(const Point<dim> &space_point) const
{
    Point<dim-1> chart_point;
    for (unsigned int d=0; d<dim-1; ++d)
        chart_point[d] = space_point[d];
    return chart_point;
}

template<int dim>
Point<dim> SinusoidalManifold<dim>::push_forward(const Point<dim-1> &chart_point) const
{
    Point<dim> space_point;
    space_point[dim-1] = amplitude;
    for (unsigned int d=0; d<dim-1; ++d)
    {
        space_point[d] = chart_point[d];
        space_point[dim-1] *= std::sin(wavenumber * chart_point[d]);
    }
    space_point[dim-1] += 1.0;
    return space_point;
}


template<int dim>
SphericalShell<dim>::SphericalShell(
        const double aspect_ratio_,
        const bool   include_interior,
        const bool   include_exterior_,
        const double exterior_length_)
:
aspect_ratio(aspect_ratio_),
include_core(include_interior),
include_exterior(include_exterior_),
exterior_length(exterior_length_),
spherical_manifold(),
interpolation_manifold()
{
    Assert(aspect_ratio > 0, ExcLowerRangeType<double>(aspect_ratio, 0));
    Assert(aspect_ratio < 1, ExcLowerRangeType<double>(1, aspect_ratio));
    Assert(2./std::sqrt(2.) < exterior_length, ExcLowerRangeType<double>(2./std::sqrt(2.),exterior_length));
}

template<>
void SphericalShell<1>::create_coarse_mesh(Triangulation<1> &/* coarse_grid */)
{
    Assert(false,ExcImpossibleInDim(1));
}

template<>
void SphericalShell<2>::create_coarse_mesh(Triangulation<2> &coarse_grid)
{
    const unsigned int dim = 2;

    // shell mesh
    if (!include_core && !include_exterior)
    {
        GridGenerator::hyper_shell(coarse_grid,
                                   Point<dim>(),
                                   aspect_ratio, 1.0);

        for(auto cell: coarse_grid.active_cell_iterators())
            cell->set_material_id(MaterialIds::Fluid);
    }
    // shell mesh including interior sphere
    else if (include_core && !include_exterior)
    {
        const double a = 1./(1.+std::sqrt(2.0));

        const unsigned int n_vertices = 12;
        const Point<dim> vertices[n_vertices] =
        {
                Point<dim>(-1,-1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,-1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,+1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,+1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,-1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,-1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,+1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,+1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,-1) * (1.0/std::sqrt(2.)),
                Point<dim>(+1,-1) * (1.0/std::sqrt(2.)),
                Point<dim>(+1,+1) * (1.0/std::sqrt(2.)),
                Point<dim>(-1,+1) * (1.0/std::sqrt(2.))
        };

        const unsigned int n_cells = 9;
        const unsigned int cell_vertices[n_cells][GeometryInfo<dim>::vertices_per_cell] =
        {
                // central rectangle
                {0,1,3,2},
                // first ring
                {4,5,0,1},
                {5,6,1,2},
                {7,3,6,2},
                {4,0,7,3},
                // second ring
                {8,9,4,5},
                {9,10,5,6},
                {11,7,10,6},
                {8,4,11,7}
        };

        const types::material_id v = MaterialIds::Vacuum;
        const types::material_id f = MaterialIds::Fluid;

        const types::manifold_id manifold_ids[n_cells] = {0,1,1,1,1,2,2,2,2};
        const types::material_id material_ids[n_cells] = {v,v,v,v,v,f,f,f,f};

        std::vector<CellData<dim>> cells(n_cells);
        for (unsigned int c=0; c<n_cells; ++c)
        {
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                cells[c].vertices[v] = cell_vertices[c][v];
            cells[c].material_id = material_ids[c];
            cells[c].manifold_id = manifold_ids[c];
        }


        coarse_grid.create_triangulation(
                std::vector<Point<dim>>(&vertices[0],&vertices[n_vertices]),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            if (!cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_manifold_id(2);
                        break;
                    }

        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(1, interpolation_manifold);

        coarse_grid.set_manifold(2, spherical_manifold);
        coarse_grid.set_all_manifold_ids_on_boundary(2);
    }
    // shell mesh including exterior sphere
    else if (!include_core && include_exterior)
    {
        const int dim = 2;

        const unsigned int n_vertices = 12;
        const Point<dim> vertices[n_vertices] =
        {
                Point<dim>(-1,-1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,-1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,+1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,+1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,-1) * (1.0/std::sqrt(2.)),
                Point<dim>(+1,-1) * (1.0/std::sqrt(2.)),
                Point<dim>(+1,+1) * (1.0/std::sqrt(2.)),
                Point<dim>(-1,+1) * (1.0/std::sqrt(2.)),
                Point<dim>(-1,-1) * (exterior_length/std::sqrt(2.)),
                Point<dim>(+1,-1) * (exterior_length/std::sqrt(2.)),
                Point<dim>(+1,+1) * (exterior_length/std::sqrt(2.)),
                Point<dim>(-1,+1) * (exterior_length/std::sqrt(2.))
        };

        const unsigned int n_cells = 8;
        const unsigned int cell_vertices[n_cells][GeometryInfo<dim>::vertices_per_cell] =
        {
                // first ring
                {4,5,0,1},
                {5,6,1,2},
                {7,3,6,2},
                {4,0,7,3},
                // second ring
                {8,9,4,5},
                {9,10,5,6},
                {11,7,10,6},
                {8,4,11,7}
        };

        const types::material_id f = MaterialIds::Fluid;
        const types::material_id v = MaterialIds::Vacuum;

        const types::manifold_id manifold_ids[n_cells] = {0,0,0,0,0,0,0,0};
        const types::material_id material_ids[n_cells] = {f,f,f,f,v,v,v,v};

        std::vector<CellData<dim>> cells(n_cells);
        for (unsigned int c=0; c<n_cells; ++c)
        {
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                cells[c].vertices[v] = cell_vertices[c][v];
            cells[c].material_id = material_ids[c];
            cells[c].manifold_id = manifold_ids[c];
        }

        coarse_grid.create_triangulation(
                std::vector<Point<dim>>(&vertices[0],&vertices[n_vertices]),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (!cell->face(f)->at_boundary())
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_manifold_id(0);
                        break;
                    }

        coarse_grid.set_manifold(0, spherical_manifold);
        coarse_grid.set_all_manifold_ids_on_boundary(0);
    }
    // shell mesh including interior and exterior sphere
    else if (include_core && include_exterior)
    {
        const double a = 1./(1.+std::sqrt(2.0));

        const unsigned int n_vertices = 16;
        const Point<dim> vertices[n_vertices] =
        {
                Point<dim>(-1,-1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,-1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,+1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,+1) * (a*aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,-1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,-1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(+1,+1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,+1) * (aspect_ratio/std::sqrt(2.)),
                Point<dim>(-1,-1) * (1.0/std::sqrt(2.)),
                Point<dim>(+1,-1) * (1.0/std::sqrt(2.)),
                Point<dim>(+1,+1) * (1.0/std::sqrt(2.)),
                Point<dim>(-1,+1) * (1.0/std::sqrt(2.)),
                Point<dim>(-1,-1) * (exterior_length/std::sqrt(2.)),
                Point<dim>(+1,-1) * (exterior_length/std::sqrt(2.)),
                Point<dim>(+1,+1) * (exterior_length/std::sqrt(2.)),
                Point<dim>(-1,+1) * (exterior_length/std::sqrt(2.))
        };

        const unsigned int n_cells = 13;
        const unsigned int cell_vertices[n_cells][GeometryInfo<dim>::vertices_per_cell] =
        {
                // central rectangle
                {0,1,3,2},
                // first ring
                {4,5,0,1},
                {5,6,1,2},
                {7,3,6,2},
                {4,0,7,3},
                // second ring
                {8,9,4,5},
                {9,10,5,6},
                {11,7,10,6},
                {8,4,11,7},
                // third ring
                {12,13,8,9},
                {13,14,9,10},
                {15,11,14,10},
                {12,8,15,11}
        };

        const types::material_id f = MaterialIds::Fluid;
        const types::material_id v = MaterialIds::Vacuum;

        const types::manifold_id manifold_ids[n_cells] = {0,1,1,1,1,2,2,2,2,2,2,2,2};
        const types::material_id material_ids[n_cells] = {v,v,v,v,v,f,f,f,f,v,v,v,v};

        std::vector<CellData<dim>> cells(n_cells);
        for (unsigned int c=0; c<n_cells; ++c)
        {
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                cells[c].vertices[v] = cell_vertices[c][v];
            cells[c].material_id = material_ids[c];
            cells[c].manifold_id = manifold_ids[c];
        }

        coarse_grid.create_triangulation(
                std::vector<Point<dim>>(&vertices[0],&vertices[n_vertices]),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            if (!cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_manifold_id(2);
                        break;
                    }

        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(1, interpolation_manifold);
        coarse_grid.set_manifold(2, spherical_manifold);
        coarse_grid.set_all_manifold_ids_on_boundary(2);
    }

    for (auto cell: coarse_grid.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    std::vector<double> dist(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        dist[v] = cell->face(f)->vertex(v).distance(Point<dim>());

                    if (std::all_of(dist.begin(), dist.end(),
                            [&](double d)->bool{return std::abs(d - aspect_ratio) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::ICB);
                    if (std::all_of(dist.begin(), dist.end(),
                            [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::CMB);
                    if (std::all_of(dist.begin(), dist.end(),
                            [&](double d)->bool{return std::abs(d - exterior_length) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::FVB);
                }
}

template<int dim>
TopographyBox<dim>::TopographyBox(const double wavenumber,
                                const double amplitude,
                                const bool   include_exterior,
                                const double exterior_length)
:
include_exterior(include_exterior),
exterior_length(exterior_length),
sinus_manifold(wavenumber, amplitude)
{
    Assert(amplitude < 1.0, ExcLowerRangeType<double>(amplitude,1.0));
}


template<int dim>
void TopographyBox<dim>::create_coarse_mesh(Triangulation<dim> &coarse_grid)
{
    if (!include_exterior)
    {
        GridGenerator::hyper_cube(coarse_grid);

        coarse_grid.set_all_manifold_ids(0);
        coarse_grid.set_all_manifold_ids_on_boundary(0);

        for (auto cell: coarse_grid.active_cell_iterators())
        {
            cell->set_material_id(MaterialIds::Fluid);

            if (cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->face(f)->at_boundary())
                    {
                        std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[dim-1];

                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        {
                            cell->face(f)->set_boundary_id(BoundaryIds::TopoBndry);
                            cell->face(f)->set_manifold_id(1);
                            break;
                        }
                    }
        }
        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);
        coarse_grid.set_manifold(1, sinus_manifold);
    }
    else if (include_exterior)
    {
        const Point<dim> origin;
        Point<dim> corner;
        for (unsigned int d=0; d<dim-1; ++d)
            corner[d] = 1.0;
        corner[dim-1] = exterior_length + 1.0;

        std::vector<std::vector<double>> step_sizes;
        for (unsigned int d=0; d<dim-1; ++d)
            step_sizes.push_back(std::vector<double>(1,1.));

        step_sizes.push_back(std::vector<double>{1.0, exterior_length});

        GridGenerator::subdivided_hyper_rectangle(
                coarse_grid,
                step_sizes,
                origin,
                corner);

        coarse_grid.set_all_manifold_ids(0);
        coarse_grid.set_all_manifold_ids_on_boundary(0);

        for (auto cell: coarse_grid.active_cell_iterators())
        {
            if (cell->center()[dim-1] < 1.0)
                cell->set_material_id(MaterialIds::Fluid);
            else if (cell->center()[dim-1] > 1.0)
                cell->set_material_id(MaterialIds::Vacuum);

            if (cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (!cell->face(f)->at_boundary())
                    {
                        std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[dim-1];

                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        {
                            cell->face(f)->set_manifold_id(1);
                            break;
                        }
                    }
        }
        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);
        coarse_grid.set_manifold(1, sinus_manifold);
    }
    else
        Assert(false, ExcInternalError());
}
}  // namespace GridFactory

template class GridFactory::SphericalShell<2>;
template class GridFactory::TopographyBox<2>;
template class GridFactory::TopographyBox<3>;
template class GridFactory::SinusoidalManifold<2>;
template class GridFactory::SinusoidalManifold<3>;
