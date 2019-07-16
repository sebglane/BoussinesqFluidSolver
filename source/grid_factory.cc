/*
 * grid_factor.cc
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#include <grid_factory.h>

#include <cmath>

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

        const types::manifold_id manifold_ids[n_cells] =
        {
                numbers::invalid_manifold_id,
                0,0,0,0,
                1,1,1,1
        };
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
                std::vector<Point<dim>>(std::begin(vertices),std::end(vertices)),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            if (!cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_all_manifold_ids(1);
                        break;
                    }

        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);

        coarse_grid.set_manifold(1, spherical_manifold);
        coarse_grid.set_all_manifold_ids_on_boundary(1);
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
                std::vector<Point<dim>>(std::begin(vertices),std::end(vertices)),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (!cell->face(f)->at_boundary())
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_all_manifold_ids(0);
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
                std::vector<Point<dim>>(std::begin(vertices),std::end(vertices)),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            if (!cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_all_manifold_ids(2);
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

template<>
void SphericalShell<3>::create_coarse_mesh(Triangulation<3> &coarse_grid)
{
    const unsigned int dim = 3;

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
        // equilibrate cell sizes at transition
        const double a = 1. / (1 + std::sqrt(3.0));

        // from the inner part to the radial cells
        const unsigned int n_vertices           = 24;
        const Point<dim>     vertices[n_vertices] =
        {
                // first the vertices of the inner cell
                Point<dim>(-1, -1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, -1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, -1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(-1, -1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(-1, +1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, +1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, +1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(-1, +1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                // now the eight vertices on the inner sphere
                Point<dim>(-1, -1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (aspect_ratio / std::sqrt(3.0)),
                // now the eight vertices on the outer sphere
                Point<dim>(-1, -1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (1.0 / std::sqrt(3.0)),
        };

        // one needs to draw the seven cubes to
        // understand what's going on here
        const unsigned int n_cells                   = 13;
        const int          cell_vertices[n_cells][8] =
        {
                {0, 1, 4, 5, 3, 2, 7, 6},      // center
                // now the six cells of the inner ring
                {8, 9, 12, 13, 0, 1, 4, 5},    // bottom
                {9, 13, 1, 5, 10, 14, 2, 6},   // right
                {11, 10, 3, 2, 15, 14, 7, 6},  // top
                {8, 0, 12, 4, 11, 3, 15, 7},   // left
                {8, 9, 0, 1, 11, 10, 3, 2},    // front
                {12, 4, 13, 5, 15, 7, 14, 6},  // back
                // now the six cells of the outer ring
                {16, 17, 20, 21, 8, 9, 12, 13},    // bottom
                {17, 21, 9, 13, 18, 22, 10, 14},   // right
                {19, 18, 11, 10, 23, 22, 15, 14},  // top
                {16, 8, 20, 12, 19, 11, 23, 15},   // left
                {16, 17, 8, 9, 19, 18, 11, 10},    // front
                {20, 12, 21, 13, 23, 15, 22, 14},  // back
        };

        const types::material_id v = MaterialIds::Vacuum;
        const types::material_id f = MaterialIds::Fluid;

        const types::manifold_id manifold_ids[n_cells] =
        {
                numbers::invalid_material_id,
                0,0,0,0,0,0,
                1,1,1,1,1,1
        };
        const types::material_id material_ids[n_cells] =
        {
                v,
                v,v,v,v,v,v,
                f,f,f,f,f,f
        };

        std::vector<CellData<dim>> cells(n_cells, CellData<dim>());

        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j)
              cells[i].vertices[j] = cell_vertices[i][j];
            cells[i].material_id = material_ids[i];
            cells[i].manifold_id = manifold_ids[i];
          }

        coarse_grid.create_triangulation(
                std::vector<Point<dim>>(std::begin(vertices),std::end(vertices)),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (!cell->face(f)->at_boundary())
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_all_manifold_ids(0);
                        break;
                    }

        coarse_grid.set_all_manifold_ids_on_boundary(1);

        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);
        coarse_grid.set_manifold(1, spherical_manifold);
    }
    // shell mesh including exterior sphere
    else if (!include_core && include_exterior)
    {
        // from the inner part to the radial cells
        const unsigned int n_vertices           = 24;
        const Point<dim>     vertices[n_vertices] =
        {
                // first the eight vertices on the inner sphere
                Point<dim>(-1, -1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (aspect_ratio / std::sqrt(3.0)),
                // now the eight vertices on the middle sphere
                Point<dim>(-1, -1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (1.0 / std::sqrt(3.0)),
                // now the eight vertices on the outer sphere
                Point<dim>(-1, -1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (exterior_length / std::sqrt(3.0)),
        };

        // one needs to draw the seven cubes to
        // understand what's going on here
        const unsigned int n_cells                   = 12;
        const int          cell_vertices[n_cells][8] =
        {
                // now the six cells of the inner ring
                {8, 9, 12, 13, 0, 1, 4, 5},    // bottom
                {9, 13, 1, 5, 10, 14, 2, 6},   // right
                {11, 10, 3, 2, 15, 14, 7, 6},  // top
                {8, 0, 12, 4, 11, 3, 15, 7},   // left
                {8, 9, 0, 1, 11, 10, 3, 2},    // front
                {12, 4, 13, 5, 15, 7, 14, 6},  // back
                // now the six cells of the outer ring
                {16, 17, 20, 21, 8, 9, 12, 13},    // bottom
                {17, 21, 9, 13, 18, 22, 10, 14},   // right
                {19, 18, 11, 10, 23, 22, 15, 14},  // top
                {16, 8, 20, 12, 19, 11, 23, 15},   // left
                {16, 17, 8, 9, 19, 18, 11, 10},    // front
                {20, 12, 21, 13, 23, 15, 22, 14},  // back
        };

        const types::material_id v = MaterialIds::Vacuum;
        const types::material_id f = MaterialIds::Fluid;

        const types::manifold_id manifold_ids[n_cells] =
        {
                0,0,0,0,0,0,
                0,0,0,0,0,0
        };
        const types::material_id material_ids[n_cells] =
        {
                f,f,f,f,f,f,
                v,v,v,v,v,v
        };

        std::vector<CellData<dim>> cells(n_cells, CellData<dim>());

        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j)
              cells[i].vertices[j] = cell_vertices[i][j];
            cells[i].material_id = material_ids[i];
            cells[i].manifold_id = manifold_ids[i];
          }

        coarse_grid.create_triangulation(
                std::vector<Point<dim>>(std::begin(vertices),std::end(vertices)),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            if (!cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_all_manifold_ids(0);
                        break;
                    }

        coarse_grid.set_all_manifold_ids_on_boundary(0);
        coarse_grid.set_manifold(0, spherical_manifold);
    }
    // shell mesh including interior and exterior sphere
    else if (include_core && include_exterior)
    {
        // equilibrate cell sizes at transition
        const double a = 1. / (1 + std::sqrt(3.0));

        // from the inner part to the radial cells
        const unsigned int n_vertices           = 32;
        const Point<dim>     vertices[n_vertices] =
        {
                // first the vertices of the inner cell
                Point<dim>(-1, -1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, -1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, -1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(-1, -1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(-1, +1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, +1, -1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(+1, +1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                Point<dim>(-1, +1, +1) * (aspect_ratio / std::sqrt(3.0) * a),
                // now the eight vertices on the inner sphere
                Point<dim>(-1, -1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (aspect_ratio / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (aspect_ratio / std::sqrt(3.0)),
                // now the eight vertices on the middle sphere
                Point<dim>(-1, -1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (1.0 / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (1.0 / std::sqrt(3.0)),
                // now the eight vertices on the outer sphere
                Point<dim>(-1, -1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, -1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, -1, +1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(-1, -1, +1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(-1, +1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, +1, -1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(+1, +1, +1) * (exterior_length / std::sqrt(3.0)),
                Point<dim>(-1, +1, +1) * (exterior_length / std::sqrt(3.0)),
        };

        // one needs to draw the seven cubes to
        // understand what's going on here
        const unsigned int n_cells                   = 19;
        const int          cell_vertices[n_cells][8] =
        {
                {0, 1, 4, 5, 3, 2, 7, 6},      // center
                // now the six cells of the inner ring
                {8, 9, 12, 13, 0, 1, 4, 5},    // bottom
                {9, 13, 1, 5, 10, 14, 2, 6},   // right
                {11, 10, 3, 2, 15, 14, 7, 6},  // top
                {8, 0, 12, 4, 11, 3, 15, 7},   // left
                {8, 9, 0, 1, 11, 10, 3, 2},    // front
                {12, 4, 13, 5, 15, 7, 14, 6},  // back
                // now the six cells of the shell ring
                {16, 17, 20, 21, 8, 9, 12, 13},    // bottom
                {17, 21, 9, 13, 18, 22, 10, 14},   // right
                {19, 18, 11, 10, 23, 22, 15, 14},  // top
                {16, 8, 20, 12, 19, 11, 23, 15},   // left
                {16, 17, 8, 9, 19, 18, 11, 10},    // front
                {20, 12, 21, 13, 23, 15, 22, 14},  // back
                // now the six cells of the outer ring
                {16+8, 17+8, 20+8, 21+8, 8+8, 9+8, 12+8, 13+8},    // bottom
                {17+8, 21+8, 9+8, 13+8, 18+8, 22+8, 10+8, 14+8},   // right
                {19+8, 18+8, 11+8, 10+8, 23+8, 22+8, 15+8, 14+8},  // top
                {16+8, 8+8, 20+8, 12+8, 19+8, 11+8, 23+8, 15+8},   // left
                {16+8, 17+8, 8+8, 9+8, 19+8, 18+8, 11+8, 10+8},    // front
                {20+8, 12+8, 21+8, 13+8, 23+8, 15+8, 22+8, 14+8},  // back
        };

        const types::material_id v = MaterialIds::Vacuum;
        const types::material_id f = MaterialIds::Fluid;

        const types::manifold_id manifold_ids[n_cells] =
        {
                numbers::invalid_manifold_id,
                0,0,0,0,0,0,
                1,1,1,1,1,1,
                1,1,1,1,1,1
        };
        const types::material_id material_ids[n_cells] =
        {
                v,
                v,v,v,v,v,v,
                f,f,f,f,f,f,
                v,v,v,v,v,v
        };

        std::vector<CellData<dim>> cells(n_cells, CellData<dim>());

        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j)
              cells[i].vertices[j] = cell_vertices[i][j];
            cells[i].material_id = material_ids[i];
            cells[i].manifold_id = manifold_ids[i];
          }

        coarse_grid.create_triangulation(
                std::vector<Point<dim>>(std::begin(vertices),std::end(vertices)),
                cells,
                SubCellData());

        for (auto cell: coarse_grid.active_cell_iterators())
            if (!cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->neighbor(f)->material_id() != cell->material_id())
                    {
                        cell->face(f)->set_all_manifold_ids(1);
                        break;
                    }

        coarse_grid.set_all_manifold_ids_on_boundary(1);
        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);
        coarse_grid.set_manifold(1, spherical_manifold);
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

template<int dim>
Cavity<dim>::Cavity(const double aspect_ratio_)
:
aspect_ratio(aspect_ratio_)
{
    Assert(aspect_ratio > 0, ExcLowerRangeType<double>(aspect_ratio, 0));
}

template<>
void Cavity<2>::create_coarse_mesh(Triangulation<2> &coarse_grid)
{
    const unsigned int dim = 2;

    std::vector<unsigned int>   repetitions(dim, 1);

    if (aspect_ratio > 1.0)
        repetitions[1] = std::round(aspect_ratio);
    else
        repetitions[1] = std::round(1. / aspect_ratio);

    GridGenerator::subdivided_hyper_rectangle(coarse_grid,
                                              repetitions,
                                              Point<dim>(),
                                              Point<dim>({1., aspect_ratio}));

    const double tol = 1e-12;
    for (auto cell: coarse_grid.active_cell_iterators())
    {
        cell->set_material_id(MaterialIds::Fluid);

        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    std::vector<Point<dim>> boundary_vertices(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        boundary_vertices[v] = cell->face(f)->vertex(v);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[0]) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Left);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[1]) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Bottom);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[0] - 1.0) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Right);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[1] - aspect_ratio) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Top);
                }
    }
}

template<>
void Cavity<3>::create_coarse_mesh(Triangulation<3> &coarse_grid)
{
    const unsigned int dim = 3;

    std::vector<unsigned int>   repetitions(dim, 1);

    if (aspect_ratio > 1.0)
        repetitions[dim-1] = std::round(aspect_ratio);
    else
        repetitions[dim-1] = std::round(1. / aspect_ratio);

    GridGenerator::subdivided_hyper_rectangle(coarse_grid,
                                              repetitions,
                                              Point<dim>(),
                                              Point<dim>({1., 1., aspect_ratio}));

    const double tol = 1e-12;
    for (auto cell: coarse_grid.active_cell_iterators())
    {
        cell->set_material_id(MaterialIds::Fluid);

        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (!cell->face(f)->at_boundary())
                {
                    std::vector<Point<dim>> boundary_vertices(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        boundary_vertices[v] = cell->face(f)->vertex(v);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[0]) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Left);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[0] - 1.0) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Right);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[1]) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Front);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[1] - 1.0) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Back);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[2]) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Bottom);

                    if (std::all_of(boundary_vertices.begin(), boundary_vertices.end(),
                            [&](Point<dim> &point)->bool{return std::abs(point[2] - aspect_ratio) < tol;}))
                        cell->face(f)->set_boundary_id(BoundaryIds::Top);
                }
    }
}


}  // namespace GridFactory

template class GridFactory::SphericalShell<2>;
template class GridFactory::SphericalShell<3>;
template class GridFactory::TopographyBox<2>;
template class GridFactory::TopographyBox<3>;
template class GridFactory::SinusoidalManifold<2>;
template class GridFactory::SinusoidalManifold<3>;
template class GridFactory::Cavity<2>;
template class GridFactory::Cavity<3>;
