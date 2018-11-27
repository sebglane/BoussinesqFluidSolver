/*
 * setup_magnetic.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/vector_tools.h>

#include "conducting_fluid_solver.h"
#include "grid_factory.h"

namespace ConductingFluid {

template<int dim>
void ConductingFluidSolver<dim>::set_active_fe_indices()
{
    std::cout << "   Setup active indices..." << std::endl;
    for (auto cell: magnetic_dof_handler.active_cell_iterators())
    {
        if (cell->material_id() == DomainIdentifiers::MaterialIds::Fluid ||
            cell->material_id() == DomainIdentifiers::MaterialIds::Solid)
            cell->set_active_fe_index(0);
        else if (cell->material_id() == DomainIdentifiers::MaterialIds::Vacuum)
            cell->set_active_fe_index(1);
        else
            Assert (false, ExcInternalError());
    }
}

template<int dim>
void ConductingFluidSolver<dim>::setup_dofs()
{

    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

    set_active_fe_indices();

    magnetic_dof_handler.distribute_dofs(fe_collection);

    DoFRenumbering::boost::king_ordering(magnetic_dof_handler);

    DoFRenumbering::block_wise(magnetic_dof_handler);

    // IO
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(magnetic_dof_handler,
                                   dofs_per_block);

    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "      Number of degrees of freedom: "
              << magnetic_dof_handler.n_dofs()
              << std::endl
              << "      Number of vector potential degrees of freedom: "
              << dofs_per_block[0]
              << std::endl
              << "      Number of scalar potential degrees of freedom: "
              << dofs_per_block[1]
              << std::endl;

    // magnetic constraints
    {
        magnetic_constraints.clear();
        DoFTools::make_hanging_node_constraints(magnetic_dof_handler,
                                                magnetic_constraints);

        const FEValuesExtractors::Scalar scalar_potential(dim);
        VectorTools::interpolate_boundary_values(magnetic_dof_handler,
                                                 DomainIdentifiers::BoundaryIds::FVB,
                                                 ZeroFunction<dim>(dim+1),
                                                 magnetic_constraints,
                                                 fe_collection.component_mask(scalar_potential));

        magnetic_constraints.close();
    }

    setup_magnetic_matrices(dofs_per_block);

    magnetic_solution.reinit(dofs_per_block);
    old_magnetic_solution.reinit(dofs_per_block);
    old_old_magnetic_solution.reinit(dofs_per_block);
    magnetic_rhs.reinit(dofs_per_block);
}

template<int dim>
void ConductingFluidSolver<dim>::setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block)
{
    BlockDynamicSparsityPattern dsp(dofs_per_block,
                                    dofs_per_block);
    Table<2,DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
            fe_collection.n_components());
    Table<2,DoFTools::Coupling> face_coupling(fe_collection.n_components(),
            fe_collection.n_components());

    for (unsigned int i=0; i<fe_collection.n_components(); ++i)
        for (unsigned int j=0; j<fe_collection.n_components(); ++j)
            if ((i<dim && j<dim) || (i==j))
                cell_coupling[i][j] = DoFTools::Coupling::always;
            else if ((i<dim && j==dim) || (i==dim && j<dim))
                face_coupling[i][j] = DoFTools::Coupling::nonzero;

    DoFTools::make_flux_sparsity_pattern(magnetic_dof_handler,
            dsp,
            cell_coupling,
            face_coupling);

    magnetic_constraints.condense(dsp);
    magnetic_sparsity_pattern.copy_from(dsp);

    magnetic_matrix.reinit(magnetic_sparsity_pattern);
}
}  // namespace ConductingFluid

// explicit instantiation
template void ConductingFluid::ConductingFluidSolver<2>::set_active_fe_indices();
template void ConductingFluid::ConductingFluidSolver<3>::set_active_fe_indices();

template void ConductingFluid::ConductingFluidSolver<2>::setup_dofs();
template void ConductingFluid::ConductingFluidSolver<3>::setup_dofs();

template void ConductingFluid::ConductingFluidSolver<2>::setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block);
template void ConductingFluid::ConductingFluidSolver<3>::setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block);
