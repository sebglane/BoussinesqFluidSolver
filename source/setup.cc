/*
 * setup_magnetic.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/function_map.h>

#include <deal.II/numerics/vector_tools.h>
#include <magnetic_diffusion_solver.h>

#include "grid_factory.h"

namespace ConductingFluid {

template<int dim>
void MagneticDiffusionSolver<dim>::setup_dofs()
{

    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

    magnetic_dof_handler.distribute_dofs(magnetic_fe);

    DoFRenumbering::Cuthill_McKee(magnetic_dof_handler);

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
              << "      Number of magnetic field degrees of freedom: "
              << dofs_per_block[0]
              << std::endl
              << "      Number of pseudo pressure degrees of freedom: "
              << dofs_per_block[1]
              << std::endl;

    // magnetic constraints
    {
        magnetic_constraints.clear();
        DoFTools::make_hanging_node_constraints(magnetic_dof_handler,
                                                magnetic_constraints);

        const FEValuesExtractors::Scalar    pseudo_pressure(dim);

        {
            const Functions::ZeroFunction<dim>  zero_function(dim+1);
            typename FunctionMap<dim,double>::type  function_map
            {{types::boundary_id(DomainIdentifiers::BoundaryIds::ICB),&zero_function},
             {types::boundary_id(DomainIdentifiers::BoundaryIds::CMB),&zero_function}};

            VectorTools::interpolate_boundary_values(
                    magnetic_dof_handler,
                    function_map,
                    magnetic_constraints,
                    magnetic_fe.component_mask(pseudo_pressure));
        }
        {
            const Functions::ZeroFunction<dim>  zero_function(dim);
            typename FunctionMap<dim,double>::type  function_map
                    {{types::boundary_id(DomainIdentifiers::BoundaryIds::ICB),&zero_function},
                     {types::boundary_id(DomainIdentifiers::BoundaryIds::CMB),&zero_function}};

            const std::set<types::boundary_id>  boundary_ids{DomainIdentifiers::BoundaryIds::ICB,
                                                             DomainIdentifiers::BoundaryIds::CMB};

            VectorTools::compute_nonzero_tangential_flux_constraints(
                    magnetic_dof_handler,
                    0,
                    boundary_ids,
                    function_map,
                    magnetic_constraints);
        }
        magnetic_constraints.close();
    }

    setup_magnetic_matrices(dofs_per_block);

    magnetic_solution.reinit(dofs_per_block);
    old_magnetic_solution.reinit(dofs_per_block);
    old_old_magnetic_solution.reinit(dofs_per_block);

    magnetic_rhs.reinit(dofs_per_block);
}

template<int dim>
void MagneticDiffusionSolver<dim>::setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block)
{
    magnetic_matrix.clear();
    magnetic_curl_matrix.clear();
    magnetic_mass_matrix.clear();
    magnetic_stabilization_matrix.clear();

    // sparsity pattern for magnetic matrix
    {
        Table<2,DoFTools::Coupling> coupling(dim+1, dim+1);

        for (unsigned int i=0; i<dim+1; ++i)
            for (unsigned int j=0; j<dim+1; ++j)
                // magnetic-magnetic coupling
                if (i<dim && j<dim)
                    coupling[i][j] = DoFTools::Coupling::always;
                // magnetic-pseudo pressure coupling
                else if ((i<dim && j==dim) || (i==dim && j<dim))
                    coupling[i][j] = DoFTools::Coupling::always;
                // pseudo pressure-pseudo pressure coupling
                else if (i==dim && i==j)
                    coupling[i][j] = DoFTools::Coupling::always;
                else
                    coupling[i][j] = DoFTools::none;

        BlockDynamicSparsityPattern dsp(dofs_per_block,
                                        dofs_per_block);

        DoFTools::make_sparsity_pattern(magnetic_dof_handler,
                                        coupling,
                                        dsp,
                                        magnetic_constraints);

        magnetic_sparsity_pattern.copy_from(dsp);
    }
    magnetic_matrix.reinit(magnetic_sparsity_pattern);

    // void sparsity pattern
    {
        // auxiliary coupling structure
        Table<2,DoFTools::Coupling> void_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                void_coupling[c][d] = DoFTools::none;

        BlockDynamicSparsityPattern dsp(dofs_per_block,
                                        dofs_per_block);

        DoFTools::make_sparsity_pattern(magnetic_dof_handler,
                                        void_coupling,
                                        dsp,
                                        magnetic_constraints);

        void_sparsity_pattern.copy_from(dsp);
    }

    magnetic_curl_matrix.reinit(void_sparsity_pattern);
    magnetic_curl_matrix.block(0,0).reinit(magnetic_sparsity_pattern.block(0,0));
    magnetic_curl_matrix.block(1,1).reinit(magnetic_sparsity_pattern.block(1,1));

    magnetic_mass_matrix.reinit(void_sparsity_pattern);
    magnetic_mass_matrix.block(0,0).reinit(magnetic_sparsity_pattern.block(0,0));

    magnetic_stabilization_matrix.reinit(void_sparsity_pattern);
    magnetic_stabilization_matrix.block(0,0).reinit(magnetic_sparsity_pattern.block(0,0));
    magnetic_stabilization_matrix.block(0,1).reinit(magnetic_sparsity_pattern.block(0,1));
    magnetic_stabilization_matrix.block(1,1).reinit(magnetic_sparsity_pattern.block(1,1));

    rebuild_magnetic_matrices = true;
}

}  // namespace ConductingFluid

// explicit instantiation
template void ConductingFluid::MagneticDiffusionSolver<3>::setup_dofs();

template void ConductingFluid::MagneticDiffusionSolver<3>::setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block);
