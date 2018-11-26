/*
 * magnetic_diffusion_solver.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#ifndef INCLUDE_CONDUCTING_FLUID_SOLVER_H_
#define INCLUDE_CONDUCTING_FLUID_SOLVER_H_

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>

#include <memory>

#include "assembly_data.h"
#include "parameters.h"
#include "timestepping.h"


namespace ConductingFluid
{

using namespace dealii;

/*
 *
 * Templated class for solving the magnetic diffusion problem
 * inside a conductor embedded in an insulator
 *
 * This solver is based on a domain decomposition.
 * Nedelec elements in the conductor and Lagrange elements
 * are used in the insulator.
 *
 */
template <int dim>
class ConductingFluidSolver
{
public:
    ConductingFluidSolver(const double       &time_step = 1e-3,
                          const unsigned int &n_steps = 10,
                          const unsigned int &output_frequency = 5,
                          const unsigned int &refinement_frequency = 100);

    void run();

private:
    void make_grid();

    void set_active_fe_indices();

    void setup_dofs();

    void setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block);

    void assemble_magnetic_system();

    void assemble_magnetic_interface_term(
            const FEFaceValuesBase<dim> &int_fe_face_values,
            const FEFaceValuesBase<dim> &ext_fe_face_values,
            std::vector<Tensor<1,dim>>  &int_phi_values,
            std::vector<typename FEValuesViews::Vector<dim>::curl_type>  &int_curl_values,
            std::vector<double> &ext_phi_values,
            FullMatrix<double> &local_interface_matrix) const;

    void distribute_magnetic_interface_term(
            const FullMatrix<double> &local_interface_matrix,
            const std::vector<types::global_dof_index> &local_fluid_dof_indices,
            const std::vector<types::global_dof_index> &local_vacuum_dof_indices);

    void solve();

    /*
     *
    void output_results() const;

    void refine_mesh();
     *
     */
    const unsigned int magnetic_degree;

    TimeStepping::IMEXCoefficients  imex_coefficients;

    Triangulation<dim>              triangulation;

    const MappingQ<dim>             mapping;

    // magnetic FiniteElement and DoFHandler
    FESystem<dim>   interior_magnetic_fe;
    FESystem<dim>   exterior_magnetic_fe;

    hp::FECollection<dim> fe_collection;
    hp::DoFHandler<dim>   magnetic_dof_handler;

    // magnetic part
    ConstraintMatrix                magnetic_constraints;

    BlockSparsityPattern            magnetic_sparsity_pattern;
    BlockSparseMatrix<double>       magnetic_matrix;

    // vectors of magnetic part
    BlockVector<double>             magnetic_solution;
    BlockVector<double>             old_magnetic_solution;
    BlockVector<double>             old_old_magnetic_solution;
    BlockVector<double>             magnetic_rhs;

    // equation coefficients
    const std::vector<double>       equation_coefficients;

    // monitor of computing times
    TimerOutput                     computing_timer;

private:
    // TODO: goes to parameter file later
    const unsigned int n_steps;
    const unsigned int output_frequency;
    const unsigned int refinement_frequency;

    // time stepping variables
    double                          timestep;
    double                          old_timestep;
    unsigned int                    timestep_number = 0;
    bool                            timestep_modified = false;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_magnetic_matrices = true;
    bool    rebuild_magnetic_preconditioner = true;
};
}  // namespace ConductingFluid

#endif /* INCLUDE_CONDUCTING_FLUID_SOLVER_H_ */
