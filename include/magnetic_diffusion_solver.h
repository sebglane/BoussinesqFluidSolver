/*
 * magnetic_diffusion_solver.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#ifndef INCLUDE_MAGNETIC_DIFFUSION_SOLVER_H_
#define INCLUDE_MAGNETIC_DIFFUSION_SOLVER_H_

#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>

#include <memory>

#include "assembly_data.h"
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
class MagneticDiffusionSolver
{
public:
    MagneticDiffusionSolver(const double          &aspect_ratio = 0.35,
                            const double          &time_step = 1e-3,
                            const unsigned int    &n_steps = 200,
                            const unsigned int    &vtk_frequency = 1,
                            const double          &t_final = 1.0);

    void run();

private:
    void make_grid();

    void setup_dofs();

    void setup_magnetic_matrices(const std::vector<types::global_dof_index> &dofs_per_block);

    void assemble_magnetic_matrices();
    void assemble_magnetic_rhs();

    void assemble_diffusion_system();
    void assemble_projection_system();

    void solve_diffusion_system();
    void solve_projection_system();

    void magnetic_step();

    void output_results(const bool initial_step=false) const;

    std::pair<double, double> compute_rms_values() const;

    const unsigned int magnetic_degree;
    const unsigned int pseudo_pressure_degree;

    TimeStepping::IMEXCoefficients  imex_coefficients;

    Triangulation<dim>              triangulation;

    const MappingQ<dim>             mapping;

    // magnetic FiniteElement and DoFHandler
    FESystem<dim>                   magnetic_fe;

    DoFHandler<dim>                 magnetic_dof_handler;

    // magnetic part
    ConstraintMatrix                magnetic_constraints;

    BlockSparsityPattern            magnetic_sparsity_pattern;
    BlockSparsityPattern            void_sparsity_pattern;

    BlockSparseMatrix<double>       magnetic_matrix;
    BlockSparseMatrix<double>       magnetic_curl_matrix;
    BlockSparseMatrix<double>       magnetic_mass_matrix;
    BlockSparseMatrix<double>       magnetic_stabilization_matrix;

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
    const double        aspect_ratio;

    // TODO: goes to parameter file later
    const unsigned int  n_steps;
    const unsigned int  vtk_frequency;
    const unsigned int  rms_frequency;

    const double        t_final;

    TableHandler        rms_table;

    // time stepping variables
    double              timestep;
    double              old_timestep;

    bool                timestep_modified = false;

    unsigned int        timestep_number = 0;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_magnetic_matrices = true;
    bool    rebuild_magnetic_diffusion_preconditioner = true;
    bool    rebuild_magnetic_projection_preconditioner = true;

};
}  // namespace ConductingFluid

#endif /* INCLUDE_MAGNETIC_DIFFUSION_SOLVER_H_ */
