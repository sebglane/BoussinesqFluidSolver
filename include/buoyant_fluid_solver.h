/*
 * buoyant_fluid_solver.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_FLUID_SOLVER_H_
#define INCLUDE_BUOYANT_FLUID_SOLVER_H_

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>


#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <memory>

#include "assembly_data.h"
#include "timestepping.h"

namespace BuoyantFluid {

using namespace dealii;

/*
 *
 * templated class for solving the Boussinesq problem
 * using Q(p+1)/Q(p)-elements
 *
 */
template <int dim>
class BuoyantFluidSolver
{
public:
    struct Parameters;

    BuoyantFluidSolver(Parameters &parameters_);

    void run();

private:
    void make_grid();

    void setup_dofs();

    void setup_temperature_matrices(const types::global_dof_index n_temperature_dofs);
    void assemble_temperature_system();
    void build_temperature_preconditioner();

    void setup_stokes_matrix(const std::vector<types::global_dof_index> dofs_per_block);
    void assemble_stokes_system();
    void build_stokes_preconditioner();

    void solve();

    std::pair<double,double>    compute_rms_values() const;
    double                      compute_cfl_number() const;

    void update_timestep(const double current_cfl_number);

    void output_results() const;

    void refine_mesh();

    Parameters                      &parameters;

    TimeStepping::IMEXCoefficients  imex_coefficients;

    Tensor<1,dim>                   rotation_vector;

    Triangulation<dim>              triangulation;

    const MappingQ<dim>             mapping;

    // temperature FiniteElement and DoFHandler
    const FE_Q<dim>                 temperature_fe;
    DoFHandler<dim>                 temperature_dof_handler;

    // stokes FiniteElement and DoFHandler
    const FESystem<dim>             stokes_fe;
    DoFHandler<dim>                 stokes_dof_handler;

    // temperature part
    ConstraintMatrix                temperature_constraints;

    SparsityPattern                 temperature_sparsity_pattern;
    SparseMatrix<double>            temperature_matrix;
    SparseMatrix<double>            temperature_mass_matrix;
    SparseMatrix<double>            temperature_stiffness_matrix;

    // vectors of temperature part
    Vector<double>                  temperature_solution;
    Vector<double>                  old_temperature_solution;
    Vector<double>                  old_old_temperature_solution;
    Vector<double>                  temperature_rhs;

    // stokes part
    ConstraintMatrix                stokes_constraints;
    ConstraintMatrix                stokes_laplace_constraints;

    BlockSparsityPattern            stokes_sparsity_pattern;
    BlockSparsityPattern            auxiliary_stokes_sparsity_pattern;
    BlockSparseMatrix<double>       stokes_matrix;
    BlockSparseMatrix<double>       stokes_laplace_matrix;

    SparseMatrix<double>            velocity_mass_matrix;
    SparseMatrix<double>            pressure_mass_matrix;

    // vectors of stokes part
    BlockVector<double>             stokes_solution;
    BlockVector<double>             old_stokes_solution;
    BlockVector<double>             old_old_stokes_solution;
    BlockVector<double>             stokes_rhs;

    // preconditioner types
    typedef TrilinosWrappers::PreconditionAMG           PreconditionerTypeA;
    typedef SparseILU<double>                           PreconditionerTypeKp;
    typedef PreconditionSSOR<SparseMatrix<double>>      PreconditionerTypeMp;
    typedef PreconditionJacobi<SparseMatrix<double>>    PreconditionerTypeT;

    // pointers to preconditioners
    std::shared_ptr<PreconditionerTypeA>        preconditioner_A;
    std::shared_ptr<PreconditionerTypeKp>       preconditioner_Kp;
    std::shared_ptr<PreconditionerTypeMp>       preconditioner_Mp;
    std::shared_ptr<PreconditionerTypeT>        preconditioner_T;

    // postprocessor class
    class PostProcessor;

    // equation coefficients
    const std::vector<double>       equation_coefficients;

    // monitor of computing times
    TimerOutput                     computing_timer;

public:
    struct Parameters
    {
        Parameters(const std::string &parameter_filename);
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);


        // runtime parameters
        bool    workstream_assembly;
        bool    assemble_schur_complement;

        // physics parameters
        double aspect_ratio;
        double Pr;
        double Ra;
        double Ek;

        bool         rotation;

        // linear solver parameters
        double rel_tol;
        double abs_tol;
        unsigned int n_max_iter;

        // time stepping parameters
        TimeStepping::IMEXType  imex_scheme;

        unsigned int    n_steps;

        double  initial_timestep;
        double  min_timestep;
        double  max_timestep;
        double  cfl_min;
        double  cfl_max;

        bool    adaptive_timestep;

        // discretization parameters
        unsigned int temperature_degree;
        unsigned int velocity_degree;

        // refinement parameters
        unsigned int n_global_refinements;
        unsigned int n_initial_refinements;
        unsigned int n_boundary_refinements;
        unsigned int n_max_levels;

        unsigned int refinement_frequency;

        // logging parameters
        unsigned int output_frequency;
    };

private:
    // time stepping variables
    double                          timestep;
    double                          old_timestep;
    unsigned int                    timestep_number = 0;
    bool                            timestep_modified = false;

    // variables for Schur complement approximation
    double                          factor_Mp = 0;
    double                          factor_Kp = 0;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_stokes_matrices = true;
    bool    rebuild_temperature_matrices = true;
    bool    rebuild_stokes_preconditioner = true;
    bool    rebuild_temperature_preconditioner = true;

    // working stream methods for temperature assembly
    void local_assemble_temperature_matrix(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            TemperatureAssembly::Scratch::Matrix<dim> &scratch,
            TemperatureAssembly::CopyData::Matrix<dim> &data);
    void copy_local_to_global_temperature_matrix(
            const TemperatureAssembly::CopyData::Matrix<dim> &data);

    void local_assemble_temperature_rhs(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            TemperatureAssembly::Scratch::RightHandSide<dim> &scratch,
            TemperatureAssembly::CopyData::RightHandSide<dim> &data);
    void copy_local_to_global_temperature_rhs(
            const TemperatureAssembly::CopyData::RightHandSide<dim> &data);

    // working stream methods for stokes assembly
    void local_assemble_stokes_matrix(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            StokesAssembly::Scratch::Matrix<dim> &scratch,
            StokesAssembly::CopyData::Matrix<dim> &data);
    void copy_local_to_global_stokes_matrix(
            const StokesAssembly::CopyData::Matrix<dim> &data);

    void local_assemble_stokes_rhs(
                const typename DoFHandler<dim>::active_cell_iterator &cell,
                StokesAssembly::Scratch::RightHandSide<dim> &scratch,
                StokesAssembly::CopyData::RightHandSide<dim> &data);
    void copy_local_to_global_stokes_rhs(
                const StokesAssembly::CopyData::RightHandSide<dim> &data);
};

}  // namespace BouyantFluid

#endif /* INCLUDE_BUOYANT_FLUID_SOLVER_H_ */
