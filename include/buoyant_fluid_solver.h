/*
 * buoyant_fluid_solver.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_FLUID_SOLVER_H_
#define INCLUDE_BUOYANT_FLUID_SOLVER_H_

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

/**
 * A namespace that contains typedefs for classes used in the linear algebra
 * description.
 */
namespace LA
{
    /**
     * Typedef for the vector type used.
     */
    typedef dealii::TrilinosWrappers::MPI::Vector Vector;

    /**
     * Typedef for the type used to describe vectors that consist of multiple
     * blocks.
     */
    typedef dealii::TrilinosWrappers::MPI::BlockVector BlockVector;

    /**
     * Typedef for the sparse matrix type used.
     */
    typedef dealii::TrilinosWrappers::SparseMatrix SparseMatrix;

    /**
     * Typedef for the type used to describe sparse matrices that consist of
     * multiple blocks.
     */
    typedef dealii::TrilinosWrappers::BlockSparseMatrix BlockSparseMatrix;

    /**
     * Typedef for the base class for all preconditioners.
     */
    typedef dealii::TrilinosWrappers::PreconditionBase PreconditionBase;

    /**
     * Typedef for the AMG preconditioner type used for the top left block of
     * the Stokes matrix.
     */
    typedef dealii::TrilinosWrappers::PreconditionAMG PreconditionAMG;

    /**
     * Typedef for the Incomplete Cholesky preconditioner used for other
     * blocks of the system matrix.
     */
    typedef dealii::TrilinosWrappers::PreconditionIC PreconditionIC;

    /**
     * Typedef for the Incomplete LU decomposition preconditioner used for
     * other blocks of the system matrix.
     */
    typedef dealii::TrilinosWrappers::PreconditionILU PreconditionILU;

    /**
     * Typedef for the Jacobi preconditioner.
     */
    typedef dealii::TrilinosWrappers::PreconditionJacobi PreconditionJacobi;

    /**
     * Typedef for the SSOR preconditioner.
     */
    typedef dealii::TrilinosWrappers::PreconditionBlockSSOR PreconditionSSOR;

    /**
     * Typedef for the SSOR preconditioner.
     */
    typedef dealii::TrilinosWrappers::PreconditionSOR PreconditionSOR;


    /**
     * Typedef for the block compressed sparsity pattern type.
     */
    typedef dealii::TrilinosWrappers::BlockSparsityPattern BlockDynamicSparsityPattern;

    /**
     * Typedef for the compressed sparsity pattern type.
     */
    typedef dealii::TrilinosWrappers::SparsityPattern DynamicSparsityPattern;

    /**
     * Typedef for conjugate gradient linear solver.
     */
    typedef dealii::TrilinosWrappers::SolverCG  SolverCG;

    /**
     * Typedef for gmres method linear solver.
     */
    typedef dealii::TrilinosWrappers::SolverGMRES  SolverGMRES;

}

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/distributed/tria.h>

#include <memory>
#include <tuple>

#include "assembly_data.h"
#include "exceptions.h"
#include "parameters.h"
#include "timestepping.h"

namespace BuoyantFluid {

using namespace dealii;

using namespace GeometryExceptions;

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
    BuoyantFluidSolver(Parameters &parameters_);

    void run();

private:
    void make_grid();
    void make_coarse_grid();

    void setup_dofs();

    void setup_temperature_matrix
    (const IndexSet &locally_owned_dofs,
     const IndexSet &locally_relevant_dofs);

    void assemble_temperature_system();
    void build_temperature_preconditioner();
    void solve_temperature_system();
    void temperature_step();

    void compute_initial_pressure();

    void setup_navier_stokes_system
    (const std::vector<IndexSet>    &locally_owned_dofs,
     const std::vector<IndexSet>    &locally_relevant_dofs);

    void assemble_navier_stokes_matrices();

    void assemble_diffusion_system();
    void assemble_projection_system();

    void build_diffusion_preconditioner();
    void build_projection_preconditioner();
    void build_pressure_mass_preconditioner();

    void solve_diffusion_system();
    void solve_projection_system();

    void navier_stokes_step();

    std::vector<double> compute_global_averages() const;
    double              compute_cfl_number() const;

    /*
     * benchmarking methods
     */
    double                      compute_radial_velocity_locally
                                (const double &radius,
                                 const double &phi,
                                 const double &theta) const;
    double                      compute_radial_velocity
                                (const double &radius,
                                 const double &phi,
                                 const double &theta) const;

    double                      compute_azimuthal_gradient_of_radial_velocity_locally
                                (const double &radius,
                                 const double &phi,
                                 const double &theta) const;
    double                      compute_azimuthal_gradient_of_radial_velocity
                                (const double &radius,
                                 const double &phi,
                                 const double &theta) const;

    std::pair<double,double>    compute_benchmark_requests_locally
                                (const double   &radius,
                                 const double   &phi,
                                 const double   &theta) const;
    std::pair<double,double>    compute_benchmark_requests
                                (const double   &radius,
                                 const double   &phi,
                                 const double   &theta) const;

    /**
     * Computes a zero of the radial velocity $v_r(r=r_\mathrm{m},\theta=0,
     * \varphi)$ for a given initial guess $\varphi^*$. The solution is located
     * on a circle of radius $r_\mathrm{m}$ in the equatorial plane. This
     * function is using the boost::math::root::bracket_and_solve_root algorithm.
     *
     * This function requires that the gradient in azimuthal direction is positive
     * for the intial guess, i.e. $\frac{\delta v_r}{\delta \varphi} > 0$.
     */
    double                      compute_zero_of_radial_velocity
                                (const double       &phi_guess,
                                 const bool          local_slope,
                                 const double       &tol = 1e-3,
                                 const unsigned int &max_iter = 100) const;

    void                        update_benchmark_point();

    void update_timestep(const double current_cfl_number);

    void output_results(const bool initial_condition=false) const;

    void refine_mesh();

    void create_snapshot(const double time=0);

    void resume_from_snapshot();

    MPI_Comm                        mpi_communicator;

    Parameters                     &parameters;

    TimeStepping::IMEXCoefficients  imex_coefficients;

    Tensor<1,dim>                   rotation_vector;

    parallel::distributed::Triangulation<dim>   triangulation;

    const MappingQ<dim>             mapping;

    // temperature FiniteElement and DoFHandler
    const FE_Q<dim>                 temperature_fe;
    DoFHandler<dim>                 temperature_dof_handler;

    IndexSet                        locally_owned_temperature_dofs;
    IndexSet                        locally_relevant_temperature_dofs;

    // stokes FiniteElement and DoFHandler
    const FESystem<dim>             navier_stokes_fe;
    DoFHandler<dim>                 navier_stokes_dof_handler;

    std::vector<IndexSet>           locally_owned_stokes_dofs;
    std::vector<IndexSet>           locally_relevant_stokes_dofs;

    // temperature part
    ConstraintMatrix                temperature_constraints;

    LA::SparseMatrix     temperature_matrix;
    LA::SparseMatrix     temperature_mass_matrix;
    LA::SparseMatrix     temperature_stiffness_matrix;

    // vectors of temperature part
    LA::Vector           temperature_solution;
    LA::Vector           old_temperature_solution;
    LA::Vector           old_old_temperature_solution;
    LA::Vector           temperature_rhs;

    // stokes part
    ConstraintMatrix                navier_stokes_constraints;
    ConstraintMatrix                stokes_pressure_constraints;

    LA::BlockSparseMatrix    navier_stokes_matrix;
    LA::BlockSparseMatrix    navier_stokes_laplace_matrix;
    LA::BlockSparseMatrix    navier_stokes_mass_matrix;

    // vectors of navier stokes part
    LA::BlockVector      navier_stokes_solution;
    LA::BlockVector      old_navier_stokes_solution;
    LA::BlockVector      old_old_navier_stokes_solution;
    LA::BlockVector      navier_stokes_rhs;

    LA::BlockVector      phi_pressure;
    LA::BlockVector      old_phi_pressure;
    LA::BlockVector      old_old_phi_pressure;

    // pointers to preconditioners
    std::shared_ptr<LA::PreconditionSSOR>
    preconditioner_temperature;

    std::shared_ptr<LA::PreconditionAMG>
    preconditioner_asymmetric_diffusion;

    std::shared_ptr<LA::PreconditionAMG>
    preconditioner_symmetric_diffusion;

    std::shared_ptr<LA::PreconditionAMG>
    preconditioner_projection;

    std::shared_ptr<LA::PreconditionJacobi>
    preconditioner_pressure_mass;

    // equation coefficients
    const std::vector<double>       equation_coefficients;

    // parallel output
    ConditionalOStream              pcout;

    // monitor of computing times
    TimerOutput                     computing_timer;

    // time stepping variables
    double                          timestep;
    double                          old_timestep;
    double                          old_alpha_zero;
    unsigned int                    timestep_number = 0;
    bool                            timestep_modified = false;

    // benchmark variables
    TableHandler                    benchmark_table;
    Point<dim>                      benchmark_point;
    double                          phi_benchmark;

    // logging variables
    TableHandler                    global_avg_table;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_navier_stokes_matrices = true,
            rebuild_temperature_matrices = true,
            rebuild_temperature_preconditioner = true,
            rebuild_diffusion_preconditioner = true,
            rebuild_projection_preconditioner = true,
            rebuild_pressure_mass_preconditioner = true;

    // working stream methods for temperature assembly
    void local_assemble_temperature_rhs
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     TemperatureAssembly::Scratch::RightHandSide<dim>       &scratch,
     TemperatureAssembly::CopyData::RightHandSide<dim>      &data);
    void copy_local_to_global_temperature_rhs
    (const TemperatureAssembly::CopyData::RightHandSide<dim>    &data);

    void local_assemble_temperature_matrix
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     TemperatureAssembly::Scratch::Matrix<dim>       &scratch,
     TemperatureAssembly::CopyData::Matrix<dim>      &data);
    void copy_local_to_global_temperature_matrix
    (const TemperatureAssembly::CopyData::Matrix<dim>    &data);


    // working stream methods for stokes assembly
    void local_assemble_stokes_matrix
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     NavierStokesAssembly::Scratch::Matrix<dim>             &scratch,
     NavierStokesAssembly::CopyData::Matrix<dim>            &data);
    void copy_local_to_global_stokes_matrix
    (const NavierStokesAssembly::CopyData::Matrix<dim>      &data);

    void local_assemble_stokes_convection_matrix
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     NavierStokesAssembly::Scratch::ConvectionMatrix<dim>   &scratch,
     NavierStokesAssembly::CopyData::ConvectionMatrix<dim>  &data);
    void copy_local_to_global_stokes_convection_matrix
    (const NavierStokesAssembly::CopyData::ConvectionMatrix<dim>    &data);

    void local_assemble_stokes_rhs_explicit
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     NavierStokesAssembly::Scratch::RightHandSide<dim>      &scratch,
     NavierStokesAssembly::CopyData::RightHandSide<dim>     &data);
    void local_assemble_stokes_rhs_implicit
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     NavierStokesAssembly::Scratch::RightHandSide<dim>      &scratch,
     NavierStokesAssembly::CopyData::RightHandSide<dim>     &data);
    void copy_local_to_global_stokes_rhs
    (const NavierStokesAssembly::CopyData::RightHandSide<dim>   &data);
};

}  // namespace BouyantFluid

#endif /* INCLUDE_BUOYANT_FLUID_SOLVER_H_ */
