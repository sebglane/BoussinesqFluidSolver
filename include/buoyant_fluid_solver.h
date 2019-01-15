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
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/vector.h>

#include <memory>

#include "assembly_data.h"
#include "parameters.h"
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
    BuoyantFluidSolver(Parameters &parameters_);

    void run();

private:
    void make_grid();

    void setup_dofs();

    void setup_temperature_matrix(const types::global_dof_index n_dofs);
    void assemble_temperature_system();
    void build_temperature_preconditioner();
    void solve_temperature_system();
    void temperature_step();

    void setup_velocity_system(const types::global_dof_index n_dofs);
    void setup_pressure_system(const types::global_dof_index n_dofs);

    void assemble_velocity_system();
    void assemble_pressure_system();

    void build_diffusion_preconditioner();
    void build_projection_preconditioner();
    void build_pressure_mass_preconditioner();

    void solve_diffusion_system();
    void solve_projection_system();

    void navier_stokes_step();

    std::pair<double,double>    compute_rms_values() const;
    double                      compute_cfl_number() const;

    void update_timestep(const double current_cfl_number);

    void output_results() const;

    void refine_mesh();

    Parameters                     &parameters;

    TimeStepping::IMEXCoefficients  imex_coefficients;

    Tensor<1,dim>                   rotation_vector;

    Triangulation<dim>              triangulation;

    const MappingQ<dim>             mapping;

    // temperature FiniteElement and DoFHandler
    const FE_Q<dim>                 temperature_fe;
    DoFHandler<dim>                 temperature_dof_handler;

    // velocity FiniteElement and DoFHandler
    const FESystem<dim>             velocity_fe;
    DoFHandler<dim>                 velocity_dof_handler;

    // pressure FiniteElement and DoFHandler
    const FE_Q<dim>                 pressure_fe;
    DoFHandler<dim>                 pressure_dof_handler;

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

    // velocity part
    ConstraintMatrix                velocity_constraints;

    SparsityPattern                 velocity_sparsity_pattern;

    SparseMatrix<double>            velocity_matrix;
    SparseMatrix<double>            velocity_laplace_matrix;
    SparseMatrix<double>            velocity_mass_matrix;

    // vectors of velocity part
    Vector<double>                  velocity_solution;
    Vector<double>                  old_velocity_solution;
    Vector<double>                  old_old_velocity_solution;
    Vector<double>                  velocity_rhs;

    // pressure part
    ConstraintMatrix                pressure_constraints;

    SparsityPattern                 pressure_sparsity_pattern;

    SparseMatrix<double>            pressure_laplace_matrix;
    SparseMatrix<double>            pressure_mass_matrix;

    // vectors of pressure part
    Vector<double>                  pressure_solution;
    Vector<double>                  old_pressure_solution;
    Vector<double>                  pressure_rhs;

    Vector<double>                  phi_solution;
    Vector<double>                  old_phi_solution;
    Vector<double>                  old_old_phi_solution;

    // preconditioner types
    typedef PreconditionJacobi<SparseMatrix<double>>
    PreconditionerTypeTemperature;

    typedef PreconditionSSOR<SparseMatrix<double>>
    PreconditionerTypeDiffusion;

    typedef SparseILU<double>
    PreconditionerTypeProjection;

    typedef PreconditionJacobi<SparseMatrix<double>>
    PreconditionerTypePressureMass;

    // pointers to preconditioners
    std::shared_ptr<PreconditionerTypeTemperature>
    preconditioner_temperature;

    std::shared_ptr<PreconditionerTypeDiffusion>
    preconditioner_diffusion;

    std::shared_ptr<PreconditionerTypeProjection>
    preconditioner_projection;

    std::shared_ptr<PreconditionerTypeProjection>
    preconditioner_pressure_mass;

    // equation coefficients
    const std::vector<double>       equation_coefficients;

    // monitor of computing times
    TimerOutput                     computing_timer;

private:
    // time stepping variables
    double                          timestep;
    double                          old_timestep;
    unsigned int                    timestep_number = 0;
    bool                            timestep_modified = false;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_velocity_matrices = true,
            rebuild_temperature_matrices = true,
            rebuild_pressure_matrices = true,
            rebuild_temperature_preconditioner = true,
            rebuild_diffusion_preconditioner = true,
            rebuild_projection_preconditioner = true,
            rebuild_pressure_mass_preconditioner = true;

    // working stream methods for temperature assembly
    void local_assemble_temperature_rhs(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            TemperatureAssembly::Scratch::RightHandSide<dim> &scratch,
            TemperatureAssembly::CopyData::RightHandSide<dim> &data);
    void copy_local_to_global_temperature_rhs(
            const TemperatureAssembly::CopyData::RightHandSide<dim> &data);

    // working stream methods for velocity assembly
    void local_assemble_velocity_rhs(
                const typename DoFHandler<dim>::active_cell_iterator &cell,
                NavierStokesAssembly::Scratch::RightHandSide<dim> &scratch,
                NavierStokesAssembly::CopyData::RightHandSide<dim> &data);
    void copy_local_to_global_velocity_rhs(
                const NavierStokesAssembly::CopyData::RightHandSide<dim> &data);

    // working stream methods for pressure assembly
    void local_assemble_pressure_rhs(
                const typename DoFHandler<dim>::active_cell_iterator &cell,
                PressureAssembly::Scratch::RightHandSide<dim> &scratch,
                PressureAssembly::CopyData::RightHandSide<dim> &data);
    void copy_local_to_global_pressure_rhs(
                const PressureAssembly::CopyData::RightHandSide<dim> &data);
};
}  // namespace BouyantFluid

#endif /* INCLUDE_BUOYANT_FLUID_SOLVER_H_ */
