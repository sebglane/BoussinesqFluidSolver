/*
 * convection_diffusion_solver.h
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_CONVECTION_DIFFUSION_SOLVER_H_
#define INCLUDE_ADSOLIC_CONVECTION_DIFFUSION_SOLVER_H_

#include <memory>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/distributed/tria.h>

#include <adsolic/boundary_conditions.h>
#include <adsolic/linear_algebra.h>
#include <adsolic/timestepping.h>
#include <adsolic/utility.h>

namespace adsolic {

using namespace dealii;

using namespace TimeStepping;

using namespace AuxiliaryFunctions;

namespace ConvectionDiffusionAssembly {

namespace Scratch {

template<int dim>
struct RightHandSide
{
    /*
     * Constructor for passing the velocity field through FEValues<dim> object.
     */
    /*
     *
    RightHandSide(const FiniteElement<dim>     &temperature_fe,
                  const Mapping<dim>           &mapping,
                  const Quadrature<dim>        &temperature_quadrature,
                  const UpdateFlags             temperature_update_flags,
                  const FiniteElement<dim>     &stokes_fe,
                  const UpdateFlags             stokes_update_flags,
                  const unsigned int            first_velocity_component,
                  const std::array<double,3>   &alpha,
                  const std::array<double,2>   &beta,
                  const std::array<double,3>   &gamma);
     *
     */

    RightHandSide(const FiniteElement<dim>     &fe,
                  const Mapping<dim>           &mapping,
                  const Quadrature<dim>        &quadrature,
                  const UpdateFlags             update_flags,
                  const std::array<double,3>   &alpha,
                  const std::array<double,2>   &beta,
                  const std::array<double,3>   &gamma);

    /*
     * Copy constructor.
     */
    RightHandSide(const RightHandSide<dim> &scratch);

    FEValues<dim>               fe_values;
    std::vector<double>         phi;
    std::vector<Tensor<1,dim>>  grad_phi;
    std::vector<double>         old_values;
    std::vector<double>         old_old_values;
    std::vector<Tensor<1,dim>>  old_gradients;
    std::vector<Tensor<1,dim>>  old_old_gradients;

    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;

    const std::array<double,3> &alpha;
    const std::array<double,2> &beta;
    const std::array<double,3> &gamma;

    const unsigned int          dofs_per_cell;
    const unsigned int          n_q_points;
};

template<int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &fe,
           const Mapping<dim>       &mapping,
           const Quadrature<dim>    &quadrature,
           const UpdateFlags         update_flags);

    Matrix(const Matrix<dim>  &scratch);

    FEValues<dim>               fe_values;

    std::vector<double>         phi;
    std::vector<Tensor<1,dim>>  grad_phi;
};

}  // namespace Scratch

namespace CopyData {

template <int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim>    &fe);
    RightHandSide(const RightHandSide<dim>    &data);

    Vector<double>                          local_rhs;
    FullMatrix<double>                      matrix_for_bc;
    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &fe);
    Matrix(const Matrix<dim>        &data);

    FullMatrix<double>      local_mass_matrix;
    FullMatrix<double>      local_laplace_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

}  // namespace CopyData

}  // namespace TemperatureAssembly


struct ConvectionDiffusionParameters
{
    ConvectionDiffusionParameters();
    ConvectionDiffusionParameters(const std::string &parameter_filename);

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    /*
     * function forwarding parameters to a stream object
     */
    template<typename Stream>
    void write(Stream &stream) const;

    // dimensionless coefficient in convection diffusion equation
    double          equation_coefficient;

    // finite element degree
    unsigned int    fe_degree;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;
    unsigned int    n_max_iter;

    // verbosity
    bool            verbose;
};


/*
 *
 * templated class for solving a convection diffusion equation of the form
 *
 *      d\phi/dt + v . grad(\phi) = C div(grad(\phi)).
 *
 */

template <int dim>
class ConvectionDiffusionSolver
{
public:
    ConvectionDiffusionSolver
    (const ConvectionDiffusionParameters &parameters,
     const parallel::distributed::Triangulation<dim> &triangulation_in,
     const MappingQ<dim>   &mapping_in,
     const IMEXTimeStepping&timestepper_in,
     const std::shared_ptr<const BC::ScalarBoundaryConditions<dim>> boundary_descriptor =
             std::shared_ptr<const BC::ScalarBoundaryConditions<dim>>(),
     const std::shared_ptr<TimerOutput> external_timer =
             std::shared_ptr<TimerOutput>());

    void advance_in_time();

    void set_convection_function(const std::shared_ptr<ConvectionFunction<dim>> &function);

    void setup_problem();

    void set_post_refinement() const;

    void setup_initial_condition(const Function<dim> &initial_field);

    const FiniteElement<dim> &get_fe() const;

    unsigned int fe_degree() const;

    types::global_dof_index n_dofs() const;

    const DoFHandler<dim>  &get_dof_handler() const;
    const ConstraintMatrix &get_constraints() const;
    const LA::Vector       &get_solution() const;

private:
    void setup_dofs();

    void extrapolate_solution();

    void setup_system_matrix
    (const IndexSet &locally_owned_dofs,
     const IndexSet &locally_relevant_dofs);

    void assemble_system();

    void assemble_system_matrix();

    void build_preconditioner();

    void solve_linear_system();

    void advance_solution();

    // reference to parameters
    const ConvectionDiffusionParameters &parameters;

    // reference to common triangulation
    const parallel::distributed::Triangulation<dim>   &triangulation;

    // reference to common mapping
    const MappingQ<dim>&mapping;

    // reference to time stepper
    const IMEXTimeStepping   &timestepper;

    // copy of equation coefficient
    const double        equation_coefficient;

    // parallel output
    ConditionalOStream  pcout;

    // pointer to boundary conditions
    std::shared_ptr<const BC::ScalarBoundaryConditions<dim>>  boundary_conditions;

    // pointer to convective function
    std::shared_ptr<const ConvectionFunction<dim>>  convection_function;

    // pointer to monitor of computing times
    std::shared_ptr<TimerOutput> computing_timer;

    // FiniteElement and DoFHandler
    const FE_Q<dim>     fe;
    DoFHandler<dim>     dof_handler;

    // matrices
    IndexSet            locally_owned_dofs;
    IndexSet            locally_relevant_dofs;

    ConstraintMatrix    constraints;

    LA::SparseMatrix    system_matrix;
    LA::SparseMatrix    mass_matrix;
    LA::SparseMatrix    stiffness_matrix;

    // vectors
    LA::Vector          solution;
    LA::Vector          old_solution;
    LA::Vector          old_old_solution;
    LA::Vector          rhs;

    // pointers to preconditioners
    std::shared_ptr<LA::PreconditionSSOR> preconditioner;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_matrices = true,
            rebuild_preconditioner = true;
    mutable bool setup_dofs_flag = true;

    // work stream methods for temperature assembly
    void local_assemble_rhs
    (const typename DoFHandler<dim>::active_cell_iterator       &cell,
     ConvectionDiffusionAssembly::Scratch::RightHandSide<dim>   &scratch,
     ConvectionDiffusionAssembly::CopyData::RightHandSide<dim>  &data);
    void copy_local_to_global_rhs
    (const ConvectionDiffusionAssembly::CopyData::RightHandSide<dim>    &data);

    void local_assemble_matrix
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     ConvectionDiffusionAssembly::Scratch::Matrix<dim>      &scratch,
     ConvectionDiffusionAssembly::CopyData::Matrix<dim>     &data);
    void copy_local_to_global_matrix
    (const ConvectionDiffusionAssembly::CopyData::Matrix<dim>    &data);
};

template<int dim>
inline void ConvectionDiffusionSolver<dim>::set_post_refinement() const
{
    setup_dofs_flag = true;
}

}  // namespace adsolic

#endif /* INCLUDE_ADSOLIC_CONVECTION_DIFFUSION_SOLVER_H_ */
