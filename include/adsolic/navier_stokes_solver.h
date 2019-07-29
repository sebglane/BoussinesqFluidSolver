/*
 * navier_stokes_solver.h
 *
 *  Created on: Jul 29, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_NAVIER_STOKES_SOLVER_H_
#define INCLUDE_ADSOLIC_NAVIER_STOKES_SOLVER_H_

#include <memory>

#include <deal.II/base/index_set.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/distributed/tria.h>

#include <adsolic/boundary_conditions.h>
#include <adsolic/linear_algebra.h>
#include <adsolic/solver_base.h>


namespace adsolic {

using namespace dealii;

/*
 *
 * enumeration for the type of the weak form of the convective term
 *
 */
enum ConvectiveWeakForm
{
    Standard,
    DivergenceForm,
    SkewSymmetric,
    RotationalForm
};

/*
 *
 * enumeration for the type of the pressure projection scheme
 *
 */
enum PressureUpdateType
{
    StandardForm,
    IrrotationalForm
};



namespace NavierStokesAssembly {

using namespace dealii;

namespace Scratch {

template<int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &fe,
           const Mapping<dim>       &mapping,
           const Quadrature<dim>    &quadrature,
           const UpdateFlags        update_flags);

    Matrix(const Matrix<dim>  &scratch);

    FEValues<dim>               fe_values;

    std::vector<double>         div_phi_velocity;
    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;

    std::vector<double>         phi_pressure;
    std::vector<Tensor<1,dim>>  grad_phi_pressure;
};

template<int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim>  &fe,
                  const Mapping<dim>        &mapping,
                  const Quadrature<dim>     &quadrature,
                  const UpdateFlags          update_flags,
                  const std::array<double,3>&alpha,
                  const std::array<double,2>&beta,
                  const std::array<double,3>&gamma);

    RightHandSide(const RightHandSide<dim>  &scratch);

    FEValues<dim>               fe_values;
    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;
    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;
    std::vector<Tensor<2,dim>>  old_velocity_gradients;
    std::vector<Tensor<2,dim>>  old_old_velocity_gradients;

    const std::array<double,3> &alpha;
    const std::array<double,2> &beta;
    const std::array<double,3> &gamma;

    const unsigned int          dofs_per_cell;
    const unsigned int          n_q_points;

    const FEValuesExtractors::Vector    velocity;
};


}  // namespace Scratch

namespace CopyData {

template <int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &fe);
    Matrix(const Matrix<dim>        &data);

    FullMatrix<double>      local_matrix;
    FullMatrix<double>      local_mass_matrix;
    FullMatrix<double>      local_laplace_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

template <int dim>
struct RightHandSide
{
    RightHandSide(const FiniteElement<dim> &fe);
    RightHandSide(const RightHandSide<dim> &data);

    Vector<double>          local_rhs;

    std::vector<types::global_dof_index>   local_dof_indices;
};

}  // namespace Copy

}  // namespace NavierStokesAssembly


struct NavierStokesParameters
{
    NavierStokesParameters();
    NavierStokesParameters(const std::string &parameter_filename);

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    /*
     * function forwarding parameters to a stream object
     */
    template<typename Stream>
    void write(Stream &stream) const;

    // dimensionless coefficients
    std::vector<double> equation_coefficients;

    LinearSolverParameters  linear_solver_parameters;

    // finite element degree
    unsigned int    fe_degree_velocity;
    unsigned int    fe_degree_pressure;

    // discretization parameters
    PressureUpdateType              projection_scheme;
    ConvectiveWeakForm              convective_weak_form;

    // verbosity
    bool            verbose;
};


/*
 *
 * templated class for solving Navier Stokes equations of the form
 *
 *      div(v) = 0
 *
 *      dv/dt + v . grad(v) + C1 Omega x v
 *          = - grad(p) + C2 div(grad(v)) - C3 T g + C4 curl(B) x B
 *
 */
template <int dim>
class NavierStokesSolver : public SolverBase<dim,LA::BlockVector>
{
public:
    NavierStokesSolver
    (const NavierStokesParameters &parameters,
     const parallel::distributed::Triangulation<dim> &triangulation_in,
     const MappingQ<dim>   &mapping_in,
     const IMEXTimeStepping&timestepper_in,
     const std::shared_ptr<const BC::NavierStokesBoundaryConditions<dim>> boundary_descriptor =
             std::shared_ptr<const BC::NavierStokesBoundaryConditions<dim>>(),
     const std::shared_ptr<TimerOutput> external_timer =
             std::shared_ptr<TimerOutput>());

    virtual void advance_in_time();

    virtual void setup_problem();

    virtual void setup_initial_condition
    (const Function<dim> &initial_field);

    virtual const FiniteElement<dim> &get_fe() const;

    virtual unsigned int fe_degree() const;
    unsigned int fe_degree_velocity() const;
    unsigned int fe_degree_pressure() const;

    /*
    void set_post_refinement() const;
     */
private:
    void setup_dofs();

    void setup_system_matrix
    (const IndexSet &locally_owned_dofs,
     const IndexSet &locally_relevant_dofs);

    void assemble_system();

    void assemble_system_matrix();

    void build_preconditioner();

    void assemble_diffusion_system();
    void assemble_projection_system();

    void build_diffusion_preconditioner();
    void build_projection_preconditioner();
    void build_pressure_mass_preconditioner();

    void solve_diffusion_system();
    void solve_projection_system();

    // reference to parameters
    const NavierStokesParameters &parameters;

    // copy of equation coefficient
    const std::vector<double>   equation_coefficients;

    // pointer to boundary conditions
    std::shared_ptr<const BC::NavierStokesBoundaryConditions<dim>>  boundary_conditions;

    // FiniteElement
    const FESystem<dim> fe;

    // matrices
    IndexSet            locally_owned_dofs;
    IndexSet            locally_relevant_dofs;

    ConstraintMatrix    hanging_node_constraints;
    ConstraintMatrix    pressure_constraints;
    ConstraintMatrix    tentative_velocity_constraints;
    ConstraintMatrix    neumann_velocity_constraints;

    LA::SparseMatrix    system_matrix;
    LA::SparseMatrix    mass_matrix;
    LA::SparseMatrix    stiffness_matrix;

    // pointers to preconditioners
    std::shared_ptr<LA::PreconditionAMG>
    preconditioner_diffusion;

    std::shared_ptr<LA::PreconditionAMG>
    preconditioner_projection;

    std::shared_ptr<LA::PreconditionJacobi>
    preconditioner_pressure_mass;

    std::shared_ptr<LA::PreconditionJacobi>
    preconditioner_velocity_mass;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_matrices = true,
            rebuild_preconditioner = true;
    mutable bool setup_dofs_flag = true;

    // work stream methods for navier stokes assembly
    void local_assemble_stokes_matrix
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     NavierStokesAssembly::Scratch::Matrix<dim>             &scratch,
     NavierStokesAssembly::CopyData::Matrix<dim>            &data);
    void copy_local_to_global_stokes_matrix
    (const NavierStokesAssembly::CopyData::Matrix<dim>      &data);

    void local_assemble_stokes_rhs
    (const typename DoFHandler<dim>::active_cell_iterator   &cell,
     NavierStokesAssembly::Scratch::RightHandSide<dim>      &scratch,
     NavierStokesAssembly::CopyData::RightHandSide<dim>     &data);
    void copy_local_to_global_stokes_rhs
    (const NavierStokesAssembly::CopyData::RightHandSide<dim>   &data);

};

}  // namespace adsolic


#endif /* INCLUDE_ADSOLIC_NAVIER_STOKES_SOLVER_H_ */
