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
#include <deal.II/base/synchronous_iterator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>

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
enum class ConvectiveWeakForm
{
    Standard,
    DivergenceForm
    /*SkewSymmetric,*/
    /*RotationalForm*/
};

/*
 *
 * enumeration for the type of the pressure projection scheme
 *
 */
enum class PressureProjectionType
{
    Standard,
    Compact,
    NoneIncremental
};


/*
 *
 * enumeration for the type of the pressure update
 *
 */
enum class PressureUpdateType
{
    StandardForm,
    IrrotationalForm
};



namespace NavierStokesAssembly {

using namespace dealii;

namespace Scratch {

template<int dim>
struct PressureMatrix
{
    PressureMatrix(const FiniteElement<dim> &fe,
                   const Mapping<dim>       &mapping,
                   const Quadrature<dim>    &quadrature,
                   const UpdateFlags        update_flags);

    PressureMatrix(const PressureMatrix<dim>  &scratch);

    FEValues<dim>               fe_values;

    std::vector<double>         phi;
    std::vector<Tensor<1,dim>>  grad_phi;

    const unsigned int          n_q_points;
};

template<int dim>
struct VelocityMatrix
{
    VelocityMatrix(const FiniteElement<dim> &fe,
                   const Mapping<dim>       &mapping,
                   const Quadrature<dim>    &quadrature,
                   const UpdateFlags        update_flags);

    VelocityMatrix(const VelocityMatrix<dim>  &scratch);

    FEValues<dim>               fe_values;

    std::vector<Tensor<1,dim>>  phi;
    std::vector<Tensor<2,dim>>  grad_phi;

    const unsigned int          n_q_points;

    const FEValuesExtractors::Vector    velocity;
};

template<int dim>
struct VelocityDiffusion
{
    VelocityDiffusion(const FiniteElement<dim>  &velocity_fe,
                      const FiniteElement<dim>  &pressure_fe,
                      const Mapping<dim>        &mapping,
                      const Quadrature<dim>     &quadrature,
                      const UpdateFlags          velocity_update_flags,
                      const UpdateFlags          pressure_update_flags,
                      const std::array<double,3>&alpha,
                      const std::array<double,2>&beta,
                      const std::array<double,3>&gamma);

    VelocityDiffusion(const VelocityDiffusion<dim>  &scratch);

    FEValues<dim>               fe_values_velocity;
    FEValues<dim>               fe_values_pressure;

    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;
    std::vector<double>         div_phi_velocity;

    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;
    std::vector<Tensor<2,dim>>  old_velocity_gradients;
    std::vector<Tensor<2,dim>>  old_old_velocity_gradients;

    std::vector<double>         old_pressure_values;
    std::vector<double>         pressure_update_values;
    std::vector<double>         old_pressure_update_values;

    const std::array<double,3> &alpha;
    const std::array<double,2> &beta;
    const std::array<double,3> &gamma;

    const unsigned int          n_q_points;

    const FEValuesExtractors::Vector    velocity;
};

template<int dim>
struct PressureProjection
{
    PressureProjection(const FiniteElement<dim>  &velocity_fe,
                       const FiniteElement<dim>  &pressure_fe,
                       const Mapping<dim>        &mapping,
                       const Quadrature<dim>     &quadrature,
                       const UpdateFlags          velocity_update_flags,
                       const UpdateFlags          pressure_update_flags,
                       const std::array<double,3>&alpha);

    PressureProjection(const PressureProjection<dim>  &scratch);

    FEValues<dim>               fe_values_velocity;
    FEValues<dim>               fe_values_pressure;

    std::vector<double>         phi_pressure;
    std::vector<Tensor<1,dim>>  grad_phi_pressure;

    std::vector<double>         velocity_divergences;

    const std::array<double,3> &alpha;

    const unsigned int          n_q_points;

    const FEValuesExtractors::Vector    velocity;
};

template<int dim>
struct VelocityCorrection
{
    VelocityCorrection(const FiniteElement<dim>  &velocity_fe,
                       const FiniteElement<dim>  &pressure_fe,
                       const Mapping<dim>        &mapping,
                       const Quadrature<dim>     &quadrature,
                       const UpdateFlags          velocity_update_flags,
                       const UpdateFlags          pressure_update_flags,
                       const std::array<double,3>&alpha);

    VelocityCorrection(const VelocityCorrection<dim>  &scratch);

    FEValues<dim>   fe_values_velocity;
    FEValues<dim>   fe_values_pressure;

    std::vector<Tensor<1,dim>>  phi_velocity;

    std::vector<Tensor<1,dim>>  tentative_velocity_values;

    std::vector<Tensor<1,dim>>  pressure_gradients;

    const std::array<double,3> &alpha;

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

    unsigned int dofs_per_cell;

    FullMatrix<double>      local_mass_matrix;
    FullMatrix<double>      local_laplace_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

template <int dim>
struct RightHandSides
{
    RightHandSides(const FiniteElement<dim> &fe,
                   const AffineConstraints<double>  &constraints);
    RightHandSides(const RightHandSides<dim> &data);

    const AffineConstraints<double> &constraints;

    unsigned int  dofs_per_cell;

    FullMatrix<double>  local_matrix_for_bc;
    Vector<double>      local_rhs;

    std::vector<types::global_dof_index>   local_dof_indices;
};

}  // namespace Copy

}  // namespace NavierStokesAssembly


namespace NavierStokesObjects
{

template<int dim>
struct DefaultObjects
{
    DefaultObjects(const parallel::distributed::Triangulation<dim>  &triangulation,
                   const Mapping<dim>   &mapping);

    virtual const DoFHandler<dim>&  get_dof_handler() const;

    virtual const AffineConstraints<double>& get_current_constraints() const;

    virtual void  setup_dofs(/* const typename FunctionMap<dim>::type &dirichlet_bcs */) = 0;

    virtual void  assemble_matrices() = 0;

    virtual const FiniteElement<dim>&   get_fe() const = 0;

    virtual const Quadrature<dim>&      get_quadrature() const = 0;

public:
    LA::SparseMatrix    mass_matrix;
    LA::SparseMatrix    stiffness_matrix;

    LA::Vector          solution;
    LA::Vector          old_solution;
    LA::Vector          old_old_solution;

    LA::Vector          rhs;

    IndexSet            locally_owned_dofs;
    IndexSet            locally_relevant_dofs;

    /*
    SparseDirectUMFPACK preconditioner_mass;
    SparseILU<double>   preconditioner;
    */

protected:
    const MPI_Comm      mpi_communicator;

    const Mapping<dim> &mapping;

    DoFHandler<dim>     dof_handler;

    AffineConstraints<double>   hanging_node_constraints;
    AffineConstraints<double>   current_constraints;

    bool                rebuild_matrices;

    virtual void setup_matrices(const IndexSet  &locally_owned_dofs,
                                const IndexSet  &locally_relevant_dofs);
};

template<int dim>
inline const DoFHandler<dim>&
DefaultObjects<dim>::get_dof_handler() const
{
    return dof_handler;
}

template<int dim>
inline const AffineConstraints<double>&
DefaultObjects<dim>::get_current_constraints() const
{
    return current_constraints;
}

template<int dim>
struct PressureObjects : public DefaultObjects<dim>
{
    PressureObjects(const parallel::distributed::Triangulation<dim> &triangulation,
                    const Mapping<dim>         &mapping,
                    const unsigned int          degree);

    virtual void setup_dofs(/*const typename FunctionMap<dim>::type &dirichlet_bcs*/);

    virtual void assemble_matrices();

    virtual const FiniteElement<dim>& get_fe() const;

    virtual const Quadrature<dim>& get_quadrature() const;

    LA::Vector  update;
    LA::Vector  old_update;

private:
    const QGauss<dim>   quadrature;
    FE_Q<dim>           fe;

    void local_assemble_matrix
    (const typename DoFHandler<dim>::active_cell_iterator  &cell,
     NavierStokesAssembly::Scratch::PressureMatrix<dim>    &scratch,
     NavierStokesAssembly::CopyData::Matrix<dim>           &data);

    void copy_local_to_global_matrix
    (const NavierStokesAssembly::CopyData::Matrix<dim>     &data);
};

template<int dim>
inline const FiniteElement<dim>&
PressureObjects<dim>::get_fe() const
{
    return fe;
}

template<int dim>
inline const Quadrature<dim>&
PressureObjects<dim>::get_quadrature() const
{
    return quadrature;
}

template<int dim>
struct VelocityObjects : public DefaultObjects<dim>
{
    VelocityObjects(const parallel::distributed::Triangulation<dim> &triangulation,
                    const Mapping<dim>         &mapping,
                    const unsigned int          degree);

    virtual void setup_dofs(/*const typename FunctionMap<dim>::type &dirichlet_bcs*/);

    virtual void assemble_matrices();

    virtual const FiniteElement<dim>& get_fe() const;

    virtual const Quadrature<dim>& get_quadrature() const;

    const AffineConstraints<double>& get_correction_constraints() const;

public:
    LA::SparseMatrix    system_matrix;
    LA::SparseMatrix    correction_mass_matrix;
    /*
     * I think that these objects are not required in the end...
     *
     */
    /*
    SparseMatrix<double>    mass_stiffness_matrix;
    SparseMatrix<double>    advection_matrix;

    Vector<double>          extrapolated_solution;
     */
    LA::Vector          tentative_solution;

private:
    const QGauss<dim>   quadrature;
    FESystem<dim>       fe;

    AffineConstraints<double>   correction_constraints;

    virtual void setup_matrices(const IndexSet  &locally_owned_dofs,
                                const IndexSet  &locally_relevant_dofs);

    void local_assemble_matrix
    (const typename DoFHandler<dim>::active_cell_iterator  &cell,
     NavierStokesAssembly::Scratch::VelocityMatrix<dim>    &scratch,
     NavierStokesAssembly::CopyData::Matrix<dim>           &data);

    void copy_local_to_global_matrix
    (const NavierStokesAssembly::CopyData::Matrix<dim>     &data);
};

template<int dim>
inline const FiniteElement<dim>&
VelocityObjects<dim>::get_fe() const
{
    return fe;
}

template<int dim>
inline const Quadrature<dim>&
VelocityObjects<dim>::get_quadrature() const
{
    return quadrature;
}

template<int dim>
inline const AffineConstraints<double>&
VelocityObjects<dim>::get_correction_constraints() const
{
    return correction_constraints;
}

}  // namespace NavierStokesObjects



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

    LinearSolverParameters  linear_solver_parameters;

    // finite element degree
    unsigned int    fe_degree_velocity;
    unsigned int    fe_degree_pressure;

    // dimensionless coefficient
    double  equation_coefficient;

    // discretization parameters
    PressureProjectionType          pressure_projection_type;
    PressureUpdateType              pressure_update_type;
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

    const DoFHandler<dim>   &get_veloctiy_dof_handler() const;
    const FiniteElement<dim>&get_velocity_fe() const;
    const LA::Vector        &get_velocity_solution() const;

    unsigned int fe_degree_velocity() const;
    unsigned int fe_degree_pressure() const;

    types::global_dof_index n_dofs_velocity() const;
    types::global_dof_index n_dofs_pressure() const;

    /*
    void set_post_refinement() const;
     */
private:
    void setup_dofs();

    void assemble_diffusion_system();
    void assemble_projection_system();
    void assemble_correction_system();

    void solve_diffusion_system();
    void solve_projection_system();
    void solve_correction_system();

    // reference to parameters
    const NavierStokesParameters &parameters;

    // copy of equation coefficient
    const double    equation_coefficient;

    // pointer to boundary conditions
    std::shared_ptr<const BC::NavierStokesBoundaryConditions<dim>>  boundary_conditions;

    // velocity and pressure objects
    NavierStokesObjects::VelocityObjects<dim>   velocity;
    NavierStokesObjects::PressureObjects<dim>   pressure;

    // joint finite element
    const FESystem<dim> fe;

    // join velocity and pressure solution
    void compute_joined_solution();


    // work stream methods for assembly
    typedef std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                       typename DoFHandler<dim>::active_cell_iterator> IteratorTuple;

    typedef SynchronousIterators<IteratorTuple> IteratorPair;

    void local_assemble_diffusion_rhs
    (const IteratorPair                                    &SI,
     NavierStokesAssembly::Scratch::VelocityDiffusion<dim> &scratch,
     NavierStokesAssembly::CopyData::RightHandSides<dim>   &data);
    void copy_local_to_global_diffusion_rhs
    (const NavierStokesAssembly::CopyData::RightHandSides<dim>   &data);

    void local_assemble_projection_rhs
    (const IteratorPair                                    &SI,
     NavierStokesAssembly::Scratch::PressureProjection<dim>&scratch,
     NavierStokesAssembly::CopyData::RightHandSides<dim>   &data);
    void copy_local_to_global_projection_rhs
    (const NavierStokesAssembly::CopyData::RightHandSides<dim>   &data);

    void local_assemble_correction_rhs
    (const IteratorPair                                    &SI,
     NavierStokesAssembly::Scratch::VelocityCorrection<dim>&scratch,
     NavierStokesAssembly::CopyData::RightHandSides<dim>   &data);
    void copy_local_to_global_correction_rhs
    (const NavierStokesAssembly::CopyData::RightHandSides<dim>   &data);
};

template<int dim>
inline unsigned int
NavierStokesSolver<dim>::fe_degree() const
{
    return fe.degree;
}

template<int dim>
inline unsigned int
NavierStokesSolver<dim>::fe_degree_velocity() const
{
    return fe.base_element(0).degree;
}

template<int dim>
inline unsigned int
NavierStokesSolver<dim>::fe_degree_pressure() const
{
    return fe.base_element(1).degree;
}

template<int dim>
inline const FiniteElement<dim>&
NavierStokesSolver<dim>::get_fe() const
{
    return fe;
}

template<int dim>
inline const DoFHandler<dim>&
NavierStokesSolver<dim>::get_veloctiy_dof_handler() const
{
    return velocity.get_dof_handler();
}

template<int dim>
inline const FiniteElement<dim>&
NavierStokesSolver<dim>::get_velocity_fe() const
{
    return fe.base_element(0);
}

template<int dim>
inline const LA::Vector&
NavierStokesSolver<dim>::get_velocity_solution() const
{
    return velocity.solution;
}

template<int dim>
inline types::global_dof_index
NavierStokesSolver<dim>::n_dofs_velocity() const
{
    return velocity.get_dof_handler().n_dofs();
}

template<int dim>
inline types::global_dof_index
NavierStokesSolver<dim>::n_dofs_pressure() const
{
    return pressure.get_dof_handler().n_dofs();
}

}  // namespace adsolic


#endif /* INCLUDE_ADSOLIC_NAVIER_STOKES_SOLVER_H_ */
