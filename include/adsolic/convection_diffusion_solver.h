/*
 * temperature_solver.h
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_TEMPERATURE_SOLVER_H_
#define INCLUDE_ADSOLIC_TEMPERATURE_SOLVER_H_


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

#include <memory>
#include <tuple>

#include <adsolic/linear_algebra.h>
#include <adsolic/timestepping.h>

namespace adsolic {

using namespace dealii;
using namespace TimeStepping;

namespace TemperatureAssembly {

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

    RightHandSide(const FiniteElement<dim>     &temperature_fe,
                  const Mapping<dim>           &mapping,
                  const Quadrature<dim>        &temperature_quadrature,
                  const UpdateFlags             temperature_update_flags,
                  TensorFunction<1,dim>        &advection_field,
                  const std::array<double,3>   &alpha,
                  const std::array<double,2>   &beta,
                  const std::array<double,3>   &gamma);

    /*
     * Copy constructor.
     */
    RightHandSide(const RightHandSide<dim> &scratch);

    FEValues<dim>               temperature_fe_values;
    std::vector<double>         phi_temperature;
    std::vector<Tensor<1,dim>>  grad_phi_temperature;
    std::vector<double>         old_temperature_values;
    std::vector<double>         old_old_temperature_values;
    std::vector<Tensor<1,dim>>  old_temperature_gradients;
    std::vector<Tensor<1,dim>>  old_old_temperature_gradients;

//    FEValues<dim>               stokes_fe_values;
    TensorFunction<1,dim>      &advection_field;
    std::vector<Tensor<1,dim>>  old_velocity_values;
    std::vector<Tensor<1,dim>>  old_old_velocity_values;

//    const FEValuesExtractors::Vector velocity;

    const std::array<double,3> &alpha;
    const std::array<double,2> &beta;
    const std::array<double,3> &gamma;

    const unsigned int          dofs_per_cell;
    const unsigned int          n_q_points;
};

template<int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &temperature_fe,
           const Mapping<dim>       &mapping,
           const Quadrature<dim>    &temperature_quadrature,
           const UpdateFlags         temperature_update_flags);

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
    RightHandSide(const FiniteElement<dim>    &temperature_fe);
    RightHandSide(const RightHandSide<dim>    &data);

    Vector<double>                          local_rhs;
    FullMatrix<double>                      matrix_for_bc;
    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
struct Matrix
{
    Matrix(const FiniteElement<dim> &temperature_fe);
    Matrix(const Matrix<dim>        &data);

    FullMatrix<double>      local_mass_matrix;
    FullMatrix<double>      local_laplace_matrix;

    std::vector<types::global_dof_index>   local_dof_indices;
};

}  // namespace CopyData

}  // namespace TemperatureAssembly


struct TemperatureParameters
{
    TemperatureParameters();
    TemperatureParameters(const std::string &parameter_filename);
    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    // dimensionless coefficient in convection diffusion equation
    double  equation_coefficient;

    // finite element degree
    unsigned int temperature_degree;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;
    unsigned int    n_max_iter;

    // verbosity
    bool            verbose;
};


/*
 *
 * templated class for solving the heat conduction equation of the form
 *
 *      dT/dt + v . grad(T) = C div(grad(T)).
 *
 */

template <int dim>
class TemperatureSolver
{
public:
    TemperatureSolver(TemperatureParameters &parameters,
                      parallel::distributed::Triangulation<dim> &triangulation_in,
                      MappingQ<dim>         &mapping_in,
                      IMEXTimeStepping      &timestepper_in,
                      TimerOutput           *external_timer = 0);

    void evaluate_time_step();

    virtual void setup_problem(const Function<dim> &initial_temperature_field);

    const FiniteElement<dim> &get_fe() const;

    const DoFHandler<dim>    &get_dof_handler() const;
    const ConstraintMatrix   &get_constraints() const;

private:
    void setup_dofs();

    void setup_temperature_matrix
    (const IndexSet &locally_owned_dofs,
     const IndexSet &locally_relevant_dofs);

    void assemble_temperature_system();

    void build_temperature_preconditioner();

    void solve_temperature_system();

    void temperature_step();

    // reference to parameters
    TemperatureParameters          &parameters;

    // reference to common triangulation
    parallel::distributed::Triangulation<dim>   &triangulation;

    // reference to common mapping
    const MappingQ<dim>&mapping;

    // reference to time stepper
    IMEXTimeStepping   &timestepper;

    // copy of equation coefficient
    const double        equation_coefficient;

    // parallel output
    ConditionalOStream  pcout;

    // pointer to monitor of computing times
    std::shared_ptr<TimerOutput> computing_timer;

    // temperature FiniteElement and DoFHandler
    const FE_Q<dim>     temperature_fe;
    DoFHandler<dim>     temperature_dof_handler;

    // temperature matrices
    IndexSet            locally_owned_temperature_dofs;
    IndexSet            locally_relevant_temperature_dofs;

    ConstraintMatrix    temperature_constraints;

    LA::SparseMatrix    temperature_matrix;
    LA::SparseMatrix    temperature_mass_matrix;
    LA::SparseMatrix    temperature_stiffness_matrix;

    // vectors of temperature part
    LA::Vector          temperature_solution;
    LA::Vector          old_temperature_solution;
    LA::Vector          old_old_temperature_solution;
    LA::Vector          temperature_rhs;

    // pointers to preconditioners
    std::shared_ptr<LA::PreconditionSSOR> preconditioner_temperature;

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_temperature_matrices = true,
            rebuild_temperature_preconditioner = true;

    // work stream methods for temperature assembly
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
};

}  // namespace adsolic

#endif /* INCLUDE_ADSOLIC_TEMPERATURE_SOLVER_H_ */
