#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace BuoyantFluid {

using namespace dealii;

/*
 *
 * enumeration for boundary identifiers
 *
 */
enum BoundaryIds
{
    ICB,
    CMB,
};


namespace TimeStepping {
/*
 *
 * enumeration for the type of the IMEX time stepping type
 *
 */
enum IMEXType
{
    CNAB,
    MCNAB,
    CNLF,
    SBDF
};

/*
 *
 * These functions return the coefficients of the variable time step IMEX stepping
 * schemes.
 *
 */
class IMEXCoefficients
{
public:

    IMEXCoefficients(const IMEXType &type_);

    std::vector<double> alpha(const double omega);
    std::vector<double> beta(const double omega);
    std::vector<double> gamma(const double omega);

    void write(std::ostream &stream) const;

private:

    void compute_alpha();
    void compute_beta();
    void compute_gamma();

    const IMEXType      type;

    std::vector<double> alpha_;
    std::vector<double> beta_;
    std::vector<double> gamma_;

    bool    update_alpha;
    bool    update_beta;
    bool    update_gamma;

    double      omega;

};

IMEXCoefficients::IMEXCoefficients(const IMEXType &type_)
:
type(type_),
alpha_(3,0),
beta_(2,0),
gamma_(3,0),
update_alpha(true),
update_beta(true),
update_gamma(true),
omega(0)
{}

std::vector<double> IMEXCoefficients::alpha(const double    timestep_ratio)
{
    if (timestep_ratio != omega)
    {
        omega = timestep_ratio;

        update_alpha = true;
        update_beta = true;
        update_gamma = true;

    }

    compute_alpha();

    return alpha_;
}

std::vector<double> IMEXCoefficients::beta(const double timestep_ratio)
{
    if (timestep_ratio != omega)
    {
        omega = timestep_ratio;

        update_alpha = true;
        update_beta = true;
        update_gamma = true;

    }

    compute_beta();

    return beta_;
}

std::vector<double> IMEXCoefficients::gamma(const double    timestep_ratio)
{
    if (timestep_ratio != omega)
    {
        omega = timestep_ratio;

        update_alpha = true;
        update_beta = true;
        update_gamma = true;

    }

    compute_gamma();

    return gamma_;
}

void IMEXCoefficients::compute_alpha()
{
    if (!update_alpha)
        return;

    if (type == IMEXType::SBDF)
    {
        alpha_[0] = (1. + 2. * omega) / (1. + omega);
        alpha_[1] = -(1. + omega);
        alpha_[2] = (omega * omega) / (1. + omega);
    }
    else if (type == IMEXType::CNAB || type == IMEXType::MCNAB)
    {
        alpha_[0] = 1.0;
        alpha_[1] = -1.0;
    }
    else if (type == IMEXType::CNLF)
    {
        alpha_[0] = 1. / (1. + omega);
        alpha_[1] = omega - 1.;
        alpha_[2] = -(omega * omega) / (1. + omega);
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }
    update_alpha = false;
}

void IMEXCoefficients::compute_beta()
{
    if (!update_beta)
        return;

    if (type == IMEXType::SBDF)
    {
        beta_[0] = (1. + omega);
        beta_[1] = -omega;
    }
    else if (type == IMEXType::CNAB ||  type == IMEXType::MCNAB)
    {
        beta_[0] = (1. + 0.5 * omega);
        beta_[1] = -0.5 * omega;
    }
    else if (type == IMEXType::CNLF)
    {
        beta_[0] = 1.;
        beta_[1] = 0.;
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }

    update_beta = false;

}
void IMEXCoefficients::compute_gamma()
{
    if (!update_gamma)
        return;

    if (type == IMEXType::SBDF)
    {
        gamma_[0] = 1.0;
        gamma_[1] = 0.0;
    }
    else if (type == IMEXType::CNAB)
    {
        gamma_[0] = 0.5;
        gamma_[1] = 0.5;
    }
    else if (type == IMEXType::MCNAB)
    {
        gamma_[0] = (8. * omega + 1.)/ (16. * omega);
        gamma_[1] = (7. * omega - 1.)/ (16. * omega);
        gamma_[2] = omega / (16. * omega);
    }
    else if (type == IMEXType::CNLF)
    {
        gamma_[0] = 0.5 / omega;
        gamma_[1] = 0.5 * (1. - 1./omega);
        gamma_[2] = 0.5;
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }
    update_gamma = false;
}

void IMEXCoefficients::write(std::ostream &stream) const
{
    stream << std::endl
           << "+-----------+----------+----------+----------+\n"
           << "|   Index   |    n+1   |    n     |    n-1   |\n"
           << "+-----------+----------+----------+----------+\n"
           << "|   alpha   | ";
    for (const auto it: alpha_)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "|   beta    |    0     | ";
    for (const auto it: beta_)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "|   gamma   | ";
    for (const auto it: gamma_)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }
    stream << std::endl
            << "+-----------+----------+----------+----------+\n";
}

}  // namespace TimeStepping


namespace EquationData {

template<int dim>
class TemperatureInitialValues : public Function<dim>
{
public:
    TemperatureInitialValues(const double inner_radius,
                             const double outer_radius,
                             const double inner_temperature,
                             const double outer_temperature);

    virtual double value(const Point<dim>   &point,
                         const unsigned int component = 0) const;

    virtual void vector_value(const Point<dim>  &point,
                              Vector<double>    &values) const;

private:
    const double ri;
    const double ro;
    const double Ti;
    const double To;
};

template <int dim>
TemperatureInitialValues<dim>::TemperatureInitialValues(
        const double inner_radius,
        const double outer_radius,
        const double inner_temperature,
        const double outer_temperature)
:
Function<dim>(1),
ri(inner_radius),
ro(outer_radius),
Ti(inner_temperature),
To(outer_temperature)
{
    Assert(To < Ti, ExcLowerRangeType<double>(To, Ti));
    Assert(ri < ro, ExcLowerRangeType<double>(ri, ro));
}

template <int dim>
double TemperatureInitialValues<dim>::value(
        const Point<dim>    &point,
        const unsigned int component) const
{
    const double radius = point.distance(Point<dim>());
    const double value = Ti + (To - Ti) / (ro - ri) * (radius - ri);
    return value;
}

template <int dim>
void TemperatureInitialValues<dim>::vector_value(
        const Point<dim> &p,
        Vector<double>   &values) const
{
    for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = value(p, c);
}

}  // namespace EquationData



namespace Assembly {

namespace Scratch {

template<int dim>
struct TemperatureMatrix
{
    TemperatureMatrix(const FiniteElement<dim> &temperature_fe,
                      const Mapping<dim>       &mapping,
                      const Quadrature<dim>    &temperature_quadrature);

    TemperatureMatrix(const TemperatureMatrix<dim>  &scratch);

    FEValues<dim>               temperature_fe_values;

    std::vector<double>         phi_T;
    std::vector<Tensor<1,dim>>  grad_phi_T;
};


template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(
        const FiniteElement<dim> &temperature_fe,
        const Mapping<dim>       &mapping,
        const Quadrature<dim>    &temperature_quadrature)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      update_values|
                      update_gradients|
                      update_JxW_values),
phi_T(temperature_fe.dofs_per_cell),
grad_phi_T(temperature_fe.dofs_per_cell)
{}

template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(const TemperatureMatrix &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_T(scratch.phi_T),
grad_phi_T(scratch.grad_phi_T)
{}

template<int dim>
struct TemperatureRightHandSide
{
    TemperatureRightHandSide(
            const FiniteElement<dim> &temperature_fe,
            const Mapping<dim>       &mapping,
            const Quadrature<dim>    &temperature_quadrature,
            const UpdateFlags         temperature_update_flags);

    TemperatureRightHandSide(const TemperatureRightHandSide<dim> &scratch);

    FEValues<dim>               temperature_fe_values;
    std::vector<double>         phi_T;
    std::vector<Tensor<1,dim>>  grad_phi_T;
    std::vector<double>         old_temperature_values;
    std::vector<double>         old_old_temperature_values;
    std::vector<Tensor<1,dim>>  old_temperature_gradients;
    std::vector<Tensor<1,dim>>  old_old_temperature_gradients;
};

template<int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
        const FiniteElement<dim>    &temperature_fe,
        const Mapping<dim>          &mapping,
        const Quadrature<dim>       &temperature_quadrature,
        const UpdateFlags            temperature_update_flags)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      temperature_update_flags),
phi_T(temperature_fe.dofs_per_cell),
grad_phi_T(temperature_fe.dofs_per_cell),
old_temperature_values(temperature_quadrature.size()),
old_old_temperature_values(temperature_quadrature.size()),
old_temperature_gradients(temperature_quadrature.size()),
old_old_temperature_gradients(temperature_quadrature.size())
{}

template<int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
        const TemperatureRightHandSide<dim> &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_T(scratch.phi_T),
grad_phi_T(scratch.grad_phi_T),
old_temperature_values(scratch.old_temperature_values),
old_old_temperature_values(scratch.old_old_temperature_values),
old_temperature_gradients(scratch.old_temperature_gradients),
old_old_temperature_gradients(scratch.old_old_temperature_gradients)
{}

}  // namespace Scratch

namespace CopyData {

template <int dim>
struct TemperatureMatrix
{
    TemperatureMatrix(const FiniteElement<dim> &temperature_fe);
    TemperatureMatrix(const TemperatureMatrix<dim> &data);

    FullMatrix<double>                      local_mass_matrix;
    FullMatrix<double>                      local_stiffness_matrix;

    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(const FiniteElement<dim> &temperature_fe)
:
local_mass_matrix(temperature_fe.dofs_per_cell),
local_stiffness_matrix(temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(const TemperatureMatrix<dim> &data)
:
local_mass_matrix(data.local_mass_matrix),
local_stiffness_matrix(data.local_stiffness_matrix),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
struct TemperatureRightHandSide
{
    TemperatureRightHandSide(const FiniteElement<dim>               &temperature_fe);
    TemperatureRightHandSide(const TemperatureRightHandSide<dim>    &data);

    Vector<double>                          local_rhs;
    FullMatrix<double>                      matrix_for_bc;
    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
    const FiniteElement<dim> &temperature_fe)
:
local_rhs(temperature_fe.dofs_per_cell),
matrix_for_bc(temperature_fe.dofs_per_cell,
              temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
    const TemperatureRightHandSide<dim> &data)
:
local_rhs(data.local_rhs),
matrix_for_bc(data.matrix_for_bc),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace CopyData

}  // namespace Assembly


/*
 *
 * templated class for solving the Boussinesq problem
 * using Q(p+1)/Q(p)-elements
 *
 */
template <int dim>
class HeatConductionProblem
{
public:
    struct Parameters;

    HeatConductionProblem(Parameters &parameters_);

    void run();

private:
    void make_grid();

    void setup_dofs();

    void setup_temperature_matrices(const types::global_dof_index n_temperature_dofs);

    void assemble_temperature_system();

    void build_temperature_preconditioner();

    void solve();

    double compute_rms_values() const;

    void output_results() const;

    void refine_mesh();

    Parameters                      &parameters;

    TimeStepping::IMEXCoefficients  imex_coefficients;

    Triangulation<dim>              triangulation;

    const MappingQ<dim>             mapping;

    const FE_Q<dim>                 temperature_fe;
    DoFHandler<dim>                 temperature_dof_handler;

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

    // preconditioner types
    typedef PreconditionJacobi<SparseMatrix<double>> PreconditionerTypeT;

    // pointers to preconditioners
    std::shared_ptr<PreconditionerTypeT>      preconditioner_T;


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

        // physics parameters
        double aspect_ratio;
        double Pr;
        double Ra;

        bool    rotation;

        // runtime parameters
        bool    workstream_assembly;

        // time stepping parameters
        TimeStepping::IMEXType  imex_scheme;

        unsigned int    n_steps;

        double          timestep;

        // discretization parameters
        unsigned int temperature_degree;

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

    // flags for rebuilding matrices and preconditioners
    bool    rebuild_temperature_matrix = true;
    bool    rebuild_temperature_preconditioner = true;

    // working stream methods for temperature assembly
    void local_assemble_temperature_matrix(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::TemperatureMatrix<dim> &scratch,
            Assembly::CopyData::TemperatureMatrix<dim> &data);
    void copy_local_to_global_temperature_matrix(
            const Assembly::CopyData::TemperatureMatrix<dim> &data);

    void local_assemble_temperature_rhs(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::TemperatureRightHandSide<dim> &scratch,
            Assembly::CopyData::TemperatureRightHandSide<dim> &data);
    void copy_local_to_global_temperature_rhs(
            const Assembly::CopyData::TemperatureRightHandSide<dim> &data);
};

template<int dim>
HeatConductionProblem<dim>::HeatConductionProblem(Parameters &parameters_)
:
parameters(parameters_),
imex_coefficients(parameters.imex_scheme),
triangulation(),
mapping(4),
// temperature part
temperature_fe(parameters.temperature_degree),
temperature_dof_handler(triangulation),
// coefficients
equation_coefficients{(parameters.rotation ? 1.0 / parameters.Pr : 1.0 / std::sqrt(parameters.Ra * parameters.Pr))},
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
// time stepping
timestep(parameters.timestep),
old_timestep(0.5 * parameters.timestep)
{
    std::cout << "Heat conduction solver by S. Glane\n"
              << "This program solves the heat conduction equation.\n"
              << "The governing equation is\n\n"
              << "\t-- Heat conduction equation:\n\t\tdT/dt = C div(grad(T)).\n\n"
              << "The coefficient C depends on the normalization as follows.\n\n";

    // generate a nice table
    std::cout << "\n\n"
              << "+-------------------+-------------------+\n"
              << "|       case        |    C              |\n"
              << "+-------------------+-------------------+\n"
              << "| Non-rotating case | 1 / sqrt(Ra * Pr) |\n"
              << "| Rotating case     | 1 /  Pr           |\n"
              << "+-------------------+-------------------+\n";

    std::cout << std::endl << "You have chosen ";

    std::stringstream ss;
    ss << "+----------+----------+----------+\n"
       << "|    Ra    |    Pr    |    C     |\n";

    if (parameters.rotation)
    {
        std::cout << "the rotating case with the following parameters: "
                  << std::endl;
    }
    else
    {
        std::cout << "the non-rotating case with the following parameters: "
                  << std::endl;
    }
    ss << "| ";
    ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ra;
    ss << " | ";
    ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pr;
    ss << " | ";


    for (unsigned int n=0; n<1; ++n)
    {
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << equation_coefficients[n];
        ss << " | ";
    }

    ss << "\n+----------+----------+----------+\n";

    std::cout << std::endl << ss.str() << std::endl;

    std::cout << std::endl << std::flush << std::fixed;
}

template<int dim>
void HeatConductionProblem<dim>::Parameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Runtime parameters");
    {
        prm.declare_entry("workstream_assembly",
                "false",
                Patterns::Bool(),
                "Use multi-threading for assembly");
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
        prm.declare_entry("p_degree_temperature",
                "1",
                Patterns::Integer(1,2),
                "Polynomial degree of the temperature discretization.");

        prm.declare_entry("aspect_ratio",
                "0.35",
                Patterns::Double(0.,1.),
                "Ratio of inner to outer radius");

        prm.enter_subsection("Refinement parameters");
        {
            prm.declare_entry("n_global_refinements",
                    "1",
                    Patterns::Integer(),
                    "Number of initial global refinements.");

            prm.declare_entry("n_initial_refinements",
                    "1",
                    Patterns::Integer(),
                    "Number of initial refinements based on the initial condition.");

            prm.declare_entry("n_boundary_refinements",
                    "1",
                    Patterns::Integer(),
                    "Number of initial boundary refinements.");

            prm.declare_entry("n_max_levels",
                    "1",
                    Patterns::Integer(),
                    "Total of number of refinements allowed during the run.");

            prm.declare_entry("refinement_freq",
                    "100",
                    Patterns::Integer(),
                    "Refinement frequency.");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    {
        prm.declare_entry("rotating_case",
                "true",
                Patterns::Bool(),
                "Turn rotation on or off");

        prm.declare_entry("Pr",
                "1.0",
                Patterns::Double(),
                "Prandtl number of the fluid");

        prm.declare_entry("Ra",
                "1.0e5",
                Patterns::Double(),
                "Rayleigh number of the flow");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        prm.declare_entry("n_steps",
                "1000",
                Patterns::Integer(),
                "Maximum number of iteration. That is the maximum number of time steps.");

        prm.declare_entry("time_step",
                "1e-4",
                Patterns::Double(),
                "time step.");

        // TODO: move to logging
        prm.declare_entry("output_freq",
                "10",
                Patterns::Integer(),
                "Output frequency.");

        prm.declare_entry("time_stepping_scheme",
                        "CNAB",
                        Patterns::Selection("CNAB|MCNAB|CNLF|SBDF"),
                        "Time stepping scheme applied.");
    }
    prm.leave_subsection();

}

template<int dim>
void HeatConductionProblem<dim>::Parameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Runtime parameters");
    {
        workstream_assembly = prm.get_bool("workstream_assembly");
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
        temperature_degree = prm.get_integer("p_degree_temperature");

        aspect_ratio = prm.get_double("aspect_ratio");

        Assert(aspect_ratio < 1., ExcLowerRangeType<double>(aspect_ratio, 1.0));

        prm.enter_subsection("Refinement parameters");
        {

            if (n_max_levels < n_global_refinements + n_boundary_refinements + n_initial_refinements)
            {
                std::ostringstream message;
                message << "Inconsistency in parameter file in definition of maximum number of levels."
                        << std::endl
                        << "maximum number of levels is: "
                        << n_max_levels
                        << ", which is less than the sum of initial global and boundary refinements,"
                        << std::endl
                        << " which is "
                        << n_global_refinements + n_boundary_refinements + n_initial_refinements
                        << " for your parameter file."
                        << std::endl;

                AssertThrow(false, ExcMessage(message.str().c_str()));
            }

            n_global_refinements = prm.get_integer("n_global_refinements");
            n_initial_refinements = prm.get_integer("n_initial_refinements");
            n_boundary_refinements = prm.get_integer("n_boundary_refinements");

            n_max_levels = prm.get_integer("n_max_levels");

            refinement_frequency = prm.get_integer("refinement_freq");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    {
        rotation = prm.get_bool("rotating_case");
        Ra = prm.get_double("Ra");
        Pr = prm.get_double("Pr");

    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        n_steps = prm.get_integer("n_steps");
        Assert(n_steps > 0, ExcLowerRange(n_steps,0));

        timestep = prm.get_double("time_step");

        // TODO: move to logging
        output_frequency = prm.get_integer("output_freq");

        std::string imex_type_str;
        imex_type_str = prm.get("time_stepping_scheme");

        if (imex_type_str == "CNAB")
            imex_scheme = TimeStepping::IMEXType::CNAB;
        else if (imex_type_str == "MCNAB")
            imex_scheme = TimeStepping::IMEXType::MCNAB;
        else if (imex_type_str == "CNLF")
            imex_scheme = TimeStepping::IMEXType::CNLF;
        else if (imex_type_str == "SBDF")
            imex_scheme = TimeStepping::IMEXType::SBDF;
    }
    prm.leave_subsection();
}


template<int dim>
void HeatConductionProblem<dim>::make_grid()
{
    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    const Point<dim> center;
    const double ri = parameters.aspect_ratio;
    const double ro = 1.0;

    GridGenerator::hyper_shell(triangulation, center, ri, ro, (dim==3) ? 96 : 12);

    std::cout << "   Number of initial cells: "
              << triangulation.n_active_cells()
              << std::endl;

    static SphericalManifold<dim>       manifold(center);

    triangulation.set_all_manifold_ids(0);
    triangulation.set_all_manifold_ids_on_boundary(1);

    triangulation.set_manifold (0, manifold);
    triangulation.set_manifold (1, manifold);

    // setting boundary ids on coarsest grid
    const double tol = 1e-12;
    for(auto cell: triangulation.active_cell_iterators())
      if (cell->at_boundary())
          for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face(f)->at_boundary())
              {
                  std::vector<double> dist(GeometryInfo<dim>::vertices_per_face);
                  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                      dist[v] = cell->face(f)->vertex(v).distance(center);
                  if (std::all_of(dist.begin(), dist.end(),
                          [&ri,&tol](double d){return std::abs(d - ri) < tol;}))
                      cell->face(f)->set_boundary_id(BoundaryIds::ICB);
                  if (std::all_of(dist.begin(), dist.end(),
                          [&ro,&tol](double d){return std::abs(d - ro) < tol;}))
                      cell->face(f)->set_boundary_id(BoundaryIds::CMB);
              }

    // initial global refinements
    if (parameters.n_global_refinements > 0)
    {
        triangulation.refine_global(parameters.n_global_refinements);
        std::cout << "      Number of cells after "
                  << parameters.n_global_refinements
                  << " global refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    // initial boundary refinements
    if (parameters.n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<parameters.n_boundary_refinements; ++step)
        {
            for (auto cell: triangulation.active_cell_iterators())
                if (cell->at_boundary())
                    cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells after "
                  << parameters.n_boundary_refinements
                  << " boundary refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }
}

template<int dim>
void HeatConductionProblem<dim>::setup_dofs()
{
    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

    // temperature part
    temperature_dof_handler.distribute_dofs(temperature_fe);

    DoFRenumbering::boost::king_ordering(temperature_dof_handler);

    // IO
    const unsigned int n_temperature_dofs = temperature_dof_handler.n_dofs();

    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "      Number of temperature degrees of freedom: "
              << n_temperature_dofs
              << std::endl;
    // temperature constraints
    {
        TimerOutput::Scope timer_section(computing_timer, "temperature constraints");
        temperature_constraints.clear();

        DoFTools::make_hanging_node_constraints(
                temperature_dof_handler,
                temperature_constraints);

        const Functions::ConstantFunction<dim> icb_temperature(0.5);
        const Functions::ConstantFunction<dim> cmb_temperature(-0.5);

        const std::map<typename types::boundary_id, const Function<dim>*>
        temperature_boundary_values = {{BoundaryIds::ICB, &icb_temperature},
                                       {BoundaryIds::CMB, &cmb_temperature}};

        VectorTools::interpolate_boundary_values(
                temperature_dof_handler,
                temperature_boundary_values,
                temperature_constraints);

        temperature_constraints.close();
    }

    // temperature matrix and vector setup
    setup_temperature_matrices(n_temperature_dofs);

    temperature_solution.reinit(n_temperature_dofs);
    old_temperature_solution.reinit(n_temperature_dofs);
    old_old_temperature_solution.reinit(n_temperature_dofs);

    temperature_rhs.reinit(n_temperature_dofs);
}


template<int dim>
void HeatConductionProblem<dim>::setup_temperature_matrices(const types::global_dof_index n_temperature_dofs)
{
    preconditioner_T.reset();

    temperature_matrix.clear();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();

    DynamicSparsityPattern dsp(n_temperature_dofs, n_temperature_dofs);

    DoFTools::make_sparsity_pattern(temperature_dof_handler,
                                    dsp,
                                    temperature_constraints);

    temperature_sparsity_pattern.copy_from(dsp);

    temperature_matrix.reinit(temperature_sparsity_pattern);
    temperature_mass_matrix.reinit(temperature_sparsity_pattern);
    temperature_stiffness_matrix.reinit(temperature_sparsity_pattern);

    rebuild_temperature_matrix = true;
}

template <int dim>
void HeatConductionProblem<dim>::local_assemble_temperature_matrix(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::TemperatureMatrix<dim> &scratch,
        Assembly::CopyData::TemperatureMatrix<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    scratch.temperature_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.grad_phi_T[k] = scratch.temperature_fe_values.shape_grad(k,q);
            scratch.phi_T[k]      = scratch.temperature_fe_values.shape_value(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += scratch.phi_T[i] * scratch.phi_T[j] * scratch.temperature_fe_values.JxW(q);
                data.local_stiffness_matrix(i,j)
                    += scratch.grad_phi_T[i] * scratch.grad_phi_T[j] * scratch.temperature_fe_values.JxW(q);
            }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_stiffness_matrix(i,j) = data.local_stiffness_matrix(j,i);
        }
}

template<int dim>
void HeatConductionProblem<dim>::copy_local_to_global_temperature_matrix(
        const Assembly::CopyData::TemperatureMatrix<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            temperature_mass_matrix);
    temperature_constraints.distribute_local_to_global(
            data.local_stiffness_matrix,
            data.local_dof_indices,
            temperature_stiffness_matrix);
}


template <int dim>
void HeatConductionProblem<dim>::local_assemble_temperature_rhs(
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        Assembly::Scratch::TemperatureRightHandSide<dim>        &scratch,
        Assembly::CopyData::TemperatureRightHandSide<dim>       &data)
{
    const std::vector<double> alpha = (timestep_number != 0?
                                            imex_coefficients.alpha(timestep/old_timestep):
                                            std::vector<double>({1.0,-1.0,0.0}));
//    const std::vector<double> beta = (timestep_number != 0?
//                                            imex_coefficients.beta(timestep/old_timestep):
//                                            std::vector<double>({1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                            imex_coefficients.gamma(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0,0.0}));

    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    data.matrix_for_bc = 0;
    data.local_rhs = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.temperature_fe_values.reinit (cell);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);
    scratch.temperature_fe_values.get_function_gradients(old_temperature_solution,
                                                         scratch.old_temperature_gradients);
    scratch.temperature_fe_values.get_function_gradients(old_old_temperature_solution,
                                                         scratch.old_old_temperature_gradients);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            scratch.phi_T[i]      = scratch.temperature_fe_values.shape_value(i, q);
            scratch.grad_phi_T[i] = scratch.temperature_fe_values.shape_grad(i, q);
        }

        const double time_derivative_temperature =
                alpha[1] * scratch.old_temperature_values[q]
                    + alpha[2] * scratch.old_temperature_values[q];

        const Tensor<1,dim> linear_term_temperature =
                gamma[1] * scratch.old_temperature_gradients[q]
                    + gamma[2] * scratch.old_old_temperature_gradients[q];

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                    - time_derivative_temperature * scratch.phi_T[i]
                    - timestep * equation_coefficients[0] * linear_term_temperature * scratch.grad_phi_T[i]
                    ) * scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    data.matrix_for_bc(j,i) += (
                                  alpha[0] * scratch.phi_T[i] * scratch.phi_T[j]
                                + gamma[0] * timestep * equation_coefficients[0] * scratch.grad_phi_T[i] * scratch.grad_phi_T[j]
                                ) * scratch.temperature_fe_values.JxW(q);

        }

    }
}

template <int dim>
void HeatConductionProblem<dim>::copy_local_to_global_temperature_rhs(
        const Assembly::CopyData::TemperatureRightHandSide<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            temperature_rhs,
            data.matrix_for_bc);
}

template<int dim>
void HeatConductionProblem<dim>::assemble_temperature_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble temperature system");

    std::cout << "   Assembling temperature system..." << std::endl;

    if (rebuild_temperature_matrix || timestep_number == 1)
        temperature_matrix = 0;
    temperature_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    if (parameters.workstream_assembly == false)
    {
        const std::vector<double> alpha = (timestep_number != 0?
                                            imex_coefficients.alpha(timestep/old_timestep):
                                            std::vector<double>({1.0,-1.0,0.0}));
//      const std::vector<double> beta_ = (timestep_number != 0?
//                                                  imex_coefficients.beta(timestep/old_timestep):
//                                                  std::vector<double>({1.0,0.0}));
        const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));

        FEValues<dim>     temperature_fe_values(mapping,
                                                temperature_fe,
                                                quadrature_formula,
                                                update_values|
                                                update_gradients|
                                                update_quadrature_points|
                                                update_JxW_values);

        const unsigned int   dofs_per_cell   = temperature_fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();

        Vector<double>       local_rhs(dofs_per_cell);
        FullMatrix<double>   local_matrix(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double>         old_temperature_values(n_q_points);
        std::vector<Tensor<1,dim> > old_temperature_gradients(n_q_points);
        std::vector<double>         old_old_temperature_values(n_q_points);
        std::vector<Tensor<1,dim> > old_old_temperature_gradients(n_q_points);


        std::vector<double>         phi_T(dofs_per_cell);
        std::vector<Tensor<1,dim> > grad_phi_T(dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
        cell = temperature_dof_handler.begin_active(),
        endc = temperature_dof_handler.end();
        for (; cell!= endc; ++cell)
        {

            local_matrix = 0;
            local_rhs = 0;

            temperature_fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            temperature_fe_values.get_function_values(old_temperature_solution,
                                                      old_temperature_values);
            temperature_fe_values.get_function_gradients(old_temperature_solution,
                                                         old_temperature_gradients);
            temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      old_old_temperature_values);
            temperature_fe_values.get_function_gradients(old_old_temperature_solution,
                                                         old_old_temperature_gradients);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    grad_phi_T[k] = temperature_fe_values.shape_grad (k,q);
                    phi_T[k]      = temperature_fe_values.shape_value (k, q);
                }

                const double time_derivative_temperature =
                        alpha[1] * old_temperature_values[q]
                            + alpha[2] * old_temperature_values[q];

                const Tensor<1,dim> linear_term_temperature =
                        gamma[1] * old_temperature_gradients[q]
                            + gamma[2] * old_old_temperature_gradients[q];

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    local_rhs(i) += (
                                - time_derivative_temperature * phi_T[i]
                                - timestep * equation_coefficients[0] * linear_term_temperature * grad_phi_T[i]
                                ) * temperature_fe_values.JxW(q);

                    if (rebuild_temperature_matrix || timestep_number == 1)
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                            local_matrix(i,j) += (
                                      alpha[0] * phi_T[i] * phi_T[j]
                                    + gamma[0] * timestep * equation_coefficients[0] * grad_phi_T[i] * grad_phi_T[j]
                                    ) * temperature_fe_values.JxW(q);
                    else if (temperature_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                            local_matrix(j,i) += (
                                      alpha[0] * phi_T[i] * phi_T[j]
                                    + gamma[0] * timestep * equation_coefficients[0] * grad_phi_T[i] * grad_phi_T[j]
                                    ) * temperature_fe_values.JxW(q);
                }
            }

            if (rebuild_temperature_matrix || timestep_number == 1)
                temperature_constraints.distribute_local_to_global(
                        local_matrix,
                        local_rhs,
                        local_dof_indices,
                        temperature_matrix,
                        temperature_rhs);
            else
                temperature_constraints.distribute_local_to_global(
                        local_rhs,
                        local_dof_indices,
                        temperature_rhs,
                        local_matrix);
        }
        rebuild_temperature_preconditioner = true;
    }
    else
    {
        // assemble temperature matrices
        if (rebuild_temperature_matrix)
        {
            temperature_mass_matrix = 0;
            temperature_stiffness_matrix = 0;

            WorkStream::run(
                    temperature_dof_handler.begin_active(),
                    temperature_dof_handler.end(),
                    std::bind(&HeatConductionProblem<dim>::local_assemble_temperature_matrix,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&HeatConductionProblem<dim>::copy_local_to_global_temperature_matrix,
                              this,
                              std::placeholders::_1),
                    Assembly::Scratch::TemperatureMatrix<dim>(temperature_fe,
                                                              mapping,
                                                              quadrature_formula),
                    Assembly::CopyData::TemperatureMatrix<dim>(temperature_fe));

            const std::vector<double> alpha = (timestep_number != 0?
                                                    imex_coefficients.alpha(timestep/old_timestep):
                                                    std::vector<double>({1.0,-1.0,0.0}));
            const std::vector<double> gamma = (timestep_number != 0?
                                                    imex_coefficients.gamma(timestep/old_timestep):
                                                    std::vector<double>({1.0,0.0,0.0}));

            temperature_matrix.copy_from(temperature_mass_matrix);
            temperature_matrix *= alpha[0];
            temperature_matrix.add(timestep * gamma[0] * equation_coefficients[0],
                                        temperature_stiffness_matrix);
            rebuild_temperature_preconditioner = true;
        }
        else if (timestep_modified|| timestep_number == 1)
        {
            Assert(timestep_number != 0, ExcInternalError());

            const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
            const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

            temperature_matrix.copy_from(temperature_mass_matrix);
            temperature_matrix *= alpha[0];
            temperature_matrix.add(timestep * gamma[0] * equation_coefficients[0],
                                   temperature_stiffness_matrix);

            rebuild_temperature_preconditioner = true;
        }

        // assemble temperature right-hand side
        WorkStream::run(
                temperature_dof_handler.begin_active(),
                temperature_dof_handler.end(),
                std::bind(&HeatConductionProblem<dim>::local_assemble_temperature_rhs,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&HeatConductionProblem<dim>::copy_local_to_global_temperature_rhs,
                          this,
                          std::placeholders::_1),
                Assembly::Scratch::TemperatureRightHandSide<dim>(temperature_fe,
                                                                 mapping,
                                                                 quadrature_formula,
                                                                 update_values|
                                                                 update_gradients|
                                                                 update_JxW_values),
                Assembly::CopyData::TemperatureRightHandSide<dim>(temperature_fe));
    }
    rebuild_temperature_matrix = false;
}

template<int dim>
void HeatConductionProblem<dim>::build_temperature_preconditioner()
{
    if (!rebuild_temperature_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build temperature preconditioner");

    preconditioner_T.reset(new PreconditionerTypeT());

    PreconditionerTypeT::AdditionalData     data;
    data.relaxation = 0.6;

    preconditioner_T->initialize(temperature_matrix,
                                 data);
}


template<int dim>
double HeatConductionProblem<dim>::compute_rms_values() const
{
    const QGauss<dim> temperature_quadrature_formula(parameters.temperature_degree + 1);

    const unsigned int n_q_points = temperature_quadrature_formula.size();

    FEValues<dim> temperature_fe_values(mapping,
                                        temperature_fe,
                                        temperature_quadrature_formula,
                                        update_values|update_JxW_values);

    std::vector<double>     temperature_values(n_q_points);

    double rms_temperature = 0;
    double volume = 0;

    for (auto cell : temperature_dof_handler.active_cell_iterators())
    {
        temperature_fe_values.reinit(cell);
        temperature_fe_values.get_function_values(temperature_solution,
                                                  temperature_values);
        for (unsigned int q=0; q<n_q_points; ++q)
        {
            rms_temperature += temperature_values[q] * temperature_values[q] * temperature_fe_values.JxW(q);
            volume += temperature_fe_values.JxW(q);
        }
    }

    rms_temperature /= volume;

    AssertIsFinite(rms_temperature);
    Assert(rms_temperature >= 0, ExcLowerRangeType<double>(rms_temperature, 0));

    return std::sqrt(rms_temperature);
}


template<int dim>
void HeatConductionProblem<dim>::output_results() const
{
    std::cout << "   Output results..." << std::endl;

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(temperature_dof_handler);
    data_out.add_data_vector(temperature_solution, "temperature");
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(timestep_number, 5) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

template<int dim>
void HeatConductionProblem<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    // error estimation based on temperature
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(temperature_dof_handler,
                                       QGauss<dim-1>(parameters.temperature_degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       temperature_solution,
                                       estimated_error_per_cell);
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.8, 0.1);
    // clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > parameters.n_max_levels)
        for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_max_levels))
            cell->clear_refine_flag();


    // preparing temperature solution transfer
    std::vector<Vector<double>> x_temperature(3);
    x_temperature[0] = temperature_solution;
    x_temperature[1] = old_temperature_solution;
    x_temperature[2] = old_old_temperature_solution;
    SolutionTransfer<dim,Vector<double>> temperature_transfer(temperature_dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    temperature_transfer.prepare_for_coarsening_and_refinement(x_temperature);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of temperature solution
    {
        std::vector<Vector<double>> tmp_temperature(3);
        tmp_temperature[0].reinit(temperature_solution);
        tmp_temperature[1].reinit(temperature_solution);
        tmp_temperature[2].reinit(temperature_solution);
        temperature_transfer.interpolate(x_temperature, tmp_temperature);

        temperature_solution = tmp_temperature[0];
        old_temperature_solution = tmp_temperature[1];
        old_old_temperature_solution = tmp_temperature[2];

        temperature_constraints.distribute(temperature_solution);
        temperature_constraints.distribute(old_temperature_solution);
        temperature_constraints.distribute(old_old_temperature_solution);
    }
}


}  // namespace BuoyantFluid
