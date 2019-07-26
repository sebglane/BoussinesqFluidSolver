/*
 * temperature_solver.cc
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */

#include <adsolic/convection_diffusion_solver.h>

namespace adsolic
{

ConvectionDiffusionParameters::ConvectionDiffusionParameters()
:
equation_coefficient(1.0),
fe_degree(1),
rel_tol(1e-6),
abs_tol(1e-9),
n_max_iter(200),
verbose(false)
{}

ConvectionDiffusionParameters::ConvectionDiffusionParameters(const std::string &parameter_filename)
:
ConvectionDiffusionParameters()
{
    ParameterHandler prm;
    declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename.c_str());

    if (!parameter_file)
    {
        parameter_file.close();

        std::ostringstream message;
        message << "Input parameter file <"
                << parameter_filename << "> not found. Creating a"
                << std::endl
                << "template file of the same name."
                << std::endl;

        std::ofstream parameter_out(parameter_filename.c_str());
        prm.print_parameters(parameter_out,
                             ParameterHandler::OutputStyle::Text);

        AssertThrow(false, ExcMessage(message.str().c_str()));
    }

    prm.parse_input(parameter_file);

    parse_parameters(prm);
}

void ConvectionDiffusionParameters::declare_parameters
(ParameterHandler &prm)
{
    prm.enter_subsection("Convection diffusion parameters");

    prm.declare_entry("equation_coefficient",
            "1.0",
            Patterns::Double(0.),
            "Diffusion coefficient.");

    prm.declare_entry("fe_degree",
            "1",
            Patterns::Integer(1,5),
            "Polynomial degree of the finite element discretization.");

    prm.declare_entry("tol_rel",
            "1e-6",
            Patterns::Double(1.0e-15,1.0),
            "Relative tolerance of the linear solver.");

    prm.declare_entry("tol_abs",
            "1e-9",
            Patterns::Double(1.0e-15,1.0),
            "Absolute tolerance of the linear solver.");

    prm.declare_entry("n_max_iter",
            "200",
            Patterns::Integer(1,1000),
            "Maximum number of iterations of the linear solver.");

    prm.declare_entry("verbose",
            "false",
            Patterns::Bool(),
            "Flag to activate verbosity.");

    prm.leave_subsection();
}

void ConvectionDiffusionParameters::parse_parameters
(ParameterHandler &prm)
{
    prm.enter_subsection("Convection diffusion parameters");

    equation_coefficient = prm.get_double("equation_coefficient");
    Assert(equation_coefficient > 0,
           ExcLowerRangeType<double>(equation_coefficient, 0));

    fe_degree = prm.get_integer("fe_degree");
    Assert(fe_degree > 0, ExcLowerRange(fe_degree, 0));

    rel_tol = prm.get_double("tol_rel");
    Assert(rel_tol > 0, ExcLowerRangeType<double>(rel_tol, 0));

    abs_tol = prm.get_double("tol_abs");
    Assert(abs_tol > 0, ExcLowerRangeType<double>(abs_tol, 0));

    n_max_iter = prm.get_integer("n_max_iter");
    Assert(n_max_iter > 0, ExcLowerRange(n_max_iter, 0));

    verbose = prm.get_bool("verbose");

    prm.leave_subsection();
}

template<typename Stream>
void ConvectionDiffusionParameters::write(Stream &stream) const
{
    stream << "Convection diffusion parameters" << std::endl
           << "   equation_coefficient: " << equation_coefficient << std::endl
           << "   fe_degree: " << fe_degree << std::endl
           << "   rel_tol: " << rel_tol << std::endl
           << "   abs_tol: " << abs_tol << std::endl
           << "   n_max_iter: " << n_max_iter << std::endl
           << "   verbose: " << (verbose? "true": "false") << std::endl;
}



template<int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(const ConvectionDiffusionParameters    &parameters_,
 const parallel::distributed::Triangulation<dim>  &triangulation_in,
 const MappingQ<dim>    &mapping_in,
 const IMEXTimeStepping &timestepper_in,
 const std::shared_ptr<const BC::ScalarBoundaryConditions<dim>> boundary_descriptor,
 const std::shared_ptr<TimerOutput> external_timer)
:
parameters(parameters_),
triangulation(triangulation_in),
mapping(mapping_in),
timestepper(timestepper_in),
equation_coefficient(parameters.equation_coefficient),
pcout(std::cout,
      Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0),
fe(parameters.fe_degree),
dof_handler(triangulation)
{
    if (boundary_descriptor.get() != 0)
        boundary_conditions = boundary_descriptor;
    else
        boundary_conditions = std::make_shared<const BC::ScalarBoundaryConditions<dim>>();

    if (external_timer.get() != 0)
        computing_timer  = external_timer;
    else
        computing_timer.reset(new TimerOutput(pcout,
                                              TimerOutput::summary,
                                              TimerOutput::wall_times));
}

template<int dim>
const FiniteElement<dim> &
ConvectionDiffusionSolver<dim>::get_fe() const
{
    return fe;
}

template<int dim>
unsigned int
ConvectionDiffusionSolver<dim>::fe_degree() const
{
    return fe.degree;
}

template<int dim>
types::global_dof_index
ConvectionDiffusionSolver<dim>::n_dofs() const
{
    return dof_handler.n_dofs();
}

template<int dim>
const DoFHandler<dim> &
ConvectionDiffusionSolver<dim>::get_dof_handler() const
{
    return dof_handler;
}

template<int dim>
const ConstraintMatrix &
ConvectionDiffusionSolver<dim>::get_constraints() const
{
    return constraints;
}

template<int dim>
const LA::Vector &
ConvectionDiffusionSolver<dim>::get_solution() const
{
    return solution;
}

template<int dim>
void ConvectionDiffusionSolver<dim>::extrapolate_solution()
{
    LA::Vector::iterator
    sol = solution.begin(),
    end_sol = solution.end();
    LA::Vector::const_iterator
    old_sol = old_solution.begin(),
    old_old_sol = old_old_solution.begin();

    // extrapolate solution from old states
    for (; sol!=end_sol; ++sol, ++old_sol, ++old_old_sol)
        *sol = timestepper.extrapolate(*old_sol, *old_old_sol);
}

template<int dim>
void ConvectionDiffusionSolver<dim>::advance_solution()
{
    LA::Vector::const_iterator
    sol = solution.begin(),
    end_sol = solution.end();

    LA::Vector::iterator
    old_sol = old_solution.begin(),
    old_old_sol = old_old_solution.begin();

    // copy solutions
    for (; sol!=end_sol; ++sol, ++old_sol, ++old_old_sol)
    {
        *old_old_sol = *old_sol;
        *old_sol = *sol;
    }
}

template<int dim>
void ConvectionDiffusionSolver<dim>::advance_in_time()
{
    if (parameters.verbose)
        pcout << "   Convection diffusion step..." << std::endl;

    if (convection_function.get() != 0)
    {
        std::stringstream ss;
        ss << "Time of the ConvectionFunction does not "
              "the match time of the IMEXTimeStepper." << std::endl
           <<  "ConvectionFunction::get_time() returns " << convection_function->get_time()
           << ", which is not equal to " << timestepper.now()
           << ", which is returned by IMEXTimeStepper::now()" << std::endl;

        Assert(timestepper.now() == convection_function->get_time(),
               ExcMessage(ss.str().c_str()));
    }

    computing_timer->enter_subsection("Convect.-Diff.");

    // extrapolate from old solutions
    extrapolate_solution();

    // assemble right-hand side (and system if necessary)
    assemble_system();

    // rebuild preconditioner for diffusion step
    build_preconditioner();

    // solve linear system
    solve_linear_system();

    // update solution vectors
    advance_solution();

    computing_timer->leave_subsection();
}

// explicit instantiation
template void ConvectionDiffusionParameters::write(std::ostream &) const;
template void ConvectionDiffusionParameters::write(ConditionalOStream &) const;

template class ConvectionDiffusionSolver<2>;
template class ConvectionDiffusionSolver<3>;

}  // namespace adsolic


