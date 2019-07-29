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
verbose(false)
{}

ConvectionDiffusionParameters::ConvectionDiffusionParameters(const std::string &parameter_filename)
:
ConvectionDiffusionParameters()
{
    ParameterHandler prm;
    declare_parameters(prm);
    linear_solver_parameters.declare_parameters(prm);

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
    linear_solver_parameters.parse_parameters(prm);
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

    verbose = prm.get_bool("verbose");

    prm.leave_subsection();
}

template<typename Stream>
void ConvectionDiffusionParameters::write(Stream &stream) const
{
    stream << "Convection diffusion parameters" << std::endl
           << "   equation_coefficient: " << equation_coefficient << std::endl
           << "   fe_degree: " << fe_degree << std::endl
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
SolverBase<dim,LA::Vector>
(triangulation_in,
 mapping_in,
 timestepper_in,
 external_timer),
parameters(parameters_),
equation_coefficient(parameters.equation_coefficient),
fe(parameters.fe_degree)
{
    if (boundary_descriptor.get() != 0)
        boundary_conditions = boundary_descriptor;
    else
        boundary_conditions = std::make_shared<const BC::ScalarBoundaryConditions<dim>>();
}

template<int dim>
const FiniteElement<dim> &
ConvectionDiffusionSolver<dim>::get_fe() const
{
    return fe;
}

template<int dim>
void ConvectionDiffusionSolver<dim>::advance_in_time()
{
    if (parameters.verbose)
        this->pcout << "   Convection diffusion step..." << std::endl;

    if (convection_function.get() != 0)
    {
        std::stringstream ss;
        ss << "Time of the ConvectionFunction does not "
              "the match time of the IMEXTimeStepper." << std::endl
           <<  "ConvectionFunction::get_time() returns " << convection_function->get_time()
           << ", which is not equal to " << this->timestepper.now()
           << ", which is returned by IMEXTimeStepper::now()" << std::endl;

        Assert(this->timestepper.now() == convection_function->get_time(),
               ExcMessage(ss.str().c_str()));
    }

    this->computing_timer->enter_subsection("Convect.-Diff.");

    // extrapolate from old solutions
    this->extrapolate_solution();

    // assemble right-hand side (and system if necessary)
    assemble_system();

    // rebuild preconditioner for diffusion step
    build_preconditioner();

    // solve linear system
    solve_linear_system();

    // update solution vectors
    this->advance_solution();

    this->computing_timer->leave_subsection();
}

// explicit instantiation
template void ConvectionDiffusionParameters::write(std::ostream &) const;
template void ConvectionDiffusionParameters::write(ConditionalOStream &) const;

template class ConvectionDiffusionSolver<2>;
template class ConvectionDiffusionSolver<3>;

}  // namespace adsolic


