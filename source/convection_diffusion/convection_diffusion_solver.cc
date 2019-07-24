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
}

void ConvectionDiffusionParameters::parse_parameters
(ParameterHandler &prm)
{
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
}

template<int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(const ConvectionDiffusionParameters &parameters_,
 parallel::distributed::Triangulation<dim> &triangulation_in,
 const MappingQ<dim>         &mapping_in,
 IMEXTimeStepping      &timestepper_in,
 TensorFunction<1,dim> &advection_function_in,
 std::shared_ptr<BC::ScalarBoundaryConditions<dim>> boundary_descriptor,
 TimerOutput           *external_timer)
:
parameters(parameters_),
triangulation(triangulation_in),
mapping(mapping_in),
timestepper(timestepper_in),
equation_coefficient(parameters.equation_coefficient),
pcout(std::cout,
      Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0),
advection_function(advection_function_in),
fe(parameters.fe_degree),
dof_handler(triangulation)
{
    if (boundary_descriptor.get() != 0)
      this->boundary_conditions = boundary_descriptor;

    // If the timer is not obtained form another class, reset it.
    if (external_timer == 0)
        computing_timer.reset(new TimerOutput(pcout,
                                              TimerOutput::summary,
                                              TimerOutput::wall_times));
    // Otherwise, just set the pointer
    else
        computing_timer.reset(external_timer);
}

template class ConvectionDiffusionSolver<2>;
template class ConvectionDiffusionSolver<3>;

}  // namespace adsolic


