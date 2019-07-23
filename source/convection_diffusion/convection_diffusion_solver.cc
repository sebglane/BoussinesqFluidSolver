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
n_max_iter(200)
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

template<int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(ConvectionDiffusionParameters &parameters_,
 parallel::distributed::Triangulation<dim> &triangulation_in,
 MappingQ<dim>      &mapping_in,
 IMEXTimeStepping   &timestepper_in,
 TimerOutput        *external_timer)
:
parameters(parameters_),
triangulation(triangulation_in),
mapping(mapping_in),
timestepper(timestepper_in),
equation_coefficient(parameters.equation_coefficient),
pcout(std::cout,
      Utilities::MPI::this_mpi_process(triangulation.get_communicator) == 0),
fe(parameters.fe_degree),
dof_handler(triangulation)
{
    // If the timer is not obtained form another class, reset it.
    if (external_timer == 0)
        computing_timer.reset(new TimerOutput(pcout,
                                              TimerOutput::summary,
                                              TimerOutput::wall_times));
    // Otherwise, just set the pointer
    else
        computing_timer.reset(external_timer);
}

}  // namespace adsolic


