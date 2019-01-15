/*
 * parameters.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include "parameters.h"

namespace BuoyantFluid {

Parameters::Parameters(const std::string &parameter_filename)
:
// runtime parameters
projection_scheme(PressureUpdateType::StandardForm),
convective_discretization(ConvectiveDiscretizationType::SkewSymmetric),
// physics parameters
aspect_ratio(0.35),
Pr(1.0),
Ra(1.0e5),
Ek(1.0e-3),
rotation(false),
// linear solver parameters
rel_tol(1e-6),
abs_tol(1e-12),
n_max_iter(100),
// time stepping parameters
imex_scheme(TimeStepping::CNAB),
n_steps(1000),
initial_timestep(1e-3),
min_timestep(1e-9),
max_timestep(1e-1),
cfl_min(0.3),
cfl_max(0.7),
adaptive_timestep(true),
// discretization parameters
temperature_degree(1),
velocity_degree(2),
// refinement parameters
n_global_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
n_max_levels(6),
refinement_frequency(10),
// logging parameters
output_frequency(10)
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


void Parameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Runtime parameters");
    {
        prm.declare_entry("assemble_schur_complement",
                "false",
                Patterns::Bool(),
                "Perform an explicit assembly of pressure stiffness matrix");
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
        prm.declare_entry("p_degree_velocity",
                "2",
                Patterns::Integer(1,2),
                "Polynomial degree of the velocity discretization. The polynomial "
                "degree of the pressure is automatically set to one less than the velocity");

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

        prm.declare_entry("Ek",
                "1.0e-3",
                Patterns::Double(),
                "Ekman number of the flow");
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver settings");
    {
        prm.declare_entry("tol_rel",
                "1e-6",
                Patterns::Double(),
                "Relative tolerance for the stokes solver.");

        prm.declare_entry("tol_abs",
                "1e-12",
                Patterns::Double(),
                "Absolute tolerance for the stokes solver.");

        prm.declare_entry("n_max_iter",
                "100",
                Patterns::Integer(0),
                "Maximum number of iterations for the stokes solver.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        prm.declare_entry("n_steps",
                "1000",
                Patterns::Integer(),
                "Maximum number of iteration. That is the maximum number of time steps.");

        prm.declare_entry("dt_initial",
                "1e-4",
                Patterns::Double(),
                "Initial time step.");

        prm.declare_entry("dt_min",
                "1e-6",
                Patterns::Double(),
                "Maximum time step.");

        prm.declare_entry("dt_max",
                "1e-1",
                Patterns::Double(),
                "Maximum time step.");

        prm.declare_entry("cfl_min",
                "0.3",
                Patterns::Double(),
                "Minimal value for the cfl number.");

        prm.declare_entry("cfl_max",
                "0.7",
                Patterns::Double(),
                "Maximal value for the cfl number.");

        prm.declare_entry("adaptive_timestep",
                "true",
                Patterns::Bool(),
                "Turn adaptive time stepping on or off");


        // TODO: move to logging
        prm.declare_entry("output_freq",
                "10",
                Patterns::Integer(),
                "Output frequency.");

        prm.declare_entry("time_stepping_scheme",
                        "CNAB",
                        Patterns::Selection("Euler|CNAB|MCNAB|CNLF|SBDF"),
                        "Time stepping scheme applied.");
    }
    prm.leave_subsection();
}

void Parameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Runtime parameters");
    {
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
        velocity_degree = prm.get_integer("p_degree_velocity");
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
        Ek = prm.get_double("Ek");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        n_steps = prm.get_integer("n_steps");
        Assert(n_steps > 0, ExcLowerRange(n_steps,0));

        initial_timestep = prm.get_double("dt_initial");
        min_timestep = prm.get_double("dt_min");
        max_timestep = prm.get_double("dt_max");
        Assert(min_timestep < max_timestep,
               ExcLowerRangeType<double>(min_timestep, min_timestep));
        Assert(min_timestep <= initial_timestep,
               ExcLowerRangeType<double>(min_timestep, initial_timestep));
        Assert(initial_timestep <= max_timestep,
               ExcLowerRangeType<double>(initial_timestep, max_timestep));

        cfl_min = prm.get_double("cfl_min");
        cfl_max = prm.get_double("cfl_max");
        Assert(cfl_min < cfl_max, ExcLowerRangeType<double>(cfl_min, cfl_max));

        adaptive_timestep = prm.get_bool("adaptive_timestep");

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
        else if (imex_type_str == "Euler")
            imex_scheme = TimeStepping::IMEXType::Euler;

        // TODO: move to logging
        output_frequency = prm.get_integer("output_freq");
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver settings");
    {
        rel_tol = prm.get_double("tol_rel");
        abs_tol = prm.get_double("tol_abs");

        n_max_iter = prm.get_integer("n_max_iter");
    }
    prm.leave_subsection();
}

}  // namespace BuoyantFluid
