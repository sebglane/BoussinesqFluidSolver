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
dim(2),
n_steps(100),
refinement_frequency(30),
t_final(1.0),
// logging parameters
vtk_frequency(10),
rms_frequency(5),
cfl_frequency(5),
// physics parameters
aspect_ratio(0.35),
Pr(1.0),
Ra(1.0e5),
Ek(1.0e-3),
rotation(false),
// linear solver parameters
rel_tol(1e-6),
abs_tol(1e-9),
n_max_iter(50),
// time stepping parameters
imex_scheme(TimeStepping::CNAB),
initial_timestep(1e-3),
min_timestep(1e-9),
max_timestep(1e-1),
cfl_min(0.3),
cfl_max(0.7),
adaptive_timestep(true),
// discretization parameters
projection_scheme(PressureUpdateType::StandardForm),
convective_weak_form(ConvectiveWeakForm::SkewSymmetric),
convective_scheme(ConvectiveDiscretizationType::Explicit),
temperature_degree(1),
velocity_degree(2),
// refinement parameters
n_global_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
n_max_levels(6)
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
        prm.declare_entry("dim",
                "2",
                Patterns::Integer(2,3),
                "Spatial dimension of the simulation");
        prm.declare_entry("n_steps",
                "1000",
                Patterns::Integer(),
                "Maximum number of time steps.");
        prm.declare_entry("t_final",
                "1.0",
                Patterns::Double(0.),
                "Final time.");
        prm.declare_entry("refinement_freq",
                "100",
                Patterns::Integer(),
                "Refinement frequency.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Initial conditions");
    {
        prm.declare_entry("temperature_perturbation",
                "None",
                Patterns::Selection("None|Sinusoidal"),
                "Type of perturbation.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Logging parameters");
    {
        prm.declare_entry("vtk_freq",
                "10",
                Patterns::Integer(),
                "Output frequency for vtk-files.");
        prm.declare_entry("rms_freq",
                "10",
                Patterns::Integer(),
                "Output frequency for rms values.");
        prm.declare_entry("cfl_freq",
                "10",
                Patterns::Integer(),
                "Output frequency of current cfl number.");
        prm.declare_entry("verbose",
                "false",
                Patterns::Bool(),
                "Flag to activate output of subroutines.");
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

        prm.declare_entry("pressure_update_type",
                "Standard",
                Patterns::Selection("Standard|Irrotational"),
                "Type of pressure projection scheme applied (Standard|Irrotational).");

        prm.declare_entry("convective_weak_form",
                "Standard",
                Patterns::Selection("Standard|DivergenceForm|SkewSymmetric|RotationalForm"),
                "Type of weak form of convective term (Standard|DivergenceForm|SkewSymmetric|RotationalForm).");

        prm.declare_entry("convective_discretization_scheme",
                "Explicit",
                Patterns::Selection("Explicit|LinearImplicit"),
                "Type of discretization scheme of convective term (Explicit|LinearImplicit).");

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
        dim = prm.get_integer("dim");
        Assert(dim > 1, ExcLowerRange(1, dim));

        refinement_frequency = prm.get_integer("refinement_freq");
        Assert(refinement_frequency > 0, ExcLowerRange(0, refinement_frequency));

        n_steps = prm.get_integer("n_steps");
        Assert(n_steps > 0, ExcLowerRange(n_steps, 0));

        t_final = prm.get_double("t_final");
        Assert(t_final > 0.0, ExcLowerRangeType<double>(t_final, 0.0));
    }
    prm.leave_subsection();

    prm.enter_subsection("Initial conditions");
    {
        std::string perturbation_string = prm.get("temperature_perturbation");

        if (perturbation_string == "None")
            temperature_perturbation = EquationData::TemperaturePerturbation::None;
        else if (perturbation_string == "Sinusoidal")
            temperature_perturbation = EquationData::TemperaturePerturbation::Sinusoidal;
        else
            AssertThrow(false, ExcMessage("Unexpected string for temperature perturbation."));
    }
    prm.leave_subsection();


    prm.enter_subsection("Logging parameters");
    {
        vtk_frequency = prm.get_integer("vtk_freq");
        Assert(vtk_frequency > 0, ExcLowerRange(0, vtk_frequency));

        rms_frequency = prm.get_integer("rms_freq");
        Assert(rms_frequency > 0, ExcLowerRange(0, rms_frequency));

        cfl_frequency = prm.get_integer("cfl_freq");
        Assert(cfl_frequency > 0, ExcLowerRange(0, cfl_frequency));

        verbose = prm.get_bool("verbose");
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
        velocity_degree = prm.get_integer("p_degree_velocity");
        Assert(velocity_degree > 1, ExcLowerRange(velocity_degree, 1));

        temperature_degree = prm.get_integer("p_degree_temperature");
        Assert(temperature_degree > 0, ExcLowerRange(temperature_degree, 0));

        aspect_ratio = prm.get_double("aspect_ratio");
        Assert(aspect_ratio < 1., ExcLowerRangeType<double>(aspect_ratio, 1.0));

        prm.declare_entry("convective_discretization_type",
                        "Standard",
                        Patterns::Selection("Standard|SkewSymmetric|RotationalForm"),
                        "Type of discretization of convective term.");


        const std::string projection_type_str
        = prm.get("pressure_update_type");

        if (projection_type_str == "Standard")
            projection_scheme = PressureUpdateType::StandardForm;
        else if (projection_type_str == "Irrotational")
            projection_scheme = PressureUpdateType::IrrotationalForm;
        else
            AssertThrow(false, ExcMessage("Unexpected string for pressure update scheme."));

        const std::string convective_weak_form_str
        = prm.get("convective_weak_form");

        if (convective_weak_form_str == "Standard")
            convective_weak_form = ConvectiveWeakForm::Standard;
        else if (convective_weak_form_str == "DivergenceForm")
            convective_weak_form = ConvectiveWeakForm::DivergenceForm;
        else if (convective_weak_form_str == "SkewSymmetric")
            convective_weak_form = ConvectiveWeakForm::SkewSymmetric;
        else if (convective_weak_form_str == "RotationalForm")
            convective_weak_form = ConvectiveWeakForm::RotationalForm;
        else
            AssertThrow(false, ExcMessage("Unexpected string for convective weak form."));

        const std::string convective_discretization_str
        = prm.get("convective_discretization_scheme");

        if (convective_discretization_str == "Explicit")
            convective_scheme = ConvectiveDiscretizationType::Explicit;
        else if (convective_discretization_str == "LinearImplicit")
            convective_scheme = ConvectiveDiscretizationType::LinearImplicit;
        else
        {
            std::cout << convective_discretization_str;
            AssertThrow(false, ExcMessage("Unexpected string for convective discretization scheme."));
        }

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
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    {
        rotation = prm.get_bool("rotating_case");

        Ra = prm.get_double("Ra");
        Assert(Ra > 0, ExcLowerRangeType<double>(Ra, 0));

        Pr = prm.get_double("Pr");
        Assert(Pr > 0, ExcLowerRangeType<double>(Pr, 0));

        Ek = prm.get_double("Ek");
        Assert(Ek > 0, ExcLowerRangeType<double>(Ek, 0));
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        initial_timestep = prm.get_double("dt_initial");
        Assert(initial_timestep > 0, ExcLowerRangeType<double>(initial_timestep, 0));
        min_timestep = prm.get_double("dt_min");
        Assert(min_timestep > 0, ExcLowerRangeType<double>(min_timestep, 0));
        max_timestep = prm.get_double("dt_max");
        Assert(max_timestep > 0, ExcLowerRangeType<double>(max_timestep, 0));

        Assert(initial_timestep < t_final, ExcLowerRangeType<double>(initial_timestep, t_final));

        Assert(min_timestep < max_timestep,
               ExcLowerRangeType<double>(min_timestep, min_timestep));
        Assert(min_timestep <= initial_timestep,
               ExcLowerRangeType<double>(min_timestep, initial_timestep));
        Assert(initial_timestep <= max_timestep,
               ExcLowerRangeType<double>(initial_timestep, max_timestep));

        cfl_min = prm.get_double("cfl_min");
        Assert(cfl_min > 0, ExcLowerRangeType<double>(cfl_min, 0));
        cfl_max = prm.get_double("cfl_max");
        Assert(cfl_max > 0, ExcLowerRangeType<double>(cfl_max, 0));
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
        else
            AssertThrow(false, ExcMessage("Unexpected string for IMEX scheme."));
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver settings");
    {
        rel_tol = prm.get_double("tol_rel");
        Assert(rel_tol > 0, ExcLowerRangeType<double>(rel_tol, 0));
        abs_tol = prm.get_double("tol_abs");
        Assert(abs_tol > 0, ExcLowerRangeType<double>(abs_tol, 0));

        n_max_iter = prm.get_integer("n_max_iter");
        Assert(n_max_iter > 0, ExcLowerRange(n_max_iter, 0));
    }
    prm.leave_subsection();
}

} // namespace BuoyantFluid
