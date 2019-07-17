/*
 * parameters.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */
#include <deal.II/base/geometric_utilities.h>

#include <parameters.h>

namespace BuoyantFluid {

Parameters::Parameters(const std::string &parameter_filename)
:
// runtime parameters
dim(2),
n_steps(100),
refinement_frequency(30),
t_final(1.0),
adaptive_refinement(true),
resume_from_snapshot(false),
// output parameters
output_flags(OutputFlags::output_default),
output_benchmark_results(false),
output_point_probe(false),
// point probe parameters
point_probe_frequency(10),
probe_points({{1.,0.,0.},}),
point_coordinate_system(CoordinateSystem::Cartesian),
point_probe_spherical(false),
// benchmark parameters
benchmark_frequency(5),
benchmark_start(100000),
// logging parameters
vtk_frequency(10),
global_avg_frequency(5),
cfl_frequency(5),
snapshot_frequency(100),
verbose(false),
// geometry parameters
geometry(GeometryType::SphericalShell),
aspect_ratio(0.35),
// physics parameters
Pr(1.0),
Ra(1.0e5),
Ek(1.0e-3),
Pm(5.0),
gravity_profile(EquationData::GravityProfile::LinearRadial),
rotation(false),
buoyancy(true),
magnetism(false),
magnetic_induction(false),
// linear solver parameters
rel_tol(1e-6),
abs_tol(1e-9),
max_iter_navier_stokes(50),
max_iter_temperature(200),
max_iter_magnetic(50),
// time stepping parameters
imex_scheme(TimeStepping::CNAB),
initial_timestep(1e-3),
min_timestep(1e-9),
max_timestep(1e-3),
cfl_min(0.3),
cfl_max(0.7),
adaptive_timestep_barrier(2),
adaptive_timestep(true),
// discretization parameters
projection_scheme(PressureUpdateType::StandardForm),
convective_weak_form(ConvectiveWeakForm::SkewSymmetric),
convective_scheme(ConvectiveDiscretizationType::Explicit),
temperature_degree(1),
velocity_degree(2),
magnetic_degree(2),
// refinement parameters
n_global_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
n_max_levels(6),
n_min_levels(3)
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
        prm.declare_entry("adaptive_refinement",
                "false",
                Patterns::Bool(),
                "Flag to activate adaptive mesh refinement.");
        prm.declare_entry("resume_from_snapshot",
                "false",
                Patterns::Bool(),
                "Flag to resume from a snapshot of an earlier simulation.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Output parameters");
    {
        prm.declare_entry("output_benchmark_results",
                "false",
                Patterns::Bool(),
                "Flag to activate benchmarking.");

        prm.declare_entry("output_point_probe",
                "false",
                Patterns::Bool(),
                "Flag to activate point probe.");

        prm.enter_subsection("Benchmarking");
        {
            prm.declare_entry("benchmark_freq",
                    "5",
                    Patterns::Integer(),
                    "Output frequency of benchmark report.");
            prm.declare_entry("benchmark_start",
                    "500",
                    Patterns::Integer(),
                    "Time step after which benchmark values are reported.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Point probe");
        {
            prm.declare_entry("point_probe_freq",
                              "5",
                              Patterns::Integer(),
                              "Output frequency of point probe.");
            Patterns::List  point(Patterns::Double(),2,3);
            prm.declare_entry("point_coordinates",
                              "1.0,0.0,0.0",
                              Patterns::List(point,1,Patterns::List::max_int_value,";"),
                              "Coordinates of the point where the solution is probed.");
            prm.declare_entry("coordinate_system",
                              "Cartesian",
                              Patterns::Selection("Cartesian|Spherical"),
                              "Type of the coordinate system used for the "
                              "coordinates of the point (Cartesian|Spherical).");
            prm.declare_entry("output_spherical_components",
                              "false",
                              Patterns::Bool(),
                              "Flag to activate output of spherical components"
                              "at the point where the solution is probed.");
        }
        prm.leave_subsection();

        prm.declare_entry("output_values",
                "true",
                Patterns::Bool(),
                "Flag to activate output of solution values.");
        prm.declare_entry("output_scalar_gradients",
                "false",
                Patterns::Bool(),
                "Flag to activate output of gradients of scalar fields.");
        prm.declare_entry("output_mpi_partition",
                "false",
                Patterns::Bool(),
                "Flag to activate output of the MPI partition.");
        prm.declare_entry("output_spherical_components",
                "false",
                Patterns::Bool(),
                "Flag to activate output of spherical components.");
        prm.declare_entry("output_velocity_curl",
                "false",
                Patterns::Bool(),
                "Flag to activate output of vorticity (curl of the velocity).");
        prm.declare_entry("output_magnetic_curl",
                "false",
                Patterns::Bool(),
                "Flag to activate output of curl of the magnetic field.");
        prm.declare_entry("output_magnetic_helicity",
                "false",
                Patterns::Bool(),
                "Flag to activate output of the magnetic helicity.");
        prm.declare_entry("output_coriolis_force",
                "false",
                Patterns::Bool(),
                "Flag to activate output of the Coriolis force.");
        prm.declare_entry("output_lorentz_force",
                "false",
                Patterns::Bool(),
                "Flag to activate output of the Lorentz force.");
        prm.declare_entry("output_buoyancy_force",
                "false",
                Patterns::Bool(),
                "Flag to activate output of the buoyancy force.");
        prm.declare_entry("output_magnetic_induction",
                "false",
                Patterns::Bool(),
                "Flag to activate output of the magnetic induction term.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Initial conditions");
    {
        prm.declare_entry("temperature_perturbation",
                "None",
                Patterns::Selection("None|Sinusoidal"),
                "Type of perturbation (None|Sinusoidal).");
    }
    prm.leave_subsection();

    prm.enter_subsection("Logging parameters");
    {
        prm.declare_entry("vtk_freq",
                "10",
                Patterns::Integer(),
                "Output frequency for vtk-files.");
        prm.declare_entry("global_avg_freq",
                "10",
                Patterns::Integer(),
                "Output frequency for global averages like rms values and energies.");
        prm.declare_entry("cfl_freq",
                "10",
                Patterns::Integer(),
                "Output frequency of current cfl number.");
        prm.declare_entry("snapshot_freq",
                "100",
                Patterns::Integer(),
                "Output frequency of snapshots.");
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
                "degree of the pressure is automatically set to one less than the velocity degree.");

        prm.declare_entry("p_degree_temperature",
                "1",
                Patterns::Integer(1,2),
                "Polynomial degree of the temperature discretization.");

        prm.declare_entry("p_degree_magnetic",
                "2",
                Patterns::Integer(1,2),
                "Polynomial degree of the magnetic field discretization. The polynomial "
                "degree of the pseudo pressure is automatically set to one less than the magnetic field one.");

        prm.declare_entry("pressure_update_type",
                "Standard",
                Patterns::Selection("Standard|Irrotational"),
                "Type of projection scheme applied for the pressure (Standard|Irrotational).");

        prm.declare_entry("magnetic_pressure_update_type",
                "Standard",
                Patterns::Selection("Standard|Irrotational"),
                "Type of projection scheme applied for the magnetic pseudo pressure (Standard|Irrotational).");

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

            prm.declare_entry("n_min_levels",
                    "1",
                    Patterns::Integer(),
                    "Minimum of number of refinements during the run.");

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Geometry");
    {
        prm.declare_entry("geometry_type",
                          "SphericalShell",
                          Patterns::Selection("SphericalShell|Cavity"),
                          "Type of the geometry (SphericalShell|Cavity).");

        prm.declare_entry("aspect_ratio",
                          "0.35",
                          Patterns::Double(0.,100.),
                          "Ratio of inner to outer radius");
    }
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    {
        prm.declare_entry("rotating_case",
                "true",
                Patterns::Bool(),
                "Turn rotation on or off.");

        prm.declare_entry("buoyant_case",
                "true",
                Patterns::Bool(),
                "Turn buoyancy on or off");

        prm.declare_entry("magnetic_case",
                "false",
                Patterns::Bool(),
                "Turn magnetism on or off.");

        prm.declare_entry("magnetic_induction",
                "false",
                Patterns::Bool(),
                "Turn the magnetic induction term and the Lorentz force term on or off "
                "in the magnetic induction equation and the momentum equation.");

        prm.declare_entry("Pr",
                "1.0",
                Patterns::Double(),
                "Prandtl number of the fluid.");

        prm.declare_entry("Ra",
                "1.0e5",
                Patterns::Double(),
                "Rayleigh number of the fluid.");

        prm.declare_entry("Ek",
                "1.0e-3",
                Patterns::Double(),
                "Ekman number of the fluid.");

        prm.declare_entry("Pm",
                "5.0",
                Patterns::Double(),
                "Magnetic Prandtl number of the fluid.");

        prm.declare_entry("GravityProfile",
                "LinearRadial",
                Patterns::Selection("ConstantRadial|LinearRadial|ConstantCartesian"),
                "Type of the gravity profile (ConstantRadial|LinearRadial|ConstantCartesian).");
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver settings");
    {
        prm.declare_entry("tol_rel",
                "1e-6",
                Patterns::Double(),
                "Relative tolerance for all linear solvers.");

        prm.declare_entry("tol_abs",
                "1e-9",
                Patterns::Double(),
                "Absolute tolerance for all linear solvers.");

        prm.declare_entry("max_iter_navier_stokes",
                "50",
                Patterns::Integer(0),
                "Maximum number of iterations for the Navier-Stokes solver.");

        prm.declare_entry("max_iter_temperature",
                "200",
                Patterns::Integer(0),
                "Maximum number of iterations for the temperature solver.");

        prm.declare_entry("max_iter_magnetic",
                "50",
                Patterns::Integer(0),
                "Maximum number of iterations for the magnetic solver.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        prm.declare_entry("dt_initial",
                "1e-6",
                Patterns::Double(),
                "Initial time step.");

        prm.declare_entry("dt_min",
                "1e-6",
                Patterns::Double(),
                "Maximum time step.");

        prm.declare_entry("dt_max",
                "1e-3",
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

        prm.declare_entry("adaptive_timestep_barrier",
                          "2",
                          Patterns::Integer(),
                          "Time step after which adaptive time stepping is applied.");

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
    prm.enter_subsection("Geometry");
    {
        const std::string coordinate_system_string
        = prm.get("geometry_type");

        if (coordinate_system_string == "SphericalShell")
            geometry = GeometryType::SphericalShell;
        else if (coordinate_system_string == "Cavity")
            geometry = GeometryType::Cavity;
        else
            AssertThrow(false, ExcMessage("Unexpected string for type of the geometry."));

        aspect_ratio = prm.get_double("aspect_ratio");
        Assert(aspect_ratio > 0., ExcLowerRangeType<double>(0., aspect_ratio));
        if (geometry == GeometryType::SphericalShell)
            Assert(aspect_ratio < 1., ExcLowerRangeType<double>(aspect_ratio, 1.0));
    }
    prm.leave_subsection();

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

        adaptive_refinement = prm.get_bool("adaptive_refinement");

        resume_from_snapshot = prm.get_bool("resume_from_snapshot");
    }
    prm.leave_subsection();

    prm.enter_subsection("Output parameters");
    {
        output_benchmark_results = prm.get_bool("output_benchmark_results");

        if (output_benchmark_results)
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("Benchmark output can only be request for a spherical"
                              " shell geometry."));

        prm.enter_subsection("Benchmarking");
        if (output_benchmark_results)
        {
            benchmark_frequency = prm.get_integer("benchmark_freq");
            Assert(benchmark_frequency > 0, ExcLowerRange(0, benchmark_frequency));

            benchmark_start = prm.get_integer("benchmark_start");
            Assert(benchmark_start > 0, ExcLowerRange(0, benchmark_start));

        }
        prm.leave_subsection();

        output_point_probe = prm.get_bool("output_point_probe");

        prm.enter_subsection("Point probe");
        if (output_point_probe)
        {

            point_probe_frequency = prm.get_integer("point_probe_freq");
            Assert(point_probe_frequency > 0, ExcLowerRange(0, point_probe_frequency));

            // get a single string which is comma-separated
            const std::string point_list = prm.get("point_coordinates");

            // seperate the string
            const std::vector<std::string> point_strings
            = Utilities::split_string_list(point_list, ";");

            std::vector<std::string>::const_iterator
            point_str = point_strings.begin(),
            end_str = point_strings.end();
            for (; point_str != end_str; ++point_str)
            {
                // seperate the string
                const std::vector<std::string> coordinate_strings
                = Utilities::split_string_list(*point_str);
                AssertDimension(coordinate_strings.size(),
                                dim);

                std::vector<double>    point(dim, 0.);

                // convert the coordinates to doubles
                typename std::vector<std::string>::const_iterator
                coord_str = coordinate_strings.begin();
                typename std::vector<double>::iterator
                coord = point.begin();
                for (; coord_str != coordinate_strings.end(); ++coord_str, ++coord)
                {
                    *coord = Utilities::string_to_double(*coord_str);
                    AssertIsFinite(*coord);
                }

                probe_points.push_back(point);
            }

            const std::string coordinate_system_string
            = prm.get("coordinate_system");

            if (coordinate_system_string == "Cartesian")
                point_coordinate_system = CoordinateSystem::Cartesian;
            else if (coordinate_system_string == "Spherical")
            {
                Assert(geometry == GeometryType::SphericalShell,
                       ExcMessage("Spherical point probe coordinates only make"
                                  "sense for a spherical shell geometry."));
                point_coordinate_system = CoordinateSystem::Spherical;
            }
            else
                AssertThrow(false, ExcMessage("Unexpected string for coordinate system."));

            point_probe_spherical = prm.get_bool("output_spherical_components");

        }
        prm.leave_subsection();

        if (prm.get_bool("output_values"))
            output_flags |= output_values;
        if (prm.get_bool("output_scalar_gradients"))
            output_flags |= output_scalar_gradients;
        if (prm.get_bool("output_mpi_partition"))
            output_flags |= output_mpi_partition;
        if (prm.get_bool("output_spherical_components"))
        {
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("Spherical component output only makes sense for a"
                              " spherical shell geometry."));
            output_flags |= output_spherical_components;
        }
        if (prm.get_bool("output_velocity_curl"))
            output_flags |= output_velocity_curl;
        if (prm.get_bool("output_magnetic_curl"))
            output_flags |= output_magnetic_curl;
        if (prm.get_bool("output_magnetic_helicity"))
            output_flags |= output_magnetic_helicity;
        if (prm.get_bool("output_coriolis_force"))
            output_flags |= output_coriolis_force;
        if (prm.get_bool("output_lorentz_force"))
            output_flags |= output_lorentz_force;
        if (prm.get_bool("output_buoyancy_force"))
            output_flags |= output_buoyancy_force;
        if (prm.get_bool("output_magnetic_induction"))
            output_flags |= output_magnetic_induction;
    }
    prm.leave_subsection();

    prm.enter_subsection("Initial conditions");
    {
        std::string perturbation_string = prm.get("temperature_perturbation");

        if (perturbation_string == "None")
            temperature_perturbation = EquationData::TemperaturePerturbation::None;
        else if (perturbation_string == "Sinusoidal")
        {
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("Sinusoidal temperature perturbation only makes"
                              " sense for a spherical shell geometry."));

            temperature_perturbation = EquationData::TemperaturePerturbation::Sinusoidal;
        }
        else
            AssertThrow(false, ExcMessage("Unexpected string for temperature perturbation."));
    }
    prm.leave_subsection();


    prm.enter_subsection("Logging parameters");
    {
        vtk_frequency = prm.get_integer("vtk_freq");
        Assert(vtk_frequency > 0, ExcLowerRange(0, vtk_frequency));

        global_avg_frequency = prm.get_integer("global_avg_freq");
        Assert(global_avg_frequency > 0, ExcLowerRange(0, global_avg_frequency));

        cfl_frequency = prm.get_integer("cfl_freq");
        Assert(cfl_frequency > 0, ExcLowerRange(0, cfl_frequency));

        snapshot_frequency = prm.get_integer("snapshot_freq");
        Assert(snapshot_frequency > 0, ExcLowerRange(0, snapshot_frequency));

        verbose = prm.get_bool("verbose");
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
        velocity_degree = prm.get_integer("p_degree_velocity");
        Assert(velocity_degree > 1, ExcLowerRange(velocity_degree, 1));

        temperature_degree = prm.get_integer("p_degree_temperature");
        Assert(temperature_degree > 0, ExcLowerRange(temperature_degree, 0));

        magnetic_degree = prm.get_integer("p_degree_magnetic");
        Assert(magnetic_degree > 1, ExcLowerRange(magnetic_degree, 1));

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

        const std::string magnetic_projection_type_str
        = prm.get("magnetic_pressure_update_type");

        if (magnetic_projection_type_str == "Standard")
            magnetic_projection_scheme = PressureUpdateType::StandardForm;
        else if (magnetic_projection_type_str == "Irrotational")
            magnetic_projection_scheme = PressureUpdateType::IrrotationalForm;
        else
            AssertThrow(false, ExcMessage("Unexpected string for magnetic pressure update scheme."));

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
            n_global_refinements = prm.get_integer("n_global_refinements");
            n_initial_refinements = prm.get_integer("n_initial_refinements");
            n_boundary_refinements = prm.get_integer("n_boundary_refinements");

            n_max_levels = prm.get_integer("n_max_levels");

            n_min_levels = prm.get_integer("n_min_levels");

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
            if (n_max_levels < n_min_levels)
            {
                std::ostringstream message;
                message << "Inconsistency in parameter file in definition of minimum and maximum number of levels."
                        << std::endl
                        << "minimum number of levels is: "
                        << n_min_levels
                        << ", which is larger than the minimum number of levels,"
                        << std::endl
                        << " which is "
                        << n_max_levels
                        << " for your parameter file."
                        << std::endl;

                AssertThrow(false, ExcMessage(message.str().c_str()));
            }
            if (n_min_levels > n_global_refinements + n_boundary_refinements + n_initial_refinements)
            {
                std::ostringstream message;
                message << "Inconsistency in parameter file in the definition of "
                           "the minimum number of levels and sum of the refinements "
                           "during the initialization."
                        << std::endl
                        << "Minimum number of levels is: "
                        << n_min_levels << ", "
                        << "which is larger than the sum of the refinements during"
                           "the initialization, "
                        << std::endl
                        << "which is "
                        << n_global_refinements + n_boundary_refinements + n_initial_refinements
                        << " for your parameter file."
                        << std::endl;

                AssertThrow(false, ExcMessage(message.str().c_str()));
            }

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    {
        rotation = prm.get_bool("rotating_case");
        if (rotation)
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("Rotation option only makes"
                              " sense for a spherical shell geometry."));

        buoyancy = prm.get_bool("buoyant_case");

        magnetism = prm.get_bool("magnetic_case");
        if (magnetism)
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("Magnetic option only makes"
                              " sense for a spherical shell geometry."));


        magnetic_induction = prm.get_bool("magnetic_induction");
        if (magnetic_induction)
            Assert(magnetic_induction && magnetism,
                   ExcMessage("The parameter magnetic_induction cannot be active "
                              "unless the parameter magnetism is active, "
                              "because the magnetic field needs to be solved "
                              "in order to compute the Lorentz force."));

        Ra = prm.get_double("Ra");
        Assert(Ra > 0, ExcLowerRangeType<double>(Ra, 0));

        Pr = prm.get_double("Pr");
        Assert(Pr > 0, ExcLowerRangeType<double>(Pr, 0));

        Ek = prm.get_double("Ek");
        Assert(Ek > 0, ExcLowerRangeType<double>(Ek, 0));

        Pm = prm.get_double("Pm");
        Assert(Pm > 0, ExcLowerRangeType<double>(Pm, 0));

        std::string profile_string = prm.get("GravityProfile");

        if (profile_string == "ConstantRadial")
        {
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("A constant radial gravity profile only makes"
                              " sense for a spherical shell geometry."));
            gravity_profile = EquationData::GravityProfile::ConstantRadial;
        }
        else if (profile_string == "LinearRadial")
        {
            Assert(geometry == GeometryType::SphericalShell,
                   ExcMessage("A linear radial gravity profile only makes"
                              " sense for a spherical shell geometry."));
            gravity_profile = EquationData::GravityProfile::LinearRadial;
        }
        else if (profile_string == "ConstantCartesian")
        {
            Assert(geometry == GeometryType::Cavity,
                   ExcMessage("A Cartesian gravity profile only makes"
                              " sense for a cartesian geometry."));
            gravity_profile = EquationData::GravityProfile::ConstantCartesian;
        }
        else
            AssertThrow(false, ExcMessage("Unexpected string for gravity profile."));

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

        adaptive_timestep_barrier = prm.get_integer("adaptive_timestep_barrier");
        Assert(adaptive_timestep_barrier > 0,
               ExcLowerRange(adaptive_timestep_barrier, 2));

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

        max_iter_navier_stokes = prm.get_integer("max_iter_navier_stokes");
        Assert(max_iter_navier_stokes > 0, ExcLowerRange(max_iter_navier_stokes, 0));

        max_iter_temperature = prm.get_integer("max_iter_temperature");
        Assert(max_iter_temperature > 0, ExcLowerRange(max_iter_temperature, 0));

        max_iter_magnetic = prm.get_integer("max_iter_magnetic");
        Assert(max_iter_temperature > 0, ExcLowerRange(max_iter_magnetic, 0));
    }
    prm.leave_subsection();
}

template<int dim>
std::vector<Point<dim>> probe_points(const Parameters &parameters)
{
    const unsigned int n_points = parameters.probe_points.size();

    std::vector<Point<dim>> probe_points(n_points);

    typename std::vector<std::vector<double>>::const_iterator
    coord = parameters.probe_points.begin(),
    end_coord = parameters.probe_points.end();
    typename std::vector<Point<dim>>::iterator
    point = probe_points.begin();

    if (parameters.point_coordinate_system == CoordinateSystem::Spherical)
        for (; coord != end_coord; ++coord, ++point)
        {
            AssertDimension(coord->size(), dim);

            std::array<double,dim> scoord;
            for (unsigned int d=0; d<dim; ++d)
                scoord[d] = coord->at(d);
            *point = GeometricUtilities::Coordinates::from_spherical(scoord);
        }
    else if (parameters.point_coordinate_system == CoordinateSystem::Cartesian)
        for (; coord != end_coord; ++coord, ++point)
        {
            AssertDimension(coord->size(), dim);

            for (unsigned d=0; d<dim; ++d)
                (*point)[d] = coord->at(d);
        }
    else
        Assert(false, ExcInternalError());

    return probe_points;
}

// explicit instantiation
template std::vector<Point<2>> probe_points<2>(const Parameters &);
template std::vector<Point<3>> probe_points<3>(const Parameters &);

} // namespace BuoyantFluid
