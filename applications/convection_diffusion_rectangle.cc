/*
 * convection_diffusion_rectangle.cc
 *
 *  Created on: Jul 24, 2019
 *      Author: sg
 */

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include <adsolic/convection_diffusion_solver.h>
#include <adsolic/utility.h>

using namespace dealii;
using namespace adsolic;

/*
 *
 * temperature field
 *
 */
template <int dim>
class TemperatureField : public Function<dim>
{
public:
    TemperatureField(const double coefficient_in = 1.0,
                     const double kx = 2.0 * numbers::PI,
                     const double ky = 2.0 * numbers::PI,
                     const double kz = 2.0 * numbers::PI);

    virtual double value(const Point<dim>  &point,
                         const unsigned int component = 0) const;
private:
    const double coefficient;
    const double kx;
    const double ky;
    const double kz;
};

template <int dim>
TemperatureField<dim>::TemperatureField
(const double coefficient_in,
 const double kx_in,
 const double ky_in,
 const double kz_in)
:
coefficient(coefficient_in),
kx(kx_in),
ky(ky_in),
kz(kz_in)
{}


template <int dim>
double TemperatureField<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
    AssertThrow(dim == 2,
                ExcImpossibleInDim(dim));
    return std::cos(kx * point[0] - numbers::PI / 2.0) * std::sin(ky * point[1])
        * std::exp(-coefficient * (kx * kx + ky * ky) * this->get_time());
}

/*
 *
 * heat conduction parameters
 *
 */
struct HeatConductionParameters
{
    HeatConductionParameters(const std::string &parameter_filename);

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    /*
     * function forwarding parameters to a stream object
     */
    template<typename Stream>
    void write(Stream &stream) const;

    ConvectionDiffusionParameters   convect_diff_params;

    TimeSteppingParameters          time_stepping_params;

    // geometry parameters
    unsigned int    dim;

    // advection field parameters
    double  amplitude;
    double  kx;
    double  ky;
    double  kz;

    // refinement parameters
    bool            adaptive_mesh_refinement;
    unsigned int    n_global_refinements;
    unsigned int    n_initial_refinements;
    unsigned int    n_boundary_refinements;
    unsigned int    n_max_levels;
    unsigned int    n_min_levels;

    // logging parameters
    unsigned int    vtk_frequency;

    // verbosity flag
    bool            verbose;
};


HeatConductionParameters::HeatConductionParameters(const std::string &parameter_filename)
:
// geometry parameters
dim(2),
// advection field parameters
amplitude(1.0),
kx(2.*numbers::PI),
ky(2.*numbers::PI),
kz(2.*numbers::PI),
// refinement parameters
adaptive_mesh_refinement(false),
n_global_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
n_max_levels(6),
n_min_levels(3),
// logging parameters
vtk_frequency(10),
// verbosity flag
verbose(false)
{
    ParameterHandler prm;

    declare_parameters(prm);

    convect_diff_params.declare_parameters(prm);
    time_stepping_params.declare_parameters(prm);

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

    convect_diff_params.parse_parameters(prm);

    time_stepping_params.parse_parameters(prm);

    if (verbose == true)
    {
        convect_diff_params.verbose = true;
        time_stepping_params.verbose = true;
    }
}


void HeatConductionParameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Geometry parameters");
    {
        prm.declare_entry("dim",
                "2",
                Patterns::Integer(2,3),
                "Spatial dimension.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Logging parameters");
    {
        prm.declare_entry("vtk_freq",
                "10",
                Patterns::Integer(),
                "Output frequency for vtk-files.");

        prm.declare_entry("verbose",
                "false",
                Patterns::Bool(),
                "Flag to activate output of subroutines.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Refinement parameters");
    {
        prm.declare_entry("adaptive_mesh_refinement",
                "false",
                Patterns::Bool(),
                "Flag to activate adaptive mesh refinement.");

        prm.declare_entry("refinement_freq",
                "100",
                Patterns::Integer(),
                "Refinement frequency.");

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

    prm.enter_subsection("Convective parameters");
    {
        prm.declare_entry("amplitude",
                "1.0",
                Patterns::Double(),
                "Amplitude of the convective field.");

        prm.declare_entry("kx",
                "6.283185307179586",
                Patterns::Double(),
                "Wave number in x-direction.");

        prm.declare_entry("ky",
                "6.283185307179586",
                Patterns::Double(),
                "Wave number in y-direction.");

        prm.declare_entry("kz",
                "6.283185307179586",
                Patterns::Double(),
                "Wave number in z-direction.");
    }
    prm.leave_subsection();
}

void HeatConductionParameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Geometry parameters");
    {
        dim = prm.get_integer("dim");
        Assert(dim > 1, ExcLowerRange(1, dim));
    }
    prm.leave_subsection();

    prm.enter_subsection("Logging parameters");
    {
        vtk_frequency = prm.get_integer("vtk_freq");
        Assert(vtk_frequency > 0, ExcLowerRange(0, vtk_frequency));

        verbose = prm.get_bool("verbose");
    }
    prm.leave_subsection();

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
    }
    prm.leave_subsection();

    prm.enter_subsection("Convective parameters");
    {
        amplitude = prm.get_double("amplitude");
        Assert(amplitude > 0.0,
               ExcMessage("Amplitude of the convective field must be positive."));

        kx = prm.get_double("kx");
        Assert(kx > 0.0,
               ExcMessage("Wave number in x-direction must be positive."));

        ky = prm.get_double("ky");
        Assert(ky > 0.0,
               ExcMessage("Wave number in y-direction must be positive."));

        kz = prm.get_double("kz");
        Assert(kz > 0.0,
               ExcMessage("Wave number in z-direction must be positive."));
    }
    prm.leave_subsection();
}

template<typename Stream>
void HeatConductionParameters::write(Stream &stream) const
{
    stream << "Heat conduction parameters" << std::endl
           << "   dim: " << dim << std::endl
           << "   amplitude: " << amplitude << std::endl
           << "   kx: " << kx << std::endl
           << "   ky: " << ky << std::endl
           << "   kz: " << kz << std::endl
           << "   adaptive_mesh_refinement: " << (adaptive_mesh_refinement? "true": "false") << std::endl
           << "   n_global_refinements: " << n_global_refinements << std::endl
           << "   n_initial_refinements: " << n_initial_refinements << std::endl
           << "   n_boundary_refinements: " << n_boundary_refinements << std::endl
           << "   n_max_levels: " << n_max_levels << std::endl
           << "   n_min_levels: " << n_min_levels << std::endl
           << "   vtk_frequency: " << vtk_frequency << std::endl
           << "   verbose: " << (verbose? "true": "false") << std::endl;

    convect_diff_params.write(stream);
    time_stepping_params.write(stream);
}

/*
 *
 * problem class
 *
 */
template <int dim>
class HeatConductionSquare
{

public:
    HeatConductionSquare(const HeatConductionParameters &parameters);

    void run();

private:

    void output_results() const;

    const HeatConductionParameters &parameters;

    MPI_Comm            mpi_communicator;

    ConditionalOStream  pcout;

    parallel::distributed::Triangulation<dim>   triangulation;

    MappingQ<dim>       mapping;

    IMEXTimeStepping    timestepper;

    AuxiliaryFunctions::ConvectionFunction<dim>    advection_function;
    TemperatureField<dim>   temperature_function;

    std::shared_ptr<BC::ScalarBoundaryConditions<dim>> boundary_conditions;

    mutable std::shared_ptr<TimerOutput>    timer;

    ConvectionDiffusionSolver<dim>    solver;
};


template <int dim>
HeatConductionSquare<dim>::HeatConductionSquare
(const HeatConductionParameters &parameters)
:
parameters(parameters),
mpi_communicator(MPI_COMM_WORLD),
pcout(std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
triangulation(mpi_communicator),
mapping(1),
timestepper(parameters.time_stepping_params),
advection_function(parameters.amplitude,
                   parameters.kx,
                   parameters.ky),
temperature_function(parameters.convect_diff_params.equation_coefficient,
                     parameters.kx,
                     parameters.ky),
boundary_conditions(new BC::ScalarBoundaryConditions<dim>()),
timer(new TimerOutput(pcout,TimerOutput::summary,TimerOutput::cpu_and_wall_times)),
solver(parameters.convect_diff_params,
       triangulation,
       mapping,
       timestepper,
       boundary_conditions,
       timer)
{
    parameters.write(pcout);
}

template <int dim>
void HeatConductionSquare<dim>::output_results () const
{
    timer->enter_subsection("Output solution.");

    DataOut<dim> data_out;

    data_out.add_data_vector(solver.get_dof_handler(),
                           solver.get_solution(),
                           "temperature");
    data_out.build_patches(parameters.convect_diff_params.fe_degree);

    // write output to disk
    const std::string filename = ("solution-" +
                                 Utilities::int_to_string(timestepper.step_no(), 5) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4) +
                                ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
          filenames.push_back(std::string("solution-") +
                              Utilities::int_to_string(timestepper.step_no(), 5) +
                              "." +
                              Utilities::int_to_string(i, 4) +
                              ".vtu");
      const std::string
      pvtu_master_filename = ("solution-" +
                              Utilities::int_to_string(timestepper.step_no(), 5) +
                              ".pvtu");
      std::ofstream pvtu_master(pvtu_master_filename.c_str());
      data_out.write_pvtu_record(pvtu_master, filenames);
    }

   timer->leave_subsection();
}

template <int dim>
void HeatConductionSquare<dim>::run()
{
    timer->enter_subsection ("Setup grid");

    pcout << "Running a " << dim << "D heat conduction problem "
          << "using " << timestepper.name()
          << ", Q"  << solver.get_fe().degree
          << " elements on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator) << " processes."
          << std::endl;

    {
        GridGenerator::hyper_cube(triangulation);

        triangulation.refine_global(parameters.n_global_refinements);
    }

    const std::shared_ptr<Function<dim>> dirichlet_function
    = std::make_shared<Functions::ZeroFunction<dim>>();

    ConvectionFunction<dim>  convective_field(parameters.amplitude,
                                           parameters.kx,
                                           parameters.ky);

    boundary_conditions->set_dirichlet_bc(0, dirichlet_function);

    timer->leave_subsection();

    solver.set_convection_function(convective_field);

    solver.setup_problem();

    solver.setup_initial_condition(TemperatureField<dim>());

    output_results();

    pcout << "Start time integration..." << std::endl;

    while (timestepper.at_end() == false)
    {
        timestepper.advance_time_step();

        timestepper.print_info(pcout);

        convective_field.set_time(timestepper.now());

        solver.advance_time_step();
    }
}

int main (int argc, char **argv)
{
    /* we initialize MPI at the start of the program. Since we will in general
     * mix MPI parallelization with threads, we also set the third argument in
     * MPI_InitFinalize that controls the number of threads to an invalid
     * number, which means that the TBB library chooses the number of threads
     * automatically, typically to the number of available cores in the
     * system. As an alternative, you can also set this number manually if you
     * want to set a specific number of threads (e.g. when MPI-only is required)
     * (cf. step-40 and 48).
     */
    try
    {
        using namespace dealii;
        using namespace adsolic;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, -1);
        deallog.depth_console (0);

        std::string paramfile;
        if (argc > 1)
            paramfile = argv[1];
        else
            paramfile = "convection_diffusion_rectangle.prm";

        HeatConductionParameters parameters(paramfile);

        if (parameters.dim == 2)
        {
            HeatConductionSquare<2> problem(parameters);
            problem.run ();
        }
        else if (parameters.dim == 3)
        {
            HeatConductionSquare<3> problem (parameters);
            problem.run ();
        }
        else
            AssertThrow (false, ExcNotImplemented());
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        return 1;
    }

    return 0;
}
