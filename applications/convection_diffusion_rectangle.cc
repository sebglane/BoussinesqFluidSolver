/*
 * convection_diffusion_rectangle.cc
 *
 *  Created on: Jul 24, 2019
 *      Author: sg
 */
#include <deal.II/base/convergence_table.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/numerics/vector_tools.h>

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
                     const double phi_x = 0.,
                     const double phi_y = 0.);

    virtual double value(const Point<dim>  &point,
                         const unsigned int component = 0) const;
private:
    const double coefficient;
    const double kx;
    const double ky;
    const double phi_x;
    const double phi_y;
};

template <int dim>
TemperatureField<dim>::TemperatureField
(const double coefficient_in,
 const double kx_in,
 const double ky_in,
 const double phi_x_in,
 const double phi_y_in)
:
coefficient(coefficient_in),
kx(kx_in),
ky(ky_in),
phi_x(phi_x_in),
phi_y(phi_y_in)
{
    Assert(dim == 2,
           ExcMessage("This class is only implemented in 2D."));
}


template <int dim>
double TemperatureField<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
    return std::cos(kx * point[0] - phi_x) * std::sin(ky * point[1] - phi_y)
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
    double  phi_x;
    double  phi_y;

    bool    modulate_amplitude;

    // refinement parameters
    bool            adaptive_mesh_refinement;
    unsigned int    n_global_refinements;
    unsigned int    n_initial_refinements;
    unsigned int    n_boundary_refinements;
    unsigned int    n_max_levels;
    unsigned int    n_min_levels;

    // cfl limits
    double cfl_min;
    double cfl_max;

    // logging parameters
    unsigned int    vtk_frequency;

    // verbosity flag
    bool            verbose;
};


HeatConductionParameters::HeatConductionParameters(const std::string &parameter_filename)
:
// geometry parameters
dim(2),
// convection field parameters
amplitude(1.0),
kx(2.*numbers::PI),
ky(2.*numbers::PI),
phi_x(0.),
phi_y(0.),
modulate_amplitude(false),
// refinement parameters
adaptive_mesh_refinement(false),
n_global_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
n_max_levels(6),
n_min_levels(3),
// time stepping parameters
cfl_min(0.3),
cfl_max(0.7),
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

        prm.declare_entry("phi_x",
                "0.0",
                Patterns::Double(),
                "Phase shift in x-direction.");

        prm.declare_entry("phi_y",
                "0.0",
                Patterns::Double(),
                "Phase shift in y-direction.");

        prm.declare_entry("modulate_amplitude",
                "false",
                Patterns::Bool(),
                "Flag to modulate the amplitude of the convection field in time.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        prm.declare_entry("cfl_min",
                "0.3",
                Patterns::Double(),
                "Minimal value for the cfl number.");

        prm.declare_entry("cfl_max",
                "0.7",
                Patterns::Double(),
                "Maximal value for the cfl number.");
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

        phi_x = prm.get_double("phi_x");

        phi_y = prm.get_double("phi_y");

        modulate_amplitude = prm.get_bool("modulate_amplitude");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time stepping settings");
    {
        cfl_min = prm.get_double("cfl_min");
        Assert(cfl_min > 0, ExcLowerRangeType<double>(cfl_min, 0));

        cfl_max = prm.get_double("cfl_max");
        Assert(cfl_max > 0, ExcLowerRangeType<double>(cfl_max, 0));
        Assert(cfl_min < cfl_max, ExcLowerRangeType<double>(cfl_min, cfl_max));
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
           << "   phi_x: " << phi_x << std::endl
           << "   phi_y: " << phi_y << std::endl
           << "   modulate_amplitude: " << (modulate_amplitude? "true": "false") << std::endl
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
    HeatConductionSquare(HeatConductionParameters &parameters);

    void run();

private:

    void output_results() const;

    double compute_l2_error() const;

    double compute_cfl_number() const;

    HeatConductionParameters &parameters;

    MPI_Comm            mpi_communicator;

    ConditionalOStream  pcout;

    parallel::distributed::Triangulation<dim>   triangulation;

    MappingQ<dim>       mapping;

    IMEXTimeStepping    timestepper;

    const std::shared_ptr<AuxiliaryFunctions::ConvectionFunction<dim>>
    convection_function;

    const std::shared_ptr<BC::ScalarBoundaryConditions<dim>> boundary_conditions;

    TemperatureField<dim>   temperature_function;

    const std::shared_ptr<TimerOutput>    timer;

    ConvectionDiffusionSolver<dim>    solver;
};


template <int dim>
HeatConductionSquare<dim>::HeatConductionSquare
(HeatConductionParameters &parameters)
:
parameters(parameters),
mpi_communicator(MPI_COMM_WORLD),
pcout(std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
triangulation(mpi_communicator),
mapping(1),
timestepper(parameters.time_stepping_params),
convection_function(new AuxiliaryFunctions::ConvectionFunction<dim>
                        (parameters.amplitude,
                         parameters.kx,
                         parameters.ky,
                         parameters.phi_x,
                         parameters.phi_y)),
boundary_conditions(new BC::ScalarBoundaryConditions<dim>()),
temperature_function(parameters.convect_diff_params.equation_coefficient,
                     parameters.kx,
                     parameters.ky),
timer(new TimerOutput(pcout,TimerOutput::summary,TimerOutput::cpu_and_wall_times)),
solver(parameters.convect_diff_params,
       triangulation,
       mapping,
       timestepper,
       boundary_conditions,
       timer)
{}

template <int dim>
void HeatConductionSquare<dim>::output_results () const
{
    TimerOutput::Scope(*timer,"Output solution.");

    const DoFHandler<dim> &temperature_dof_handler = solver.get_dof_handler();
    const LA::Vector &temperature_solution = solver.get_solution();

    DataOut<dim> data_out;

    data_out.attach_dof_handler(temperature_dof_handler);


    data_out.add_data_vector(temperature_dof_handler,
                             temperature_solution,
                             "temperature");
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(timestepper.step_no(), 5) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output(filename.c_str());
    data_out.write_vtu(output);

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
}

template <int dim>
double HeatConductionSquare<dim>::compute_l2_error() const
{
    TimerOutput::Scope(*timer, "Compute error.");
    {
        std::stringstream ss;
        ss << "Time of the TemperatureField does not "
              "the match time of the IMEXTimeStepper." << std::endl
           <<  "TemperatureField::get_time() returns " << temperature_function.get_time()
           << ", which is not equal to " << timestepper.now()
           << ", which is return by IMEXTimeStepper::now()" << std::endl;

        Assert(timestepper.now() == temperature_function.get_time(),
               ExcMessage(ss.str().c_str()));
    }

    const DoFHandler<dim>& temperature_dof_handler = solver.get_dof_handler();
    const LA::Vector& temperature_solution = solver.get_solution();

    const QGauss<dim>   quadrature(solver.get_fe().degree + 2);

    Vector<double>   cellwise_error(triangulation.n_active_cells());

    VectorTools::integrate_difference(mapping,
                                      temperature_dof_handler,
                                      temperature_solution,
                                      temperature_function,
                                      cellwise_error,
                                      quadrature,
                                      VectorTools::L2_norm);

    const double global_error
    = VectorTools::compute_global_error(triangulation,
                                        cellwise_error,
                                        VectorTools::L2_norm);

    AssertIsFinite(global_error);
    Assert(global_error > 0.0,
           ExcLowerRangeType<double>(global_error, 0.0));

    return global_error;
}


template<int dim>
double HeatConductionSquare<dim>::compute_cfl_number() const
{
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

    const DoFHandler<dim>& temperature_dof_handler = solver.get_dof_handler();
    const FiniteElement<dim>& temperature_fe = solver.get_fe();

    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            temperature_fe.degree);

    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping,
                            temperature_fe,
                            quadrature_formula,
                            update_quadrature_points);

    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    double max_cfl = 0;

    for (auto cell: temperature_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);

            convection_function->value_list(fe_values.get_quadrature_points(),
                                            velocity_values);

            double  max_cell_velocity = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                max_cell_velocity = std::max(max_cell_velocity,
                                             velocity_values[q].norm());
            }
            max_cfl = std::max(max_cfl,
                               max_cell_velocity / (cell->diameter() * std::sqrt(dim)));
        }

    const double polynomial_degree = double(temperature_fe.degree);

    const double local_cfl = max_cfl * timestepper.step_size() / polynomial_degree;

    const double global_cfl
    = Utilities::MPI::max(local_cfl, mpi_communicator);

    return global_cfl;
}

template <int dim>
void HeatConductionSquare<dim>::run()
{
    timer->enter_subsection("Setup grid.");

    pcout << "Running a " << dim << "D heat conduction problem "
          << "using " << timestepper.name()
          << ", Q"  << solver.get_fe().degree
          << " elements on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator) << " processes."
          << std::endl;

    {
        GridGenerator::hyper_cube(triangulation);

        const double tol = 1e-12;

        /*
         * enumeration of the boundaries
         *
         *     *--- 2 ---*
         *     |         |
         *
         *     3         1
         *
         *     |         |
         *     *--- 0 ---*
         *                             *
         */
        for (auto cell: triangulation.active_cell_iterators())
            if (cell->at_boundary())
                for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->face(f)->at_boundary())
                    {
                        std::vector<Point<dim>> points(GeometryInfo<dim>::vertices_per_face);
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            points[v] = cell->face(f)->vertex(v);

                        if (std::all_of(points.begin(), points.end(),
                                [&](Point<dim> &p)->bool{return std::abs(p[1]) < tol;}))
                            cell->face(f)->set_boundary_id(0);
                        if (std::all_of(points.begin(), points.end(),
                                [&](Point<dim> &p)->bool{return std::abs(p[0] - 1.0) < tol;}))
                            cell->face(f)->set_boundary_id(1);
                        if (std::all_of(points.begin(), points.end(),
                                [&](Point<dim> &p)->bool{return std::abs(p[1] - 1.0) < tol;}))
                            cell->face(f)->set_boundary_id(2);
                        if (std::all_of(points.begin(), points.end(),
                                [&](Point<dim> &p)->bool{return std::abs(p[0]) < tol;}))
                            cell->face(f)->set_boundary_id(3);
                    }

        std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>>
        periodicity_vector;

        GridTools::collect_periodic_faces(triangulation,
                                          0,
                                          2,
                                          1,
                                          periodicity_vector);
        GridTools::collect_periodic_faces(triangulation,
                                          1,
                                          3,
                                          0,
                                          periodicity_vector);

        triangulation.add_periodicity(periodicity_vector);

        triangulation.refine_global(parameters.n_global_refinements);
    }

    boundary_conditions->set_periodic_bc(0, 1, 3);
    boundary_conditions->set_periodic_bc(1, 0, 2);

    timer->leave_subsection();

    solver.set_convection_function(convection_function);

    const double n_steps = parameters.time_stepping_params.n_steps;

    // spatial convergence study
    {
    ConvergenceTable    spatial_convergence_table;
    parameters.time_stepping_params.adaptive_timestep = false;
    parameters.time_stepping_params.n_steps = 100;
    const unsigned int n_cycles = std::max(int(parameters.n_max_levels - parameters.n_global_refinements),
                                           3);
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
        // reset objects
        solver.set_post_refinement();
        timestepper.reset();
        convection_function->set_time(timestepper.now());
        temperature_function.set_time(timestepper.now());

        // run cycle
        solver.setup_problem();

        pcout << "Cycle: " << Utilities::int_to_string(cycle, 2) << ", "
              << "n_cells: " << Utilities::to_string(triangulation.n_global_active_cells(), 8) << ", "
              << "n_dofs: "  << Utilities::to_string(solver.n_dofs(), 8)
              << std::endl;

        solver.setup_initial_condition(temperature_function);

        pcout << "Start time integration..." << std::endl;
        double max_cfl = 0;
        while (timestepper.at_end() == false)
        {
            // solve problem
            solver.advance_in_time();

            // The solver succeeded, therefore
            // advance time in timestepper
            timestepper.advance_in_time();

            // update time of relevant objects
            convection_function->set_time(timestepper.now());
            temperature_function.set_time(timestepper.now());

            // determine new timestep
            {
                const double cfl = compute_cfl_number();
                max_cfl = std::max(max_cfl, cfl);
                if (cfl > parameters.cfl_max || cfl < parameters.cfl_min)
                {
                    const double desired_time_step
                    = 0.5 * (parameters.cfl_min + parameters.cfl_max)
                                    * timestepper.step_size() / cfl;

                    timestepper.set_time_step(desired_time_step);
                }
            }
        }
        timestepper.print_info(pcout);

        // postprocessing of solution
        const double l2_error = compute_l2_error();
        spatial_convergence_table.add_value("cycle", cycle);
        spatial_convergence_table.add_value("cells", triangulation.n_global_active_cells());
        spatial_convergence_table.add_value("n_dofs", solver.n_dofs());
        spatial_convergence_table.add_value("L2 error", l2_error);
        spatial_convergence_table.add_value("max cfl", max_cfl);
        pcout << "   L2-error: " << l2_error << std::endl;
        pcout << "   max(cfl): " << max_cfl << std::endl;

        // output
        output_results();

        // refinement
        triangulation.refine_global();
    }
    spatial_convergence_table.set_precision("L2 error", 3);
    spatial_convergence_table.set_scientific("L2 error", true);

    spatial_convergence_table.evaluate_convergence_rates("L2 error", ConvergenceTable::reduction_rate);
    spatial_convergence_table.evaluate_convergence_rates("L2 error", ConvergenceTable::reduction_rate_log2);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        spatial_convergence_table.write_text(pcout.get_stream());
    }

    for (unsigned int i=0; i<2; ++i)
    {
        for (auto cell: triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
                cell->set_coarsen_flag();
        triangulation.execute_coarsening_and_refinement();
    }

    {

    ConvergenceTable    temporal_convergence_table;
    double step_size = 100.0 * parameters.time_stepping_params.initial_timestep;
    parameters.time_stepping_params.n_steps = n_steps;
    const unsigned int n_cycles = 6;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle, step_size /= 2.0)
    {
        // reset objects
        solver.set_post_refinement();
        timestepper.reset(step_size);
        convection_function->set_time(timestepper.now());
        temperature_function.set_time(timestepper.now());

        // run cycle
        solver.setup_problem();

        pcout << "Cycle: " << Utilities::int_to_string(cycle, 2) << ", "
              << "n_cells: " << Utilities::to_string(triangulation.n_global_active_cells(), 8) << ", "
              << "n_dofs: "  << Utilities::to_string(solver.n_dofs(), 8) << ", "
              << "step_size: "  << step_size
              << std::endl;

        solver.setup_initial_condition(temperature_function);

        pcout << "Start time integration..." << std::endl;
        double max_cfl = 0;
        while (timestepper.at_end() == false)
        {
            // solve problem
            solver.advance_in_time();

            // The solver succeeded, therefore
            // advance time in timestepper
            timestepper.advance_in_time();

            // update time of relevant objects
            convection_function->set_time(timestepper.now());
            temperature_function.set_time(timestepper.now());

            // determine new timestep
            {
                const double cfl = compute_cfl_number();
                max_cfl = std::max(max_cfl, cfl);
                if (cfl > parameters.cfl_max || cfl < parameters.cfl_min)
                {
                    const double desired_time_step
                    = 0.5 * (parameters.cfl_min + parameters.cfl_max)
                                    * timestepper.step_size() / cfl;

                    timestepper.set_time_step(desired_time_step);
                }
            }
        }
        timestepper.print_info(pcout);

        // postprocessing of solution
        const double l2_error = compute_l2_error();
        temporal_convergence_table.add_value("cycle", cycle);
        temporal_convergence_table.add_value("step size", step_size);
        temporal_convergence_table.add_value("L2 error", l2_error);
        temporal_convergence_table.add_value("max cfl", max_cfl);
        pcout << "   L2-error: " << l2_error << std::endl;
        pcout << "   max(cfl): " << max_cfl << std::endl;
    }
    temporal_convergence_table.set_precision("L2 error", 3);
    temporal_convergence_table.set_scientific("L2 error", true);

    temporal_convergence_table.evaluate_convergence_rates("L2 error", "step size", ConvergenceTable::reduction_rate);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        temporal_convergence_table.write_text(pcout.get_stream());
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

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
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
