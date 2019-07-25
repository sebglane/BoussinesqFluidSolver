/*
 * timestepping.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */
#include <deal.II/base/conditional_ostream.h>

#include <adsolic/timestepping.h>
#include <adsolic/parameters.h>

namespace TimeStepping
{

TimeSteppingParameters::TimeSteppingParameters()
:
imex_scheme(TimeStepping::CNAB),
n_steps(100),
adaptive_timestep(true),
adaptive_timestep_barrier(2),
initial_timestep(1e-3),
min_timestep(1e-9),
max_timestep(1e-3),
final_time(1.0),
verbose(false)
{}

TimeSteppingParameters::TimeSteppingParameters(const std::string &parameter_filename)
:
TimeSteppingParameters()
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

void TimeSteppingParameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time stepping settings");
    {
        prm.declare_entry("time_stepping_scheme",
                "CNAB",
                Patterns::Selection("Euler|CNAB|MCNAB|CNLF|SBDF"),
                "Time stepping scheme applied.");

        prm.declare_entry("n_steps",
                "10",
                Patterns::Integer(),
                "Maximum number of time steps.");

        prm.declare_entry("adaptive_timestep",
                "true",
                Patterns::Bool(),
                "Turn adaptive time stepping on or off");

        prm.declare_entry("adaptive_timestep_barrier",
                "2",
                Patterns::Integer(),
                "Time step after which adaptive time stepping is applied.");

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

        prm.declare_entry("final_time",
                "1.0",
                Patterns::Double(0.),
                "Final time.");

        prm.declare_entry("verbose",
                "false",
                Patterns::Bool(),
                "Activate verbose output.");
    }
    prm.leave_subsection();
}

void TimeSteppingParameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time stepping settings");
    {
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

        adaptive_timestep = prm.get_bool("adaptive_timestep");
        if (adaptive_timestep)
            adaptive_timestep_barrier = prm.get_integer("adaptive_timestep_barrier");
            Assert(adaptive_timestep_barrier > 0,
                   ExcLowerRange(adaptive_timestep_barrier, 0));

        n_steps = prm.get_integer("n_steps");
        Assert(n_steps > 0, ExcLowerRange(n_steps, 0));

        initial_timestep = prm.get_double("dt_initial");
        Assert(initial_timestep > 0,
               ExcLowerRangeType<double>(initial_timestep, 0));

        if (adaptive_timestep)
        {
            min_timestep = prm.get_double("dt_min");
            Assert(min_timestep > 0,
                   ExcLowerRangeType<double>(min_timestep, 0));

            max_timestep = prm.get_double("dt_max");
            Assert(max_timestep > 0,
                   ExcLowerRangeType<double>(max_timestep, 0));

            Assert(min_timestep < max_timestep,
                   ExcLowerRangeType<double>(min_timestep, min_timestep));
            Assert(min_timestep <= initial_timestep,
                   ExcLowerRangeType<double>(min_timestep, initial_timestep));
            Assert(initial_timestep <= max_timestep,
                   ExcLowerRangeType<double>(initial_timestep, max_timestep));
        }

        final_time = prm.get_double("final_time");
        Assert(final_time > 0.0, ExcLowerRangeType<double>(final_time, 0.0));
        Assert(initial_timestep < final_time,
               ExcLowerRangeType<double>(initial_timestep, final_time));

        verbose = prm.get_bool("verbose");
    }
    prm.leave_subsection();
}

template<typename Stream>
void TimeSteppingParameters::write
(Stream &stream) const
{
    stream << "Time stepping parameters" << std::endl
           << "   imex_scheme: ";
    switch (imex_scheme)
    {
    case IMEXType::Euler:
         stream << "Euler" << std::endl;
        break;
    case IMEXType::CNAB:
        stream << "CNAB" << std::endl;
        break;
    case IMEXType::MCNAB:
        stream << "MCNAB" << std::endl;
        break;
    case IMEXType::CNLF:
        stream << "CNLF" << std::endl;
        break;
    case IMEXType::SBDF:
        stream << "SBDF" << std::endl;
        break;
    }
    stream << "   n_steps: " << n_steps << std::endl
           << "   adaptive_timestep: " << (adaptive_timestep? "true": "false") << std::endl
           << "   adaptive_timestep_barrier: " << adaptive_timestep_barrier << std::endl
           << "   initial_timestep: " << initial_timestep << std::endl
           << "   min_timestep: " << min_timestep << std::endl
           << "   max_timestep: " << max_timestep << std::endl
           << "   final_time: " << final_time << std::endl
           << "   verbose: " << (verbose? "true": "false") << std::endl;
}

IMEXTimeStepping::IMEXTimeStepping(const TimeSteppingParameters &prm)
:
type(prm.imex_scheme),
start_time(0.0),
end_time(prm.final_time),
start_step_val(prm.initial_timestep),
min_step_val(prm.min_timestep),
max_step_val(prm.max_timestep),
step_val(start_step_val),
desired_step_val(start_step_val),
old_step_val(0.0),
old_old_step_val(0.0),
omega(1.0),
current_time(0.0),
previous_time(0.0),
pre_previous_time(0.0),
old_extrapol_factor(1.0),
old_old_extrapol_factor(0.0),
adaptive_timestep(prm.adaptive_timestep),
adaptive_barrier(prm.adaptive_timestep_barrier),
step_no_val(0),
max_step_no(prm.n_steps),
at_end_time(false),
verbose(prm.verbose)
{
    alpha_array[0] = 1.0;
    alpha_array[1] = -1.0;
    alpha_array[2] = 0.;

    beta_array[0] = 1.;
    beta_array[1] = 0.;

    gamma_array[0] = 1.0;
    gamma_array[1] = 0.0;
    gamma_array[2] = 0.0;
}

double IMEXTimeStepping::advance_time_step()
{
    Assert(at_end_time == false,
           ExcMessage("Final time already reached, cannot proceed"));

    if (adaptive_timestep && step_no_val >= adaptive_barrier)
        advance_adaptive();
    else
        advance_fixed();

    return current_time;
}

void IMEXTimeStepping::advance_fixed()
{
    Assert(omega == 1.0,
           ExcMessage("Time step ration is not equal to one."));

    old_old_step_val = old_step_val;
    old_step_val = step_val;

    // Try incrementing time by tentative step size
    double tentative_time = current_time + step_val;

    // Check if we shot over the final time and set
    // the flag that the final time is reached.
    // In fixed time stepping we do not adjust
    // the final time.
    if (!at_end_time && tentative_time > end_time)
        at_end_time = true;

    // Update the coefficients after the second step. The first two step are
    // Euler steps because the initial condition might be inconsistent.
    // In the third step a second-order scheme may be used.
    if (step_no_val == 1)
    {
        update_alpha();
        update_beta();
        update_gamma();

        // compute weights for extrapolation. Do not extrapolate in second time
        // step because initial condition might not have been consistent
        update_extrapol_factors();

        coefficients_changed = true;
    }
    else if (step_no_val == 2)
    {
        coefficients_changed = false;
    }

    pre_previous_time = previous_time;
    previous_time = current_time;
    current_time = tentative_time;
    step_no_val++;
}


void IMEXTimeStepping::advance_adaptive()
{
    double tentative_step_val = desired_step_val;
    // Do time step control, but not in
    // first step.
    if (current_time != start())
    {
        old_old_step_val = old_step_val;
        old_step_val = step_val;
    }

    // Try incrementing time by tentative step size
    double tentative_time = current_time + tentative_step_val;
    step_val = tentative_step_val;

    // If we just missed the final time, increase
    // the step size a bit. This way, we avoid a
    // very small final step. If the step shot
    // over the final time, adjust it so we hit
    // the final time exactly.
    double small_step = .01 * tentative_step_val;
    if (!at_end_time && tentative_time > end_time - small_step)
    {
        step_val = end_time - current_time;
        tentative_time = end_time;
        at_end_time = true;
    }

    // Update the coefficients if necessary
    const double old_omega = omega;
    const double tentative_omega = step_val / old_step_val;
    if (std::fabs(tentative_omega - old_omega) > 1e-12)
    {
        omega = tentative_omega;

        update_alpha();
        update_beta();
        update_gamma();

        coefficients_changed = true;

      // compute weights for extrapolation. Do not extrapolate in second time
      // step because initial condition might not have been consistent
      if (step_no_val > 1)
          update_extrapol_factors();
    }
    else
    {
        tentative_time = current_time + old_step_val;
        step_val = old_step_val;
        coefficients_changed = false;
    }

    pre_previous_time = previous_time;
    previous_time = current_time;
    current_time = tentative_time;
    step_no_val++;
}

std::string IMEXTimeStepping::name() const
{
    switch (type)
    {
    case IMEXType::Euler:
        return std::string("Euler");
        break;
    case IMEXType::CNAB:
        return std::string("CNAB");
        break;
    case IMEXType::MCNAB:
        return std::string("MCNAB");
        break;
    case IMEXType::CNLF:
        return std::string("CNLF");
        break;
    case IMEXType::SBDF:
        return std::string("SBDF");
        break;
    default:
        AssertThrow(false, ExcInternalError());
        break;
    }
}

IMEXType IMEXTimeStepping::scheme() const
{
    return type;
}

void IMEXTimeStepping::set_time_step(double desired_value)
{
    if (!adaptive_timestep ||
        (adaptive_timestep && step_no_val < adaptive_barrier) ||
        step_no() < 1)
        return;

    const double damping_factor = 1.0; // 0.5;

    // When setting a new time step size one needs to consider three things:
    //  - That it is not smaller than the minimum given
    //  - That it is not larger than the maximum step size given
    //  - That the change from the previous value is not too big, which should
    //    be fulfilled automatically in this case because we look at quantities
    //    that vary slowly.
    //
    // Therefore we check if the damping is activated.
    if (damping_factor != 1.0)
    {
        desired_step_val = std::min(step_size() / damping_factor,
                                    std::max(desired_value, damping_factor * step_size()));
    }
    else
        desired_step_val = desired_value;

    Assert(desired_value >= min_step_val,
           ExcLowerRangeType<double>(desired_value, min_step_val));
    Assert(max_step_val >= desired_value,
           ExcLowerRangeType<double>(max_step_val, desired_value));
}

template<typename Stream>
void IMEXTimeStepping::write(Stream &stream) const
{
    if (verbose == false)
        return;

    stream << "+-----------+----------+----------+----------+\n"
           << "|   Index   |    n+1   |    n     |    n-1   |\n"
           << "+-----------+----------+----------+----------+\n"
           << "|   alpha   | ";
    for (const auto it: alpha_array)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "|   beta    |    -     | ";
    for (const auto it: beta_array)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "|   gamma   | ";
    for (const auto it: gamma_array)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "| extra_pol |    -     | ";
    for (const auto it: std::array<double,2>({old_extrapol_factor,
                                              old_old_extrapol_factor}))
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
    stream << std::fixed << std::setprecision(6);
    return;
}

std::array<double,3> IMEXTimeStepping::alpha() const
{
    return alpha_array;
}

std::array<double,2> IMEXTimeStepping::beta() const
{
    return beta_array;
}

std::array<double,3> IMEXTimeStepping::gamma() const
{
    return gamma_array;
}

void IMEXTimeStepping::update_alpha()
{
    if (type == IMEXType::SBDF)
    {
        alpha_array[0] = (1. + 2. * omega) / (1. + omega);
        alpha_array[1] = -(1. + omega);
        alpha_array[2] = (omega * omega) / (1. + omega);
    }
    else if (type == IMEXType::CNAB || type == IMEXType::MCNAB)
    {
        alpha_array[0] = 1.0;
        alpha_array[1] = -1.0;
    }
    else if (type == IMEXType::CNLF)
    {
        alpha_array[0] = 1. / (1. + omega);
        alpha_array[1] = omega - 1.;
        alpha_array[2] = -(omega * omega) / (1. + omega);
    }
    else if (type == IMEXType::Euler)
    {
        alpha_array[0] = 1.0;
        alpha_array[1] = -1.0;
        alpha_array[2] = 0.;
    }
}

void IMEXTimeStepping::update_beta()
{
    if (type == IMEXType::SBDF)
    {
        beta_array[0] = (1. + omega);
        beta_array[1] = -omega;
    }
    else if (type == IMEXType::CNAB ||  type == IMEXType::MCNAB)
    {
        beta_array[0] = (1. + 0.5 * omega);
        beta_array[1] = -0.5 * omega;
    }
    else if (type == IMEXType::CNLF)
    {
        beta_array[0] = 1.;
        beta_array[1] = 0.;
    }
    else if (type == IMEXType::Euler)
    {
        beta_array[0] = 1.;
        beta_array[1] = 0.;
    }
}

void IMEXTimeStepping::update_gamma()
{
    if (type == IMEXType::SBDF)
    {
        gamma_array[0] = 1.0;
        gamma_array[1] = 0.0;
    }
    else if (type == IMEXType::CNAB)
    {
        gamma_array[0] = 0.5;
        gamma_array[1] = 0.5;
    }
    else if (type == IMEXType::MCNAB)
    {
        gamma_array[0] = (8. * omega + 1.)/ (16. * omega);
        gamma_array[1] = (7. * omega - 1.)/ (16. * omega);
        gamma_array[2] = omega / (16. * omega);
    }
    else if (type == IMEXType::CNLF)
    {
        gamma_array[0] = 0.5 / omega;
        gamma_array[1] = 0.5 * (1. - 1./omega);
        gamma_array[2] = 0.5;
    }
    else if (type == IMEXType::Euler)
    {
        gamma_array[0] = 1.0;
        gamma_array[1] = 0.0;
        gamma_array[2] = 0.0;
    }
}

void IMEXTimeStepping::update_extrapol_factors()
{
    old_extrapol_factor = 1. + omega;
    old_old_extrapol_factor = omega;
}

// explicit instantiation
template void TimeSteppingParameters::write(std::ostream &) const;
template void TimeSteppingParameters::write(ConditionalOStream &) const;

template void IMEXTimeStepping::write(std::ostream &) const;
template void IMEXTimeStepping::write(ConditionalOStream &) const;

}  // namespace TimeStepping
