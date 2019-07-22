/*
 * timestepping.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include <timestepping.h>
#include <parameters.h>

namespace TimeStepping
{

IMEXTimeStepping::IMEXTimeStepping(const Parameters &prm)
:
type(prm.imex_scheme),
start_time(0.0),
end_time(prm.final_time),
start_step_val(prm.initial_timestep),
min_step_val(prm.min_timestep),
max_step_val(prm.max_timestep),
step_val(start_step_val),
old_step_val(0.0),
old_old_step_val(0.0),
omega(1.0),
old_extrapol_factor(1.0),
old_old_extrapol_factor(0.0),
step_val(0)
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

    double tentative_step_val = step_val;
    // Do time step control, but not in
    // first step.
    if (current_time != start())
    {
        old_old_step_val = old_step_val;
        old_step_val = step_val;

        if (type == IMEXType::SBDF && step_no_val == 1)
          tentative_step_val = step_val;

        if (tentative_step_val > max_step_val)
            tentative_step_val = max_step_val;
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
    if (std::fabs(omega - old_omega) > 1e-12)
    {
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
        coefficients_changed = false;

    previous_time = current_time;
    current_time = tentative_time;
    step_no_val++;

    return current_time;
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

template<typename Stream>
void IMEXTimeStepping::write(Stream &stream) const
{
    stream << std::endl
           << "+-----------+----------+----------+----------+\n"
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

    stream << std::endl << "|   beta    |    0     | ";
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
    stream << std::endl
           << "+-----------+----------+----------+----------+\n";
}

}  // namespace TimeStepping
