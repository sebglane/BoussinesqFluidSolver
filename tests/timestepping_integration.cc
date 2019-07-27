/*
 * timestepping_integration.cc
 *
 *  Created on: Jul 26, 2019
 *      Author: sg
 */
#include <iostream>
#include <exception>
#include <vector>

#define _USE_MATH_DEFINES // for C++
#include <cmath>

#include <adsolic/timestepping.h>

using namespace TimeStepping;


void checkTimeStepper(const TimeSteppingParameters &parameters)
{
    IMEXTimeStepping        timestepper(parameters);

    const double a = 1.0;

    std::function<double (const double)> f = [&a](const double x) { return a / (x * x); };

    std::function<double (void)> g = [](void) { return 1.0; };

    std::function<double (const double)> y = [&a](const double t)
            { return std::pow(a*(std::exp(3.*t)-1.0)+std::exp(3.*t), 1./3.); };

    std::vector<double> x(3, 1.0);

    const double exact_solution = y(1.0);

    // DGL: x_t  == a / x(t)^2 -x(t), x(t=0) == 1, x(t) = 1 / sqrt(1 + 2 t)

    if (parameters.adaptive_timestep)
    {
        std::cout << "Adaptive time stepping with "
                  << timestepper.name() << " scheme" << std::endl;

        const double inital_step_size = timestepper.step_size();
        const double final_time = timestepper.final();

        std::function<double (const double)> step_size = [&inital_step_size,&final_time](const double t)
                    { return inital_step_size + inital_step_size/2. * std::sin(10.* M_PI * t /final_time) ; };

        while (!timestepper.at_end())
        {
            const std::array<double,3> &alpha = timestepper.alpha();
            const std::array<double,2> &beta = timestepper.beta();
            const std::array<double,3> &gamma = timestepper.gamma();

            const double lhs = alpha[0] / timestepper.step_size() - gamma[0] * g();
            const double rhs
            = -(alpha[1] * x[1] + alpha[2] * x[2]) / timestepper.step_size()
            + beta[0] * f(x[1]) + beta[1] * f(x[2])
            + gamma[1] * g() * x[1] + gamma[2] * g() * x[2];

            x[0] = rhs / lhs;

            timestepper.advance_in_time();

            x[2] = x[1];
            x[1] = x[0];

            timestepper.set_time_step(step_size(timestepper.now()));
        }
        timestepper.print_info(std::cout);

        std::cout << "Solution: " << x[0] << ", "
                  << "suggested value: " << exact_solution << ", "
                  << "error: " << std::abs(x[0] - exact_solution ) << std::endl;
    }
    else
    {
        std::cout << "Fixed time stepping with "
                  << timestepper.name() << " scheme" << std::endl;

        while (!timestepper.at_end())
        {
            const std::array<double,3> &alpha = timestepper.alpha();

            const std::array<double,2> &beta = timestepper.beta();
            const std::array<double,3> &gamma = timestepper.gamma();

            const double lhs = alpha[0] / timestepper.step_size() - gamma[0] * g();
            const double rhs
            = -(alpha[1] * x[1] + alpha[2] * x[2]) / timestepper.step_size()
            + beta[0] * f(x[1]) + beta[1] * f(x[2])
            + gamma[1] * g() * x[1] + gamma[2] * g() * x[2];

            x[0] = rhs / lhs;

            timestepper.advance_in_time();

            x[2] = x[1];
            x[1] = x[0];

            timestepper.set_time_step(parameters.initial_timestep);
        }
        timestepper.print_info(std::cout);

        std::cout << "Solution: " << x[0] << ", "
                  << "suggested value: " << exact_solution  << ", "
                  << "error: " << std::abs(x[0] - exact_solution ) << std::endl;
    }

    return;
}

int main(int argc, char **argv)
{
    try
    {
        std::string parameter_filename;
        if (argc>1)
            parameter_filename = argv[1];
        else
            parameter_filename = "timestepping_integration.prm";

        TimeSteppingParameters  parameters(parameter_filename);

        // fixed time steps
        parameters.adaptive_timestep = false;

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::Euler;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::SBDF;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::CNAB;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::MCNAB;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::CNLF;
        checkTimeStepper(parameters);

        // adaptive time steps
        parameters.adaptive_timestep = true;

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::SBDF;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::CNAB;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::MCNAB;
        checkTimeStepper(parameters);

        std::cout << "================================="
                     "========================================" << std::endl;
        parameters.imex_scheme = IMEXType::CNLF;
        checkTimeStepper(parameters);
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
