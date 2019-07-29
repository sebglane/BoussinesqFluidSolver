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

#include <deal.II/base/convergence_table.h>

#include <adsolic/timestepping.h>

using namespace TimeStepping;

using namespace dealii;

std::vector<double> checkTimeStepper(const TimeSteppingParameters &parameters)
{
    IMEXTimeStepping        timestepper(parameters);

    const double a = 2.0;

    std::function<double (const double)> f = [&a](const double x) { return a / (x * x); };

    std::function<double (void)> g = [](void) { return 1.0; };

    std::function<double (const double)> y = [&a](const double t)
            { return std::pow(a*(std::exp(3.*t)-1.0)+std::exp(3.*t), 1./3.); };

    std::vector<double> x(3, 1.0);

    // DGL: dx(t)/dt == a/x(t)^2 - x(t), x(0) == 1
    const double exact_solution = y(timestepper.final());

    if (parameters.adaptive_timestep)
    {
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

        return std::vector<double>{std::abs(x[0] - exact_solution ),
                                   std::abs(x[0] - exact_solution) / exact_solution};
    }
    else
    {
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
        }
        return std::vector<double>{std::abs(x[0] - exact_solution ),
                                   std::abs(x[0] - exact_solution) / exact_solution};
    }
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
        const double initial_timestep = parameters.initial_timestep;

        // fixed time steps
        parameters.adaptive_timestep = false;
        {
            ConvergenceTable    convergence_table;
            for (unsigned int cycle=0; cycle<10; ++cycle, parameters.initial_timestep /= 2.0)
            {
                convergence_table.add_value("cycle", cycle);
                convergence_table.add_value("step size", parameters.initial_timestep);

                parameters.imex_scheme = IMEXType::Euler;
                std::vector<double> err = checkTimeStepper(parameters);
                convergence_table.add_value("Euler abs.", err[0]);
                convergence_table.add_value("Euler rel.", err[1]);

                parameters.imex_scheme = IMEXType::SBDF;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("SBDF abs.", err[0]);
                convergence_table.add_value("SBDF rel.", err[1]);

                parameters.imex_scheme = IMEXType::CNAB;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("CNAB abs.", err[0]);
                convergence_table.add_value("CNAB rel.", err[1]);

                parameters.imex_scheme = IMEXType::MCNAB;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("MCNAB abs.", err[0]);
                convergence_table.add_value("MCNAB rel.", err[1]);

                parameters.imex_scheme = IMEXType::CNLF;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("CNLF abs.", err[0]);
                convergence_table.add_value("CNLF rel.", err[1]);
            }
            convergence_table.set_precision("step size", 3);
            convergence_table.set_precision("Euler abs.", 3);
            convergence_table.set_precision("Euler rel.", 3);
            convergence_table.set_precision("SBDF abs.", 3);
            convergence_table.set_precision("SBDF rel.", 3);
            convergence_table.set_precision("CNAB abs.", 3);
            convergence_table.set_precision("CNAB rel.", 3);
            convergence_table.set_precision("MCNAB abs.", 3);
            convergence_table.set_precision("MCNAB rel.", 3);
            convergence_table.set_precision("CNLF abs.", 3);
            convergence_table.set_precision("CNLF rel.", 3);

            convergence_table.set_scientific("step size", true);
            convergence_table.set_scientific("Euler abs.", true);
            convergence_table.set_scientific("Euler rel.", true);
            convergence_table.set_scientific("SBDF abs.", true);
            convergence_table.set_scientific("SBDF rel.", true);
            convergence_table.set_scientific("CNAB abs.", true);
            convergence_table.set_scientific("CNAB rel.", true);
            convergence_table.set_scientific("MCNAB abs.", true);
            convergence_table.set_scientific("MCNAB rel.", true);
            convergence_table.set_scientific("CNLF abs.", true);
            convergence_table.set_scientific("CNLF rel.", true);

            convergence_table
            .omit_column_from_convergence_rate_evaluation("cycle");
            convergence_table
            .omit_column_from_convergence_rate_evaluation("step size");

            convergence_table
            .evaluate_all_convergence_rates("step size", ConvergenceTable::reduction_rate);

            std::cout << "================================="
                         "========================================" << std::endl;
            std::cout << "===== Fixed step size ==========="
                         "========================================" << std::endl;
            convergence_table.write_text(std::cout);
            std::cout << "================================="
                         "========================================" << std::endl;
        }
        // adaptive time steps
        parameters.adaptive_timestep = true;
        parameters.initial_timestep = initial_timestep;
        {
            ConvergenceTable    convergence_table;
            for (unsigned int cycle=0; cycle<10; ++cycle, parameters.initial_timestep /= 2.0)
            {
                convergence_table.add_value("cycle", cycle);
                convergence_table.add_value("step size", parameters.initial_timestep);

                parameters.imex_scheme = IMEXType::Euler;
                std::vector<double> err = checkTimeStepper(parameters);
                convergence_table.add_value("Euler abs.", err[0]);
                convergence_table.add_value("Euler rel.", err[1]);

                parameters.imex_scheme = IMEXType::SBDF;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("SBDF abs.", err[0]);
                convergence_table.add_value("SBDF rel.", err[1]);

                parameters.imex_scheme = IMEXType::CNAB;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("CNAB abs.", err[0]);
                convergence_table.add_value("CNAB rel.", err[1]);

                parameters.imex_scheme = IMEXType::MCNAB;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("MCNAB abs.", err[0]);
                convergence_table.add_value("MCNAB rel.", err[1]);

                parameters.imex_scheme = IMEXType::CNLF;
                err = checkTimeStepper(parameters);
                convergence_table.add_value("CNLF abs.", err[0]);
                convergence_table.add_value("CNLF rel.", err[1]);
            }
            convergence_table.set_precision("step size", 3);
            convergence_table.set_precision("Euler abs.", 3);
            convergence_table.set_precision("Euler rel.", 3);
            convergence_table.set_precision("SBDF abs.", 3);
            convergence_table.set_precision("SBDF rel.", 3);
            convergence_table.set_precision("CNAB abs.", 3);
            convergence_table.set_precision("CNAB rel.", 3);
            convergence_table.set_precision("MCNAB abs.", 3);
            convergence_table.set_precision("MCNAB rel.", 3);
            convergence_table.set_precision("CNLF abs.", 3);
            convergence_table.set_precision("CNLF rel.", 3);

            convergence_table.set_scientific("step size", true);
            convergence_table.set_scientific("Euler abs.", true);
            convergence_table.set_scientific("Euler rel.", true);
            convergence_table.set_scientific("SBDF abs.", true);
            convergence_table.set_scientific("SBDF rel.", true);
            convergence_table.set_scientific("CNAB abs.", true);
            convergence_table.set_scientific("CNAB rel.", true);
            convergence_table.set_scientific("MCNAB abs.", true);
            convergence_table.set_scientific("MCNAB rel.", true);
            convergence_table.set_scientific("CNLF abs.", true);
            convergence_table.set_scientific("CNLF rel.", true);

            convergence_table
            .omit_column_from_convergence_rate_evaluation("cycle");
            convergence_table
            .omit_column_from_convergence_rate_evaluation("step size");

            convergence_table
            .evaluate_all_convergence_rates("step size", ConvergenceTable::reduction_rate);

            std::cout << "================================="
                    "========================================" << std::endl;
            std::cout << "===== Adaptive step size ========"
                    "========================================" << std::endl;
            convergence_table.write_text(std::cout);
            std::cout << "================================="
                    "========================================" << std::endl;
        }
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
