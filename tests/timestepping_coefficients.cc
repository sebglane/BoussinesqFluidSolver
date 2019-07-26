/*
 * timestepping_coefficients.cc
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */
#include <iostream>
#include <exception>
#include <functional>
#include <vector>

#include <adsolic/timestepping.h>

using namespace TimeStepping;

void checkTimeStepper(const TimeSteppingParameters &parameters)
{
    IMEXTimeStepping        timestepper(parameters);

    if (parameters.adaptive_timestep)
    {
        std::cout << "Adaptive time stepping with "
                  << timestepper.name() << " scheme" << std::endl;

        const std::vector<double> timesteps({0.1,0.1,0.1,0.05,0.15,0.9});

        while (!timestepper.at_end())
        {
            timestepper.print_info(std::cout);
            timestepper.write(std::cout);
            /*
             * solve problem
             *
             * compute desired time step
             */
            timestepper.set_time_step(timesteps[timestepper.step_no()]);
            timestepper.advance_in_time();
        }
    }
    else
    {
        std::cout << "Fixed time stepping with "
                  << timestepper.name() << " scheme" << std::endl;

        const unsigned int max_cnt = 10;

        while (!timestepper.at_end() && timestepper.step_no() < max_cnt)
        {
            timestepper.print_info(std::cout);
            timestepper.write(std::cout);
            /*
             * solve problem
             *
             * compute desired time step
             */
            timestepper.set_time_step(1.0);
            timestepper.advance_in_time();
        }
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
            parameter_filename = "timestepping_coefficients.prm";

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
