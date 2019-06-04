#include <magnetic_diffusion_solver.h>
#include <iostream>
#include <exception>


int main(int argc, char *argv[])
{
    using namespace dealii;
    using namespace ConductingFluid;

    try
    {
        std::string parameter_filename;
        if (argc>=2)
            parameter_filename = argv[1];
        else
            parameter_filename = "default_parameters.prm";

        MagneticDiffusionSolver<3>    problem_3D;
        problem_3D.run();
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
