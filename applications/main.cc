#include <iostream>
#include <exception>

#include <adsolic/buoyant_fluid_solver.h>

int main(int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace BuoyantFluid;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 2);

        std::string parameter_filename;
        if (argc>=2)
            parameter_filename = argv[1];
        else
            parameter_filename = "default_parameters.prm";

        Parameters              parameters(parameter_filename);
        if (parameters.dim == 2)
        {
            BuoyantFluidSolver<2>   problem_2D(parameters);
            problem_2D.run();
        }
        else if (parameters.dim == 3)
        {
            BuoyantFluidSolver<3>   problem_3D(parameters);
            problem_3D.run();
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
