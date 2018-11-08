#include <iostream>
#include <exception>
#include <vector>

#include "buoyant_fluid_solver.h"

int main(int argc, char *argv[])
{
    using namespace dealii;
    using namespace BuoyantFluid;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                        numbers::invalid_unsigned_int);
    try
    {
        std::string parameter_filename;
        if (argc>=2)
            parameter_filename = argv[1];
        else
            parameter_filename = "default_parameters.prm";

        const int dim = 2;
        Parameters              parameters_2D(parameter_filename);
        BuoyantFluidSolver<dim> problem_2D(parameters_2D);
        problem_2D.run();
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
