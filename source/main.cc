#include <iostream>
#include <exception>

#include "conducting_fluid_solver.h"

int main(int argc, char *argv[])
{
    using namespace dealii;
    using namespace ConductingFluid;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                        numbers::invalid_unsigned_int);
    try
    {
        std::string parameter_filename;
        if (argc>=2)
            parameter_filename = argv[1];
        else
            parameter_filename = "default_parameters.prm";

        ConductingFluidSolver<3> problem_3D(1e-3, 10, 1);
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
