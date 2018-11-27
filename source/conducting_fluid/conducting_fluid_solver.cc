/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "conducting_fluid_solver.h"
#include "grid_factory.h"
#include "initial_values.h"
#include "postprocessor.h"

namespace ConductingFluid {

template<int dim>
ConductingFluidSolver<dim>::ConductingFluidSolver(
        const double       &timestep,
        const unsigned int &n_steps,
        const unsigned int &output_frequency,
        const unsigned int &refinement_frequency)
:
magnetic_degree(1),
imex_coefficients(TimeStepping::IMEXType::CNAB),
triangulation(),
mapping(4),
// magnetic part
interior_magnetic_fe(FE_Nedelec<dim>(magnetic_degree), 1,
                     FE_Nothing<dim>(), 1),
exterior_magnetic_fe(FE_Nothing<dim>(dim), 1,
                     FE_Q<dim>(magnetic_degree), 1),
magnetic_dof_handler(triangulation),
// coefficients
equation_coefficients{1.0},
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
// time stepping
timestep(timestep),
old_timestep(timestep),
// TODO: goes to parameter file later
n_steps(n_steps),
output_frequency(output_frequency),
refinement_frequency(refinement_frequency)
{
    fe_collection.push_back(interior_magnetic_fe);
    fe_collection.push_back(exterior_magnetic_fe);
}

template<int dim>
void ConductingFluidSolver<dim>::output_results() const
{
    std::cout << "   Output results..." << std::endl;

    std::vector<std::string> solution_names(dim,"vector_potential");
    solution_names.push_back("scalar_potential");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // prepare data out object
    DataOut<dim, hp::DoFHandler<dim>>    data_out;
    data_out.attach_dof_handler(magnetic_dof_handler);
    data_out.add_data_vector(magnetic_solution,
                             solution_names,
                             DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(timestep_number, 5) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

/*
 *
template<int dim>
void ConductingFluidSolver<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    triangulation.set_all_refine_flags();

    // preparing magnetic solution transfer
    std::vector<BlockVector<double>> x_magnetic(3);
    x_magnetic[0] = magnetic_solution;
    x_magnetic[1] = old_magnetic_solution;
    x_magnetic[2] = old_old_magnetic_solution;
    SolutionTransfer<dim,BlockVector<double>> magnetic_transfer(magnetic_dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    magnetic_transfer.prepare_for_coarsening_and_refinement(x_magnetic);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of magnetic solution
    {
        std::vector<BlockVector<double>>    tmp_magnetic(3);
        tmp_magnetic[0].reinit(magnetic_solution);
        tmp_magnetic[1].reinit(magnetic_solution);
        tmp_magnetic[2].reinit(magnetic_solution);
        magnetic_transfer.interpolate(x_magnetic, tmp_magnetic);

        magnetic_solution = tmp_magnetic[0];
        old_magnetic_solution = tmp_magnetic[1];
        old_old_magnetic_solution = tmp_magnetic[2];

        magnetic_constraints.distribute(magnetic_solution);
        magnetic_constraints.distribute(old_magnetic_solution);
        magnetic_constraints.distribute(old_old_magnetic_solution);
    }
    // set rebuild flags
    rebuild_magnetic_matrices = true;
}
 *
 */


template <int dim>
void ConductingFluidSolver<dim>::solve()
{
    std::cout << "   Solving magnetic system..." << std::endl;

    TimerOutput::Scope  timer_section(computing_timer, "magnetic solve");

    magnetic_constraints.set_zero(magnetic_solution);

    PrimitiveVectorMemory<BlockVector<double>> vector_memory;

    SolverControl solver_control(1000,
            1e-6 * magnetic_rhs.l2_norm());

    SolverGMRES<BlockVector<double>>
    solver(solver_control,
            vector_memory,
            SolverGMRES<BlockVector<double>>::AdditionalData(30, true));

    PreconditionJacobi<BlockSparseMatrix<double>> preconditioner;
    preconditioner.initialize(magnetic_matrix,
                              PreconditionJacobi<BlockSparseMatrix<double>>::AdditionalData());

    solver.solve(magnetic_matrix,
                 magnetic_solution,
                 magnetic_rhs,
                 preconditioner);

    std::cout << "      "
            << solver_control.last_step()
            << " GMRES iterations for magnetic system, "
            << std::endl;

    magnetic_constraints.distribute(magnetic_solution);
}


template<int dim>
std::pair<double, double> ConductingFluidSolver<dim>::compute_rms_values() const
{
    const QGauss<dim> quadrature(magnetic_degree + 2);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(quadrature);
    q_collection.push_back(quadrature);

    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values|
                                   update_JxW_values);

    const FEValuesExtractors::Vector vector_potential(0);
    const FEValuesExtractors::Scalar scalar_potential(dim);

    std::vector<Tensor<1,dim>>  vector_potential_values(q_collection[0].size());
    std::vector<double>         scalar_potential_values(q_collection[1].size());


    double rms_vector_potential = 0;
    double rms_scalar_potential = 0;
    double fluid_volume = 0;
    double vacuum_volume = 0;

    for (auto cell: magnetic_dof_handler.active_cell_iterators())
    {
        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        if (cell->material_id() == DomainIdentifiers::MaterialIds::Fluid)
        {
            AssertDimension(vector_potential_values.size(),
                            fe_values.n_quadrature_points);

            fe_values[vector_potential].get_function_values(magnetic_solution,
                                                            vector_potential_values);

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
                rms_vector_potential += vector_potential_values[q] * vector_potential_values[q] * fe_values.JxW(q);
                fluid_volume += fe_values.JxW(q);
            }
        }
        // vacuum domain
        else if (cell->material_id() == DomainIdentifiers::MaterialIds::Vacuum)
        {
            AssertDimension(scalar_potential_values.size(),
                            fe_values.n_quadrature_points);

            fe_values[scalar_potential].get_function_values(magnetic_solution,
                                                            scalar_potential_values);

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
                rms_scalar_potential += scalar_potential_values[q] * scalar_potential_values[q] * fe_values.JxW(q);
                vacuum_volume += fe_values.JxW(q);
            }
        }
        else
            Assert(false, ExcInternalError());
    }

    rms_vector_potential /= fluid_volume;
    AssertIsFinite(rms_vector_potential);
    Assert(rms_vector_potential >= 0,
           ExcLowerRangeType<double>(rms_vector_potential, 0));

    rms_scalar_potential /= vacuum_volume;
    AssertIsFinite(rms_scalar_potential);
    Assert(rms_scalar_potential >= 0,
           ExcLowerRangeType<double>(rms_scalar_potential, 0));

    return std::pair<double,double>(std::sqrt(rms_vector_potential),
                                    std::sqrt(rms_scalar_potential));
}




template<int dim>
void ConductingFluidSolver<dim>::run()
{
    make_grid();

    setup_dofs();

    const EquationData::MagneticInitialValues<dim> initial_potential;
    const Functions::ZeroFunction<dim>             zero_function(dim+1);


    const std::map<types::material_id, const Function<dim>*>
    initial_values = {{DomainIdentifiers::MaterialIds::Fluid, &initial_potential},
                      {EquationData::BoundaryIds::CMB, &zero_function}};
    VectorTools::interpolate_based_on_material_id(
                             mapping,
                             magnetic_dof_handler,
                             initial_values,
                             old_magnetic_solution);

    magnetic_constraints.distribute(old_magnetic_solution);

    magnetic_solution = old_magnetic_solution;

    output_results();

    double time = 0;

    do
    {
        std::cout << "step: " << Utilities::int_to_string(timestep_number, 8) << ", "
                  << "time: " << time << ", "
                  << "time step: " << timestep
                  << std::endl;

        assemble_magnetic_system();

        solve();
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute rms values");

            const std::pair<double,double> rms_values = compute_rms_values();

            std::cout << "   vector potential rms value: "
                      << rms_values.first
                      << std::endl
                      << "   scalar potential rms value: "
                      << rms_values.second
                      << std::endl;
        }

        if (timestep_number % output_frequency == 0 && timestep_number != 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }
        /*
         *
        // mesh refinement
        if ((timestep_number > 0) && (timestep_number % refinement_frequency == 0))
            refine_mesh();
         *
         */

        // advance magnetic solution
        old_old_magnetic_solution = old_magnetic_solution;
        old_magnetic_solution = magnetic_solution;

        // extrapolate magnetic solution
        magnetic_solution.sadd(1. + timestep / old_timestep,
                               timestep / old_timestep,
                               old_old_magnetic_solution);
        // advance in time
        time += timestep;
        ++timestep_number;

    } while (timestep_number <= n_steps);

    if (n_steps % output_frequency != 0)
        output_results();

    std::cout << std::fixed;

    computing_timer.print_summary();
    computing_timer.reset();

    std::cout << std::endl;
}
}  // namespace ConductingFluid

// explicit instantiation
template class ConductingFluid::ConductingFluidSolver<3>;
