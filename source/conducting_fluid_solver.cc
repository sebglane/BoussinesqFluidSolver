/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

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
        const double        &aspect_ratio,
        const double        &timestep,
        const unsigned int  &n_steps,
        const unsigned int  &vtk_frequency,
        const double        &t_final)
:
magnetic_degree(2),
pseudo_pressure_degree(magnetic_degree - 1),
imex_coefficients(TimeStepping::IMEXType::SBDF),
triangulation(Triangulation<dim>::MeshSmoothing::limit_level_difference_at_vertices),
mapping(4),
// magnetic part
magnetic_fe(FESystem<dim>(FE_Q<dim>(magnetic_degree), dim), 1,
            FE_Q<dim>(pseudo_pressure_degree), 1),
magnetic_dof_handler(triangulation),
// coefficients
equation_coefficients{1.0},
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
//// time stepping
timestep(timestep),
old_timestep(timestep),
aspect_ratio(aspect_ratio),
//// TODO: goes to parameter file later
t_final(t_final),
n_steps(n_steps),
vtk_frequency(vtk_frequency),
rms_frequency(10)
{}

template<int dim>
void ConductingFluidSolver<dim>::output_results(const bool initial_condition) const
{
    std::cout << "   Output results..." << std::endl;

    std::vector<std::string> solution_names(dim,"magnetic_field");
    solution_names.push_back("pseudo_pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // prepare data out object
    DataOut<dim, DoFHandler<dim>>    data_out;
    data_out.attach_dof_handler(magnetic_dof_handler);
    data_out.add_data_vector(magnetic_solution,
                             solution_names,
                             DataOut<dim,DoFHandler<dim> >::type_dof_data,
                             data_component_interpretation);

    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  (initial_condition ?
                                  "initial":
                                  Utilities::int_to_string(timestep_number, 5)) +
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

template<int dim>
std::pair<double, double> ConductingFluidSolver<dim>::compute_rms_values() const
{
    const QGauss<dim> quadrature(magnetic_degree + 1);

    FEValues<dim> fe_values(mapping,
                            magnetic_fe,
                            quadrature,
                            update_values|
                            update_JxW_values);

    const unsigned int n_q_points = quadrature.size();

    std::vector<Tensor<1,dim>>  magnetic_field_values(n_q_points);
    std::vector<double>  pseudo_pressure_values(n_q_points);

    const FEValuesExtractors::Vector magnetic_field(0);
    const FEValuesExtractors::Scalar pseudo_pressure(dim);


    double rms_magnetic_field = 0;
    double rms_pseudo_pressure = 0;
    double volume = 0;

    for (auto cell: magnetic_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        fe_values[magnetic_field].get_function_values(magnetic_solution,
                                                      magnetic_field_values);
        fe_values[pseudo_pressure].get_function_values(magnetic_solution,
                                                       pseudo_pressure_values);
        for (unsigned int q=0; q<n_q_points; ++q)
        {
            rms_magnetic_field += magnetic_field_values[q] * magnetic_field_values[q] * fe_values.JxW(q);
            rms_pseudo_pressure += pseudo_pressure_values[q] * pseudo_pressure_values[q] * fe_values.JxW(q);
            volume += fe_values.JxW(q);
        }
    }

    rms_magnetic_field /= volume;
    AssertIsFinite(rms_magnetic_field);
    Assert(rms_magnetic_field >= 0,
           ExcLowerRangeType<double>(rms_magnetic_field, 0));

    rms_pseudo_pressure /= volume;
    AssertIsFinite(rms_pseudo_pressure);
    Assert(rms_pseudo_pressure >= 0,
           ExcLowerRangeType<double>(rms_pseudo_pressure, 0));

    return std::pair<double,double>(std::sqrt(rms_magnetic_field),
                                    std::sqrt(rms_pseudo_pressure));
}

template<int dim>
void ConductingFluidSolver<dim>::run()
{
    make_grid();

    setup_dofs();

    const EquationData::InitialField<dim>   initial_field;

    VectorTools::interpolate(mapping,
                             magnetic_dof_handler,
                             initial_field,
                             old_magnetic_solution);

    magnetic_constraints.distribute(old_magnetic_solution);

    magnetic_solution = old_magnetic_solution;

    output_results(true);

    /*
     *

    double time = 0;

    do
    {
        std::cout << "step: " << Utilities::int_to_string(timestep_number, 8) << ", "
                  << "time: " << time << ", "
                  << "time step: " << timestep
                  << std::endl;

        // evolve magnetic field
        magnetic_step();

        if (timestep_number % rms_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute rms values");

            const std::pair<double,double> rms_values = compute_rms_values();

            std::cout << "   magnetic field rms value: "
                      << rms_values.first
                      << std::endl
                      << "   pseudo pressure rms value: "
                      << rms_values.second
                      << std::endl;
        }

        if (timestep_number % vtk_frequency == 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }

        // copy magnetic solution
        old_old_magnetic_solution = old_magnetic_solution;
        old_magnetic_solution = magnetic_solution;

        // extrapolate magnetic solution
        magnetic_solution.sadd(1. + timestep / old_timestep,
                               timestep / old_timestep,
                               old_old_magnetic_solution);
        // advance in time
        time += timestep;
        ++timestep_number;

    } while (timestep_number < n_steps && time < t_final);

    if (n_steps % vtk_frequency != 0)
        output_results();
     *
     */

    std::cout << std::fixed;

    computing_timer.print_summary();
    computing_timer.reset();

    std::cout << std::endl;
}
}  // namespace ConductingFluid

// explicit instantiation
template class ConductingFluid::ConductingFluidSolver<3>;
