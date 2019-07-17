/*
 * postprocess_solution.cc
 *
 *  Created on: Jan 10, 2019
 *      Author: sg
 */
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// Sadly this include file is not found
// #include <boost/filesystem.hpp>

#include <buoyant_fluid_solver.h>
#include <postprocessor.h>
#include <geometric_utilities.h>

namespace BuoyantFluid {

template<int dim>
std::vector<double> BuoyantFluidSolver<dim>::compute_global_averages() const
{
    const QGauss<dim> velocity_quadrature(parameters.velocity_degree + 1);

    const QGauss<dim> temperature_quadrature(parameters.temperature_degree + 1);

    const QGauss<dim> magnetic_quadrature(parameters.magnetic_degree + 1);

    const unsigned int n_velocity_q_points = velocity_quadrature.size();
    const unsigned int n_temperature_q_points = temperature_quadrature.size();
    const unsigned int n_magnetic_q_points = magnetic_quadrature.size();

    FEValues<dim> stokes_fe_values(mapping,
                                   navier_stokes_fe,
                                   velocity_quadrature,
                                   update_values|
                                   update_JxW_values);

    FEValues<dim> temperature_fe_values(mapping,
                                        temperature_fe,
                                        temperature_quadrature,
                                        update_values|
                                        update_JxW_values);

    FEValues<dim> magnetic_fe_values(mapping,
                                     magnetic_fe,
                                     magnetic_quadrature,
                                     update_values|
                                     update_JxW_values);

    std::vector<double>         temperature_values(n_temperature_q_points);
    std::vector<Tensor<1,dim>>  velocity_values(n_velocity_q_points);
    std::vector<Tensor<1,dim>>  magnetic_values(n_magnetic_q_points);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector magnetic_field(0);

    double local_sum_velocity_sqrd = 0;
    double local_sum_temperature = 0;
    double local_sum_magnetic_field_sqrd = 0;
    double local_navier_stokes_volume = 0;
    double local_temperature_volume = 0;
    double local_magnetic_volume = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = navier_stokes_dof_handler.begin_active(),
    temperature_cell = temperature_dof_handler.begin_active(),
    magnetic_cell = magnetic_dof_handler.begin_active(),
    endc = navier_stokes_dof_handler.end();

    if (parameters.magnetism)
    {
        for (; cell != endc; ++cell, ++temperature_cell, ++magnetic_cell)
        if (cell->is_locally_owned())
        {
            stokes_fe_values.reinit(cell);
            temperature_fe_values.reinit(temperature_cell);
            magnetic_fe_values.reinit(magnetic_cell);

            temperature_fe_values.get_function_values(temperature_solution,
                                                      temperature_values);

            stokes_fe_values[velocities].get_function_values(navier_stokes_solution,
                                                             velocity_values);

            magnetic_fe_values[magnetic_field].get_function_values(magnetic_solution,
                                                                   magnetic_values);

            for (unsigned int q=0; q<n_velocity_q_points; ++q)
            {
                local_sum_velocity_sqrd += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
                local_navier_stokes_volume += stokes_fe_values.JxW(q);
            }
            for (unsigned int q=0; q<n_temperature_q_points; ++q)
            {
                local_sum_temperature += temperature_values[q] * temperature_fe_values.JxW(q);
                local_temperature_volume += temperature_fe_values.JxW(q);
            }
            for (unsigned int q=0; q<n_magnetic_q_points; ++q)
            {
                local_sum_magnetic_field_sqrd += magnetic_values[q] * magnetic_values[q] * magnetic_fe_values.JxW(q);
                local_magnetic_volume += magnetic_fe_values.JxW(q);
            }
        }

        AssertIsFinite(local_sum_velocity_sqrd);
        AssertIsFinite(local_sum_temperature);
        AssertIsFinite(local_sum_magnetic_field_sqrd);
        AssertIsFinite(local_navier_stokes_volume);
        AssertIsFinite(local_temperature_volume);
        AssertIsFinite(local_magnetic_volume);

        Assert(local_sum_velocity_sqrd >= 0, ExcLowerRangeType<double>(local_sum_velocity_sqrd, 0));
        if (parameters.geometry == GeometryType::SphericalShell)
           Assert(local_sum_temperature >= 0, ExcLowerRangeType<double>(local_sum_temperature, 0));
        Assert(local_sum_magnetic_field_sqrd >= 0, ExcLowerRangeType<double>(local_sum_magnetic_field_sqrd, 0));
        Assert(local_navier_stokes_volume >= 0, ExcLowerRangeType<double>(local_navier_stokes_volume, 0));
        Assert(local_temperature_volume >= 0, ExcLowerRangeType<double>(local_temperature_volume, 0));
        Assert(local_magnetic_volume >= 0, ExcLowerRangeType<double>(local_magnetic_volume, 0));

        const std::vector<double> local_sums({local_sum_velocity_sqrd,
                                              local_navier_stokes_volume,
                                              local_sum_temperature,
                                              local_temperature_volume,
                                              local_sum_magnetic_field_sqrd,
                                              local_magnetic_volume});

        std::vector<double> global_sums(local_sums.size());

        Utilities::MPI::sum<std::vector<double>>(local_sums,
                                                 mpi_communicator,
                                                 global_sums);

        const double rms_velocity = std::sqrt(global_sums[0] / global_sums[1]);
        const double kinetic_energy = 0.5 * global_sums[0] / global_sums[1];
        const double avg_temperature = global_sums[2] / global_sums[3];
        const double rms_magnetic_field = std::sqrt(global_sums[4] / global_sums[5]);
        const double magnetic_energy = 0.5 * global_sums[4] / global_sums[5];

        return std::vector<double>({rms_velocity, kinetic_energy,
                                    avg_temperature,
                                    rms_magnetic_field, magnetic_energy});
    }
    else
    {
        for (; cell != endc; ++cell, ++temperature_cell)
        if (cell->is_locally_owned())
        {
            stokes_fe_values.reinit(cell);
            temperature_fe_values.reinit(temperature_cell);

            temperature_fe_values.get_function_values(temperature_solution,
                                                      temperature_values);

            stokes_fe_values[velocities].get_function_values(navier_stokes_solution,
                                                             velocity_values);

            for (unsigned int q=0; q<n_velocity_q_points; ++q)
            {
                local_sum_velocity_sqrd += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
                local_navier_stokes_volume += stokes_fe_values.JxW(q);
            }
            for (unsigned int q=0; q<n_temperature_q_points; ++q)
            {
                local_sum_temperature += temperature_values[q] * temperature_fe_values.JxW(q);
                local_temperature_volume += temperature_fe_values.JxW(q);
            }
        }

        AssertIsFinite(local_sum_velocity_sqrd);
        AssertIsFinite(local_sum_temperature);
        AssertIsFinite(local_navier_stokes_volume);
        AssertIsFinite(local_temperature_volume);

        Assert(local_sum_velocity_sqrd >= 0, ExcLowerRangeType<double>(local_sum_velocity_sqrd, 0));
        if (parameters.geometry == GeometryType::SphericalShell)
           Assert(local_sum_temperature >= 0, ExcLowerRangeType<double>(local_sum_temperature, 0));
        Assert(local_navier_stokes_volume >= 0, ExcLowerRangeType<double>(local_navier_stokes_volume, 0));
        Assert(local_temperature_volume >= 0, ExcLowerRangeType<double>(local_temperature_volume, 0));

        const std::vector<double> local_sums({local_sum_velocity_sqrd,
                                              local_navier_stokes_volume,
                                              local_sum_temperature,
                                              local_temperature_volume});

        std::vector<double> global_sums(local_sums.size());

        Utilities::MPI::sum<std::vector<double>>(local_sums,
                                                 mpi_communicator,
                                                 global_sums);

        const double rms_velocity = std::sqrt(global_sums[0] / global_sums[1]);
        const double kinetic_energy = 0.5 * global_sums[0] / global_sums[1];
        const double avg_temperature = global_sums[2] / global_sums[3];

        return std::vector<double>({rms_velocity, kinetic_energy,
                                    avg_temperature});
    }
}

template<int dim>
std::vector<double> BuoyantFluidSolver<dim>::compute_point_value_locally
(const Point<dim> &point) const
{
    std::vector<double> point_values((parameters.magnetism? 2*dim+3: dim+2),
                                     std::numeric_limits<double>::quiet_NaN());

    try
    {
        // velocity and pressure
        Vector<double>  navier_stokes_values(navier_stokes_fe.n_components());
        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 point,
                                 navier_stokes_values);

        for (unsigned int i=0; i<navier_stokes_fe.n_components(); ++i)
            point_values[i] = navier_stokes_values[i];

        // temperature
        point_values[dim+1] = VectorTools::point_value(mapping,
                                                        temperature_dof_handler,
                                                        temperature_solution,
                                                        point);
        // magnetic field and magnetic pressure
        if (parameters.magnetism)
        {
            Vector<double>  magnetic_values(magnetic_fe.n_components());

            VectorTools::point_value(mapping,
                                     magnetic_dof_handler,
                                     navier_stokes_solution,
                                     point,
                                     navier_stokes_values);
            for (unsigned int i=0; i<magnetic_fe.n_components(); ++i)
                point_values[dim+2+i] = magnetic_values[i];
        }

        return point_values;
    }
    catch (VectorTools::ExcPointNotAvailableHere    &exc)
    {
        return point_values;
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
        std::abort();
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
        std::abort();
    }
}

template<int dim>
std::vector<double> BuoyantFluidSolver<dim>::compute_point_value
(const Point<dim> &point) const
{
    const std::vector<double> local_point_values
    = compute_point_value_locally(point);

    std::vector<std::vector<double>> all_local_point_values
    = Utilities::MPI::gather(mpi_communicator,
                             local_point_values);

    std::map<unsigned int, std::vector<double>> point_values_to_send;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<double> point_values;

        point_values.resize(parameters.magnetism? 2*dim+3: dim+2,
                            std::numeric_limits<double>::quiet_NaN());

        unsigned int    nan_counter = 0;
        for (const auto v: all_local_point_values)
        {
            if (std::all_of(v.begin(), v.end(), [](double d){return std::isnan(d);}))
                nan_counter += 1;
            else
                point_values = v;
        }

        const unsigned int n_mpi_processes
        = Utilities::MPI::n_mpi_processes(mpi_communicator);

        Assert(nan_counter == (n_mpi_processes - 1),
               ExcDimensionMismatch(nan_counter, n_mpi_processes - 1));

        for (const auto v: point_values)
            AssertIsFinite(v);

        for (unsigned int p=1; p<n_mpi_processes; ++p)
            point_values_to_send[p] = point_values;
    }

    const std::map<unsigned int, std::vector<double>>
    point_values_received
    = Utilities::MPI::some_to_some(mpi_communicator,
                                   point_values_to_send);

    std::vector<double>   point_values;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        AssertDimension(point_values_received.size(), 1);

        Assert(point_values_received.begin()->first == 0,
               ExcInternalError());

        point_values = point_values_received.begin()->second;

        for (const auto v: point_values)
            AssertIsFinite(v);
    }
    else
    {
        AssertDimension(point_values_received.size(), 0);

        point_values = point_values_to_send[1];
    }

    if (parameters.point_probe_spherical)
    {
        const std::array<double,dim> scoord
        = GeometricUtilities::Coordinates::to_spherical(point);

        const std::array<Tensor<1,dim>,dim> sbasis
        = CoordinateTransformation::spherical_basis(scoord);

        Tensor<1,dim>   velocity_field;
        for (unsigned int d=0; d<dim; ++d)
            velocity_field[d] = point_values[d];

        const std::array<double,dim> spherical_velocity
        = CoordinateTransformation::spherical_projections(velocity_field,
                                                          sbasis);
        for (unsigned int d=0; d<dim; ++d)
            point_values.push_back(spherical_velocity[d]);

        if (parameters.magnetism)
        {
            Tensor<1,dim>   magnetic_field;
            for (unsigned int d=0; d<dim; ++d)
                magnetic_field[d] = point_values[dim+2+d];

            const std::array<double,dim> spherical_magnetic_field
            = CoordinateTransformation::spherical_projections(magnetic_field,
                                                              sbasis);
            for (unsigned int d=0; d<dim; ++d)
                point_values.push_back(spherical_magnetic_field[d]);
        }
    }

    return point_values;
}

template <int dim>
double BuoyantFluidSolver<dim>::compute_cfl_number() const
{
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping,
                            navier_stokes_fe,
                            quadrature_formula,
                            update_values);

    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    const double largest_viscosity = std::max(equation_coefficients[1],
                                              equation_coefficients[3]);

    double max_cfl = 0;
    for (auto cell: navier_stokes_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit (cell);
            fe_values[velocities].get_function_values(navier_stokes_solution,
                                                      velocity_values);

            double  max_cell_velocity = 0;
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                max_cell_velocity = std::max(max_cell_velocity,
                                             velocity_values[q].norm());
            }
            max_cfl = std::max(max_cfl,
                               max_cell_velocity / (cell->diameter() * std::sqrt(dim)));
            max_cfl = std::max(max_cfl,
                               largest_viscosity / (cell->diameter() * cell->diameter() * dim));
        }

    const double max_polynomial_degree
    = double(std::max(parameters.temperature_degree,
                      parameters.velocity_degree));

    const double local_cfl = max_cfl * timestep * max_polynomial_degree;

    const double global_cfl
    = Utilities::MPI::max(local_cfl, mpi_communicator);

    return global_cfl;
}

template<int dim>
void BuoyantFluidSolver<dim>::output_results(const bool initial_condition) const
{
    if (parameters.verbose)
        pcout << "Output results..." << std::endl;

    // create joint dof handler
    DoFHandler<dim>     joint_dof_handler(triangulation);
    // create joint solution vector
    LA::Vector          joint_solution;

    if (parameters.magnetism)
    {
        // create joint finite element
        const FESystem<dim> joint_fe(navier_stokes_fe, 1,
                                     temperature_fe, 1,
                                     magnetic_fe, 1);

        // distribute dofs
        joint_dof_handler.distribute_dofs(joint_fe);

        Assert(joint_dof_handler.n_dofs() ==
               navier_stokes_dof_handler.n_dofs() +
               temperature_dof_handler.n_dofs() +
               magnetic_dof_handler.n_dofs(),
               ExcInternalError());

        // create joint solution
        LA::Vector distributed_joint_solution;
        distributed_joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                                          mpi_communicator);
        {
            std::vector<types::global_dof_index> local_joint_dof_indices(joint_fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_stokes_dof_indices(navier_stokes_fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_temperature_dof_indices(temperature_fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_magnetic_dof_indices(magnetic_fe.dofs_per_cell);

            typename DoFHandler<dim>::active_cell_iterator
            joint_cell       = joint_dof_handler.begin_active(),
            joint_endc       = joint_dof_handler.end(),
            stokes_cell      = navier_stokes_dof_handler.begin_active(),
            temperature_cell = temperature_dof_handler.begin_active(),
            magnetic_cell = magnetic_dof_handler.begin_active();
            for (; joint_cell!=joint_endc; ++joint_cell, ++stokes_cell, ++temperature_cell, ++magnetic_cell)
            if (joint_cell->is_locally_owned())
            {
                joint_cell->get_dof_indices(local_joint_dof_indices);
                stokes_cell->get_dof_indices(local_stokes_dof_indices);
                temperature_cell->get_dof_indices(local_temperature_dof_indices);
                magnetic_cell->get_dof_indices(local_magnetic_dof_indices);

                for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                    if (joint_fe.system_to_base_index(i).first.first == 0)
                    {
                        Assert (joint_fe.system_to_base_index(i).second < local_stokes_dof_indices.size(),
                                ExcInternalError());
                        distributed_joint_solution(local_joint_dof_indices[i])
                        = navier_stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                    }
                    else if (joint_fe.system_to_base_index(i).first.first == 1)
                    {
                        Assert (joint_fe.system_to_base_index(i).second < local_temperature_dof_indices.size(),
                                ExcInternalError());
                        distributed_joint_solution(local_joint_dof_indices[i])
                        = temperature_solution(local_temperature_dof_indices[joint_fe.system_to_base_index(i).second]);
                    }
                    else if (joint_fe.system_to_base_index(i).first.first == 2)
                    {
                        Assert (joint_fe.system_to_base_index(i).second < local_magnetic_dof_indices.size(),
                                ExcInternalError());
                        distributed_joint_solution(local_joint_dof_indices[i])
                        = magnetic_solution(local_magnetic_dof_indices[joint_fe.system_to_base_index(i).second]);
                    }
            }
        }
        distributed_joint_solution.compress(VectorOperation::insert);

        // initialize locally relevant index set
        IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
        DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                                locally_relevant_joint_dofs);
        // initialize joint solution vector
        joint_solution.reinit(locally_relevant_joint_dofs,
                              mpi_communicator);
        // assign joint solution vector
        joint_solution = distributed_joint_solution;
    }
    else
    {
        // create joint finite element
        const FESystem<dim> joint_fe(navier_stokes_fe, 1,
                                     temperature_fe, 1);

        // distribute dofs
        joint_dof_handler.distribute_dofs(joint_fe);

        Assert(joint_dof_handler.n_dofs() ==
               navier_stokes_dof_handler.n_dofs() +
               temperature_dof_handler.n_dofs(),
               ExcInternalError());

        // create joint solution
        LA::Vector  distributed_joint_solution;
        distributed_joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                                          mpi_communicator);
        {
            std::vector<types::global_dof_index> local_joint_dof_indices(joint_fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_stokes_dof_indices(navier_stokes_fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_temperature_dof_indices(temperature_fe.dofs_per_cell);

            typename DoFHandler<dim>::active_cell_iterator
            joint_cell       = joint_dof_handler.begin_active(),
            joint_endc       = joint_dof_handler.end(),
            stokes_cell      = navier_stokes_dof_handler.begin_active(),
            temperature_cell = temperature_dof_handler.begin_active();
            for (; joint_cell!=joint_endc; ++joint_cell, ++stokes_cell, ++temperature_cell)
            if (joint_cell->is_locally_owned())
            {
                joint_cell->get_dof_indices(local_joint_dof_indices);
                stokes_cell->get_dof_indices(local_stokes_dof_indices);
                temperature_cell->get_dof_indices(local_temperature_dof_indices);

                for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                    if (joint_fe.system_to_base_index(i).first.first == 0)
                    {
                        Assert (joint_fe.system_to_base_index(i).second < local_stokes_dof_indices.size(),
                                ExcInternalError());
                        distributed_joint_solution(local_joint_dof_indices[i])
                        = navier_stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                    }
                    else if (joint_fe.system_to_base_index(i).first.first == 1)
                    {
                        Assert (joint_fe.system_to_base_index(i).first.first == 1,
                                ExcInternalError());
                        Assert (joint_fe.system_to_base_index(i).second < local_temperature_dof_indices.size(),
                                ExcInternalError());
                        distributed_joint_solution(local_joint_dof_indices[i])
                        = temperature_solution(local_temperature_dof_indices[joint_fe.system_to_base_index(i).second]);
                    }
            }
        }
        distributed_joint_solution.compress(VectorOperation::insert);

        // initialize locally relevant index set
        IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
        DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                                locally_relevant_joint_dofs);
        // initialize joint solution vector
        joint_solution.reinit(locally_relevant_joint_dofs,
                              mpi_communicator);
        // assign joint solution vector
        joint_solution = distributed_joint_solution;
    }


    // create post processor
    PostProcessor<dim>
    postprocessor(Utilities::MPI::this_mpi_process(mpi_communicator),
                  parameters.magnetism,
                  parameters.output_flags);

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    data_out.add_data_vector(joint_solution, postprocessor);
    data_out.build_patches();

    /*
     *
    // create results directory
    {
        // verify that the output directory actually exists. if it doesn't, create
        // it on processor zero
        bool success;

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            using namespace boost::filesystem;

            const path  results_path("results");

            if (!is_directory(results_path))
            {
                pcout << "----------------------------------------------------"
                      << std::endl
                      << "The output directory appears not to exist\n"
                      << "and will be created for you.\n"
                      << "----------------------------------------------------"
                      << std::endl;

                success = create_directory(results_path);

            }
            else
            {
                success = true;
            }
            // Broadcast error code
            MPI_Bcast(&success, 1, MPIU_BOOL, 0, mpi_communicator);
            AssertThrow(success == false,
                        ExcMessage(std::string("Can't create the output directory.")));
        }
        else
        {
            // Wait to receive error code, and throw ExcInternalError if directory
            // creation has failed
            MPI_Bcast (&success, 1, MPIU_BOOL, 0, mpi_communicator);
            if (success == false)
                throw ExcInternalError();
        }
    }
     *
     */

    // write output to disk
    const std::string filename = ("solution-" +
                                  (initial_condition ? "initial":
                                   Utilities::int_to_string (timestep_number, 5)) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
            filenames.push_back(std::string("solution-") +
                                (initial_condition ? "initial":
                                Utilities::int_to_string (timestep_number, 5)) +
                                "." +
                                Utilities::int_to_string(i, 4) +
                                ".vtu");
        const std::string
        pvtu_master_filename = ("solution-" +
                                (initial_condition ? "initial":
                                Utilities::int_to_string (timestep_number, 5)) +
                                ".pvtu");
        std::ofstream pvtu_master(pvtu_master_filename.c_str());
        data_out.write_pvtu_record(pvtu_master, filenames);
        const std::string
        visit_master_filename = ("solution-" +
                                 (initial_condition ? "initial":
                                 Utilities::int_to_string (timestep_number, 5)) +
                                 ".visit");
        std::ofstream visit_master(visit_master_filename.c_str());
        DataOutBase::write_visit_record(visit_master, filenames);
    }
}
}  // namespace BuoyantFluid

// explicit instantiation
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_point_value_locally(const Point<2> &) const;
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_point_value_locally(const Point<3> &) const;

template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_point_value(const Point<2> &) const;
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_point_value(const Point<3> &) const;

template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_global_averages() const;
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_global_averages() const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_cfl_number() const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_cfl_number() const;

template void
BuoyantFluid::BuoyantFluidSolver<2>::output_results
(const bool) const;
template void
BuoyantFluid::BuoyantFluidSolver<3>::output_results
(const bool) const;
