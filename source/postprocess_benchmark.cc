/*
 * postprocess_benchmark.cc
 *
 *  Created on: May 15, 2019
 *      Author: sg
 */

#include <deal.II/base/geometric_utilities.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/math/tools/roots.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <functional>
#include <algorithm>
#include <array>

#include <buoyant_fluid_solver.h>
#include <geometric_utilities.h>
#include <postprocessor.h>

DeclException0(ExcBoostNoConvergence);

namespace BuoyantFluid {

template<int dim>
double BuoyantFluidSolver<dim>::compute_radial_velocity_locally
(const double    &radius,
 const double    &phi,
 const double    &theta) const
{
    std::array<double,dim> scoord;
    scoord[0] = radius;

    switch (dim)
    {
        case 2:
            scoord[1] = phi;
            break;
        case 3:
            scoord[1] = theta;
            scoord[2] = phi;
            break;
        default:
            Assert(false, ExcImpossibleInDim(dim));
            break;
    }

    const Point<dim> x = GeometricUtilities::Coordinates::from_spherical(scoord);

    Tensor<1,dim>   velocity;
    try
    {
        Vector<double>  values(navier_stokes_fe.n_components());

        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 values);

        for (unsigned int d=dim; d<dim; ++d)
            velocity[d] = values[d];
    }
    catch (VectorTools::ExcPointNotAvailableHere    &exc)
    {
        return std::numeric_limits<double>::quiet_NaN();
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

    const std::array<Tensor<1,dim>, dim> sbasis
    = CoordinateTransformation::spherical_basis(scoord);

    const std::array<double, dim> spherical_projections
    = CoordinateTransformation::spherical_projections(velocity, sbasis);

    return spherical_projections[0];
}

template<int dim>
double BuoyantFluidSolver<dim>::compute_radial_velocity
(const double    &radius,
 const double    &phi,
 const double    &theta) const
{
    Assert(radius <= 1.0,
           ExcLowerRangeType<double>(1., radius));
    Assert(radius >= parameters.aspect_ratio,
           ExcLowerRangeType<double>(radius, parameters.aspect_ratio));

    const double local_radial_velocity
    = compute_radial_velocity_locally(radius,
                                      (phi > 2. * numbers::PI ? phi - 2. * numbers::PI: phi ),
                                      theta);

    std::vector<double> all_radial_velocities
    = Utilities::MPI::gather(mpi_communicator,
                             local_radial_velocity);

    std::map<unsigned int, double>  doubles_to_send;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        double  radial_velocity = std::numeric_limits<double>::quiet_NaN();

        unsigned int    nan_counter = 0;

        for (const auto v: all_radial_velocities)
            if (std::isnan(v))
                nan_counter += 1;
            else
                radial_velocity = v;

        const unsigned int n_mpi_processes
        = Utilities::MPI::n_mpi_processes(mpi_communicator);

        Assert(nan_counter == (n_mpi_processes - 1),
               ExcDimensionMismatch(nan_counter, n_mpi_processes - 1));
        AssertIsFinite(radial_velocity);

        for (unsigned int p=1; p<n_mpi_processes; ++p)
            doubles_to_send[p] = radial_velocity;
    }

    const std::map<unsigned int, double>   doubles_received
    = Utilities::MPI::some_to_some(mpi_communicator,
                                   doubles_to_send);

    double   radial_velocity;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        AssertDimension(doubles_received.size(), 1);

        Assert(doubles_received.begin()->first == 0,
               ExcInternalError());

        radial_velocity = doubles_received.begin()->second;
    }
    else
    {
        AssertDimension(doubles_received.size(), 0);

        radial_velocity = doubles_to_send[1];
    }

    return radial_velocity;
}

template<int dim>
double  BuoyantFluidSolver<dim>::compute_azimuthal_gradient_of_radial_velocity_locally
(const double    &radius,
 const double    &phi,
 const double    &theta) const
{
    std::array<double,dim> scoord;
    scoord[0] = radius;

    switch (dim)
    {
        case 2:
            scoord[1] = phi;
            break;
        case 3:
            scoord[1] = theta;
            scoord[2] = phi;
            break;
        default:
            Assert(false, ExcImpossibleInDim(dim));
            break;
    }

    const Point<dim> x = GeometricUtilities::Coordinates::from_spherical(scoord);

    Tensor<1,dim>   velocity;
    Tensor<2,dim>   velocity_gradient;
    try
    {
        Vector<double>              values(navier_stokes_fe.n_components());
        std::vector<Tensor<1,dim>>  gradients(navier_stokes_fe.n_components());

        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 values);

        VectorTools::point_gradient(mapping,
                                    navier_stokes_dof_handler,
                                    navier_stokes_solution,
                                    x,
                                    gradients);

        for (unsigned int d=0; d<dim; ++d)
        {
            velocity[d] = values[d];
            for (unsigned e=0; e<dim; ++e)
                velocity_gradient[d][e] = gradients[d][e];
        }

    }
    catch (VectorTools::ExcPointNotAvailableHere    &exc)
    {
        return std::numeric_limits<double>::quiet_NaN();
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

    const std::array<Tensor<1,dim>, dim> sbasis
    = CoordinateTransformation::spherical_basis(scoord);

    const std::array<double, dim> spherical_velocity
    = CoordinateTransformation::spherical_projections(velocity, sbasis);

    const double azimuthal_velocity = spherical_velocity.back();

    const std::array<std::array<double, dim>,dim>
    spherical_velocity_gradient
    = CoordinateTransformation::spherical_projections(velocity_gradient,
                                                      sbasis);

    const double projected_gradient = spherical_velocity_gradient[0].back();

    return radius * projected_gradient + azimuthal_velocity;
}



template<int dim>
double BuoyantFluidSolver<dim>::compute_azimuthal_gradient_of_radial_velocity
(const double    &radius,
 const double    &theta,
 const double    &phi) const
{
    const double local_azimuthal_velocity_gradient
    = compute_azimuthal_gradient_of_radial_velocity_locally(radius, theta, phi);

    std::vector<double> all_gradients
    = Utilities::MPI::gather(mpi_communicator,
                             local_azimuthal_velocity_gradient);

    std::map<unsigned int, double>   gradients_to_send;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        double  gradient = std::numeric_limits<double>::quiet_NaN();;

        unsigned int    nan_counter = 0;
        for (const auto v: all_gradients)
        {
            if (std::isnan(v))
                nan_counter += 1;
            else
                gradient = v;
        }

        const unsigned int n_mpi_processes
        = Utilities::MPI::n_mpi_processes(mpi_communicator);

        Assert(nan_counter == (n_mpi_processes - 1),
               ExcDimensionMismatch(nan_counter, n_mpi_processes - 1));

        for (unsigned int p=1; p<n_mpi_processes; ++p)
            gradients_to_send[p] = gradient;
    }

    const std::map<unsigned int, double>   gradients_received
    = Utilities::MPI::some_to_some(mpi_communicator,
                                   gradients_to_send);

    double   gradient;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        AssertDimension(gradients_received.size(), 1);

        Assert(gradients_received.begin()->first == 0,
               ExcInternalError());

        gradient = gradients_received.begin()->second;
    }
    else
    {
        AssertDimension(gradients_received.size(), 0);

        gradient = gradients_to_send[1];
    }

    return gradient;
}

template<int dim>
std::vector<double> BuoyantFluidSolver<dim>::compute_benchmark_requests_locally() const
{
    std::array<double,dim> scoord;
    scoord[0] = 0.5 * (1. + parameters.aspect_ratio);

    switch (dim)
    {
        case 2:
            scoord[1] = phi_benchmark;
            break;
        case 3:
            scoord[1] = numbers::PI / 2.;
            scoord[2] = phi_benchmark;
            break;
        default:
            Assert(false, ExcImpossibleInDim(dim));
            break;
    }

    const Point<dim> x = dealii::GeometricUtilities::Coordinates::from_spherical(scoord);

    Tensor<1,dim>   velocity;
    double          temperature_value(temperature_fe.n_components());
    Tensor<1,dim>   magnetic_field;
    try
    {

        Vector<double>  velocity_values(navier_stokes_fe.n_components());

        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 velocity_values);
        for (unsigned int d=0; d<dim; ++d)
            velocity[d] = velocity_values[d];

        temperature_value = VectorTools::point_value(mapping,
                                                     temperature_dof_handler,
                                                     temperature_solution,
                                                     x);

        if (parameters.magnetism && dim == 3)
        {
            Vector<double>  magnetic_values(magnetic_fe.n_components());
            VectorTools::point_value(mapping,
                                     magnetic_dof_handler,
                                     magnetic_solution,
                                     x,
                                     magnetic_values);
            for (unsigned int d=0; d<dim; ++d)
                magnetic_field[d] = magnetic_values[d];
        }

    }
    catch (VectorTools::ExcPointNotAvailableHere    &exc)
    {
        if (parameters.magnetism && dim == 3)
            return std::vector<double>(3, std::numeric_limits<double>::quiet_NaN());
        else
            return std::vector<double>(2, std::numeric_limits<double>::quiet_NaN());
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

    const std::array<Tensor<1,dim>, dim> sbasis
    = CoordinateTransformation::spherical_basis(scoord);

    const std::array<double, dim> spherical_velocity
    = CoordinateTransformation::spherical_projections(velocity, sbasis);

    const double azimuthal_velocity = spherical_velocity.back();

    if (parameters.magnetism && dim == 3)
    {
        const std::array<double, dim> spherical_magnetic_field
        = CoordinateTransformation::spherical_projections(velocity, sbasis);

        const double polar_magnetic_field = spherical_magnetic_field[1];

        return std::vector<double>{temperature_value,
                                   azimuthal_velocity,
                                   polar_magnetic_field};
    }
    else
        return std::vector<double>{temperature_value, azimuthal_velocity};
}

template<int dim>
std::vector<double>  BuoyantFluidSolver<dim>::compute_benchmark_requests() const
{
    const std::vector<double> local_benchmark_requests
    = compute_benchmark_requests_locally();

    std::vector<std::vector<double>> all_benchmark_requests
    = Utilities::MPI::gather(mpi_communicator,
                             local_benchmark_requests);

    std::map<unsigned int, std::vector<double>> benchmark_request_to_send;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<double> benchmark_requests;

        if (parameters.magnetism && dim == 3)
            benchmark_requests.resize(3, std::numeric_limits<double>::quiet_NaN());
        else
            benchmark_requests.resize(2, std::numeric_limits<double>::quiet_NaN());

        unsigned int    nan_counter = 0;
        for (const auto v: all_benchmark_requests)
        {
            if (std::all_of(v.begin(), v.end(), [](double d){return std::isnan(d);}))
                nan_counter += 1;
            else
                benchmark_requests = v;
        }

        const unsigned int n_mpi_processes
        = Utilities::MPI::n_mpi_processes(mpi_communicator);

        Assert(nan_counter == (n_mpi_processes - 1),
               ExcDimensionMismatch(nan_counter, n_mpi_processes - 1));

        for (const auto v: benchmark_requests)
            AssertIsFinite(v);

        for (unsigned int p=1; p<n_mpi_processes; ++p)
            benchmark_request_to_send[p] = benchmark_requests;
    }

    const std::map<unsigned int, std::vector<double>>
    benchmark_request_received
    = Utilities::MPI::some_to_some(mpi_communicator,
                                   benchmark_request_to_send);

    std::vector<double>   benchmark_requests;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        AssertDimension(benchmark_request_received.size(), 1);

        Assert(benchmark_request_received.begin()->first == 0,
               ExcInternalError());

        benchmark_requests = benchmark_request_received.begin()->second;

        for (const auto v: benchmark_requests)
            AssertIsFinite(v);
    }
    else
    {
        AssertDimension(benchmark_request_received.size(), 0);

        benchmark_requests = benchmark_request_to_send[1];
    }

    return benchmark_requests;
}

template<int dim>
double  BuoyantFluidSolver<dim>::compute_zero_of_radial_velocity
(const double        &phi_guess,
 const bool           local_slope,
 const double        &tol,
 const unsigned int  &max_iter) const
{
    using namespace boost::math::tools;

    Assert(tol > 0.0, ExcLowerRangeType<double>(tol, 0));
    Assert(max_iter > 0, ExcLowerRange(max_iter, 0));

    Assert((phi_guess < 0.?
           (phi_guess >= -numbers::PI && phi_guess <= numbers::PI):
           (phi_guess >= 0. && phi_guess <= 2. * numbers::PI)),
           ExcAzimuthalAngleRange(phi_guess));

    const double radius = 0.5 * (1. + parameters.aspect_ratio),
                 theta = numbers::PI / 2.;

    auto boost_max_iter = boost::numeric_cast<boost::uintmax_t>(max_iter);

    // declare lambda functions for boost root finding
    auto function = [this,radius,theta](const double &x)
    {
        return compute_radial_velocity(radius, x, theta);
    };
    auto tolerance_criterion = [tol,function](const double &a, const double &b)
    {
        return std::abs(function(a)) <= tol && std::abs(function(b)) <= tol;
    };

    double phi = -2. * numbers::PI;
    try
    {
        auto phi_interval
        = bracket_and_solve_root(
                function,
                phi_guess,
                1.05,
                local_slope,
                tolerance_criterion,
                boost_max_iter);
        phi = 0.5 * (phi_interval.first + phi_interval.second);
    }
    catch (std::exception &exc)
    {
        throw ExcBoostNoConvergence();
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

    if (phi < 0.)
        phi += 2. * numbers::PI;
    else if (phi > 2. * numbers::PI)
        phi -= 2. * numbers::PI;
    return phi;
}

template<int dim>
void BuoyantFluidSolver<dim>::update_benchmark_point()
{
    if (parameters.verbose)
        pcout << "Updating benchmark point..." << std::endl;

    const double radius = 0.5 * (1. + parameters.aspect_ratio);

    if (phi_benchmark > 0.)
    {
        const double gradient_at_trial_point
        = compute_azimuthal_gradient_of_radial_velocity(radius, phi_benchmark, 0);

        const double phi
        = compute_zero_of_radial_velocity(phi_benchmark,
                                          gradient_at_trial_point > 0.,
                                          1e-6);

        const double gradient_at_zero
        = compute_azimuthal_gradient_of_radial_velocity(radius, phi, 0);

        if(gradient_at_zero > 0
                && phi_benchmark >= 0.
                && phi_benchmark <= 2. * numbers::PI)
        {
            phi_benchmark = phi;
            return;
        }
    }

    /*
     * A valid initial guess has not been computed yet or was not good enough.
     * We try to find a benchmark point from a set of trial points.
     */
    // initialize trial points
    const unsigned int n_trial_points = 16;
    std::vector<double> trial_points;
    trial_points.push_back(1e-3 * 2. * numbers::PI / static_cast<double>(n_trial_points));
    for (unsigned int i=1; i<n_trial_points; ++i)
        trial_points.push_back(i * 2. * numbers::PI / static_cast<double>(n_trial_points));

    bool            point_found = false;
    unsigned int    cnt = 0;
    while(cnt < n_trial_points && point_found == false)
    {
        const double gradient_at_trial_point
        = compute_azimuthal_gradient_of_radial_velocity(radius, trial_points[cnt], 0);

        try
        {
            const double phi = compute_zero_of_radial_velocity(trial_points[cnt],
                                                               gradient_at_trial_point > 0.,
                                                               1e-6);

            const double gradient_at_zero
            = compute_azimuthal_gradient_of_radial_velocity(radius, trial_points[cnt], 0);

            if (gradient_at_zero > 0.)
            {
                point_found = true;
                phi_benchmark = phi;
            }
            ++cnt;
        }
        catch(ExcBoostNoConvergence &exc)
        {
            ++cnt;
            continue;
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
    if (!point_found)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on update_benchmark_point!" << std::endl
                  << "The algorithm did not find a benchmark point using "
                  << n_trial_points << " trial points in [0,2*pi)."
                  << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
    }
    if (phi_benchmark < 0. || phi_benchmark > 2. * numbers::PI)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on update_benchmark_point!" << std::endl
                  << "The algorithm did not find a benchmark point in [0,2*pi)."
                  << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
    }
}

}  // namespace BuoyantFluid

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_radial_velocity_locally
(const double &,
 const double &,
 const double &) const;
template double BuoyantFluid::BuoyantFluidSolver<3>::
compute_radial_velocity_locally
(const double &,
 const double &,
 const double &) const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_radial_velocity
(const double &,
 const double &,
 const double &) const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_radial_velocity
(const double &,
 const double &,
 const double &) const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_azimuthal_gradient_of_radial_velocity_locally
(const double &,
 const double &,
 const double &) const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_azimuthal_gradient_of_radial_velocity_locally
(const double &,
 const double &,
 const double &) const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_azimuthal_gradient_of_radial_velocity
(const double &,
 const double &,
 const double &) const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_azimuthal_gradient_of_radial_velocity
(const double &,
 const double &,
 const double &) const;

template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_benchmark_requests_locally() const;
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_benchmark_requests_locally() const;

template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_benchmark_requests() const;
template std::vector<double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_benchmark_requests() const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_zero_of_radial_velocity
(const double        &,
 const bool           ,
 const double        &,
 const unsigned int  &) const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_zero_of_radial_velocity
(const double        &,
 const bool           ,
 const double        &,
 const unsigned int  &) const;


template void
BuoyantFluid::BuoyantFluidSolver<2>::update_benchmark_point();
template void
BuoyantFluid::BuoyantFluidSolver<3>::update_benchmark_point();
