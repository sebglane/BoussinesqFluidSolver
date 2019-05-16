/*
 * postprocess_benchmark.cc
 *
 *  Created on: May 15, 2019
 *      Author: sg
 */

#include <deal.II/numerics/vector_tools.h>

#include <boost/math/tools/roots.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <functional>
#include <algorithm>

#include "buoyant_fluid_solver.h"
#include "postprocessor.h"

DeclException1(ExcPostiveRadius,
               double,
               << "The radius, r = " << arg1
               << ", is not positive.");

DeclException1(ExcPolarAngleRange,
               double,
               << "The polar angle theta = " << arg1
               << ", is not in the half-open range [0,pi).");

DeclException1(ExcAzimuthalAngleRange,
               double,
               << "The azimuthal angle, phi =  " << arg1
               << ", is not in the half-open range [0,2*pi).");

DeclException1(ExcNoBenchmarkPointFound,
               int,
               << "The algorithm did not found a benchmark "
               << "using " << arg1
               << " trial points in [0,2*pi).");

DeclException0(ExcBoostNoConvergence);

namespace BuoyantFluid {
template<>
double BuoyantFluidSolver<2>::compute_radial_velocity_locally(
        const double    &radius,
        const double    &phi,
        const double    &/* phi */) const
{
    const unsigned dim = 2;

    Assert(radius > 0.0, ExcPostiveRadius(radius));

    if (phi < 0.)
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    const Point<dim>    x(radius * cos(phi), radius * sin(phi));

    Vector<double>  values(navier_stokes_fe.n_components());

    try
    {
        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 values);
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

    const double radial_component = values[0] * cos(phi) + values[1] * sin(phi);

    return radial_component;
}

template<>
double BuoyantFluidSolver<3>::compute_radial_velocity_locally
(const double    &radius,
 const double    &theta,
 const double    &phi) const
{
    const unsigned dim = 3;

    Assert(radius > 0.0, ExcPostiveRadius(radius));

    Assert(theta >= 0. && theta <= numbers::PI,
           ExcPolarAngleRange(theta));

    if (phi < 0.)
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    const Point<dim>    x(radius * cos(phi) * sin(theta),
                          radius * sin(phi) * sin(theta),
                          radius * cos(theta));

    Vector<double>  values(navier_stokes_fe.n_components());

    try
    {
        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 values);
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

    const double radial_component
    = values[0] * cos(phi) * sin(theta)
    + values[1] * sin(phi) * sin(theta)
    + values[2] * cos(theta);

    return radial_component;
}

template<int dim>
double BuoyantFluidSolver<dim>::compute_radial_velocity
(const double    &radius,
 const double    &theta,
 const double    &phi) const
{
    const double local_radial_velocity
    = compute_radial_velocity_locally(radius, theta, phi);

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

template<>
double  BuoyantFluidSolver<2>::compute_azimuthal_gradient_of_radial_velocity_locally
(const double    &radius,
 const double    &phi,
 const double    &/* phi */) const
{
    const unsigned dim = 2;

    Assert(radius > 0.0, ExcPostiveRadius(radius));

    if (phi < 0.)
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    const Point<dim>    x(radius * cos(phi), radius * sin(phi));

    Vector<double>              values(navier_stokes_fe.n_components());
    std::vector<Tensor<1,dim>>  gradients(navier_stokes_fe.n_components());
    try
    {

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

    Tensor<1,dim>   velocity;
    Tensor<2,dim>   velocity_gradient;
    for (unsigned int d=0; d<dim; ++d)
    {
        velocity[d] = values[d];
        for (unsigned e=0; e<dim; ++e)
            velocity_gradient[d][e] = gradients[d][e];
    }

    const Tensor<1,dim>   radial_basis_vector({cos(phi), sin(phi)});
    const Tensor<1,dim>   azimuthal_basis_vector({-sin(phi), cos(phi)});

    const double azimuthal_velocity = azimuthal_basis_vector * velocity;

    const double projected_gradient
    = ( radial_basis_vector * (velocity_gradient * azimuthal_basis_vector) );

    return radius * projected_gradient + azimuthal_velocity;
}


template<>
double BuoyantFluidSolver<3>::compute_azimuthal_gradient_of_radial_velocity_locally
(const double    &radius,
 const double    &theta,
 const double    &phi) const
{
    const unsigned dim = 3;

    Assert(radius > 0.0, ExcPostiveRadius(radius));

    Assert(theta >= 0. && theta <= numbers::PI,
           ExcPolarAngleRange(theta));

    if (phi < 0.)
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    const Point<dim>    x(radius * cos(phi) * sin(theta),
                          radius * sin(phi) * sin(theta),
                          radius * cos(theta));

    Vector<double>              values(navier_stokes_fe.n_components());
    std::vector<Tensor<1,dim>>  gradients(navier_stokes_fe.n_components());
    try
    {

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

    Tensor<1,dim>   velocity;
    Tensor<2,dim>   velocity_gradient;
    for (unsigned int d=0; d<dim; ++d)
    {
        velocity[d] = values[d];
        for (unsigned e=0; e<dim; ++e)
            velocity_gradient[d][e] = gradients[d][e];
    }

    const Tensor<1,dim>   radial_basis_vector({cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)});
    const Tensor<1,dim>   azimuthal_basis_vector({-sin(phi), cos(phi) , 0.});

    const double azimuthal_velocity = azimuthal_basis_vector * velocity;

    const double projected_gradient
    = ( radial_basis_vector * (velocity_gradient * azimuthal_basis_vector) );

    return radius * sin(theta) * projected_gradient + sin(phi) * azimuthal_velocity;
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

template<>
std::pair<double,double> BuoyantFluidSolver<2>::compute_benchmark_requests_locally
(const double &radius,
 const double &phi,
 const double &/* phi */) const
{
    const unsigned int dim = 2;

    Assert(radius > 0.0, ExcPostiveRadius(radius));

    if (phi < 0.)
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    const Point<dim>    x(radius * cos(phi),
                          radius * sin(phi));

    Vector<double>  velocity_values(navier_stokes_fe.n_components());
    double          temperature_value(temperature_fe.n_components());
    try
    {

        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 velocity_values);

        temperature_value = VectorTools::point_value(mapping,
                                                     temperature_dof_handler,
                                                     temperature_solution,
                                                     x);
    }
    catch (VectorTools::ExcPointNotAvailableHere    &exc)
    {
        return std::pair<double,double>(std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN());
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

    Tensor<1,dim>   velocity;
    for (unsigned int d=0; d<dim; ++d)
        velocity[d] = velocity_values[d];

    const Tensor<1,dim>   azimuthal_basis_vector({-sin(phi), cos(phi)});

    const double azimuthal_velocity = azimuthal_basis_vector * velocity;

    return std::pair<double,double>(temperature_value, azimuthal_velocity);
}


template<>
std::pair<double,double> BuoyantFluidSolver<3>::compute_benchmark_requests_locally
(const double &radius,
 const double &theta,
 const double &phi) const
{
    const unsigned int dim = 3;

    Assert(radius > 0.0, ExcPostiveRadius(radius));

    Assert(theta >= 0. && theta <= numbers::PI,
           ExcPolarAngleRange(theta));

    if (phi < 0.)
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    const Point<dim>    x(radius * cos(phi) * sin(theta),
                          radius * sin(phi) * sin(theta),
                          radius * cos(theta));

    Vector<double>  velocity_values(navier_stokes_fe.n_components());
    double          temperature_value(temperature_fe.n_components());
    try
    {

        VectorTools::point_value(mapping,
                                 navier_stokes_dof_handler,
                                 navier_stokes_solution,
                                 x,
                                 velocity_values);

        temperature_value = VectorTools::point_value(mapping,
                                                     temperature_dof_handler,
                                                     temperature_solution,
                                                     x);
    }
    catch (VectorTools::ExcPointNotAvailableHere    &exc)
    {
        return std::pair<double,double>(std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN());
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

    Tensor<1,dim>   velocity;
    for (unsigned int d=0; d<dim; ++d)
        velocity[d] = velocity_values[d];

    const Tensor<1,dim>   azimuthal_basis_vector({-sin(phi), cos(phi) , 0.});

    const double azimuthal_velocity = azimuthal_basis_vector * velocity;

    return std::pair<double,double>(temperature_value, azimuthal_velocity);
}

template<int dim>
std::pair<double,double>  BuoyantFluidSolver<dim>::compute_benchmark_requests
(const double    &radius,
 const double    &theta,
 const double    &phi) const
{
    const std::pair<double,double> local_benchmark_requests
    = compute_benchmark_requests_locally(radius, theta, phi);

    std::vector<std::pair<double,double>> all_benchmark_requests
    = Utilities::MPI::gather(mpi_communicator,
                             local_benchmark_requests);

    std::map<unsigned int, std::pair<double,double>> benchmark_request_to_send;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::pair<double,double>
        benchmark_requests(std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN());

        unsigned int    nan_counter = 0;
        for (const auto v: all_benchmark_requests)
        {
            if (std::isnan(v.first) || std::isnan(v.second))
                nan_counter += 1;
            else
                benchmark_requests = v;
        }

        const unsigned int n_mpi_processes
        = Utilities::MPI::n_mpi_processes(mpi_communicator);

        Assert(nan_counter == (n_mpi_processes - 1),
               ExcDimensionMismatch(nan_counter, n_mpi_processes - 1));

        AssertIsFinite(benchmark_requests.first);
        AssertIsFinite(benchmark_requests.second);

        for (unsigned int p=1; p<n_mpi_processes; ++p)
            benchmark_request_to_send[p] = benchmark_requests;
    }

    const std::map<unsigned int, std::pair<double,double>>
    benchmark_request_received
    = Utilities::MPI::some_to_some(mpi_communicator,
                                   benchmark_request_to_send);

    std::pair<double,double>   benchmark_requests;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        AssertDimension(benchmark_request_received.size(), 1);

        Assert(benchmark_request_received.begin()->first == 0,
               ExcInternalError());

        benchmark_requests = benchmark_request_received.begin()->second;

        AssertIsFinite(benchmark_requests.first);
        AssertIsFinite(benchmark_requests.second);
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
 const double        &tol,
 const unsigned int  &max_iter) const
{
    using namespace boost::math::tools;

    Assert(tol > 0.0, ExcLowerRangeType<double>(tol, 0));
    Assert(max_iter > 0, ExcLowerRange(max_iter, 0));

    if (phi_guess < 0.)
    {
        Assert((phi_guess >= -numbers::PI) && (phi_guess <= numbers::PI),
               ExcAzimuthalAngleRange(phi_guess));
    }
    else
    {
        Assert((phi_guess >= 0.) && (phi_guess <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi_guess));
    }

    const double radius = 0.5 * (1. + parameters.aspect_ratio);
    const double theta = numbers::PI / 2.;

    auto boost_max_iter = boost::numeric_cast<boost::uintmax_t>(max_iter);

    // declare lambda functions for boost root finding
    auto function = [this,radius,theta](const double &x){ return compute_radial_velocity(radius, theta, x); };
    auto tolerance_criterion = [tol](const double &a, const double &b){ return abs(a-b) <= tol; };

    double phi = -2. * numbers::PI;
    try
    {
        auto phi_interval
        = bracket_and_solve_root(
                function,
                phi_guess,
                2.0,
                true,
                tolerance_criterion,
                boost_max_iter);
        phi = 0.5 * (phi_interval.first + phi_interval.second);
    }
    catch (std::exception &exc)
    {
        AssertThrow(false, ExcBoostNoConvergence());
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
    {
        Assert((phi >= -numbers::PI) && (phi <= numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }
    else
    {
        Assert((phi >= 0.) && (phi <= 2. * numbers::PI),
               ExcAzimuthalAngleRange(phi));
    }

    return phi;
}

template<int dim>
void BuoyantFluidSolver<dim>::update_benchmark_point()
{
    const double radius = 0.5 * (1. + parameters.aspect_ratio);

    if (phi_benchmark < numbers::PI)
    {
        /**
         * A valid initial has not been computed yet. We try to find a benchmark
         * point from a set of trial points.
         */

        // initialize trial points
        const unsigned int n_trial_points = 16;
        std::vector<double> trial_points;
        for (unsigned int i=0; i<n_trial_points; ++i)
            trial_points.push_back(i * 2. * numbers::PI);

        bool            point_found = false;
        unsigned int    cnt = 0;
        while(cnt < n_trial_points && point_found == false)
        {
            const double gradient_at_trial_point
            = compute_azimuthal_gradient_of_radial_velocity(radius, 0, trial_points[cnt]);

            if (gradient_at_trial_point < 0.)
                continue;

            try
            {
                const double phi = compute_zero_of_radial_velocity(trial_points[cnt]);

                pcout << "Anticipated benchmark location: " << phi << std::endl;

                const double gradients_at_zero
                = compute_azimuthal_gradient_of_radial_velocity(radius, 0, trial_points[cnt]);

                if (gradients_at_zero > 0.)
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
        Assert(point_found, ExcNoBenchmarkPointFound(n_trial_points));
    }
    else
    {
        const double gradient_at_trial_point
        = compute_azimuthal_gradient_of_radial_velocity(radius, 0, phi_benchmark);

        Assert(gradient_at_trial_point > 0,
               ExcLowerRangeType<double>(0, gradient_at_trial_point));

        const double phi = compute_zero_of_radial_velocity(phi_benchmark);

        const double gradient_at_zero
        = compute_azimuthal_gradient_of_radial_velocity(radius, 0, phi);

        Assert(gradient_at_zero > 0,
               ExcLowerRangeType<double>(0, gradient_at_zero));

        phi_benchmark = phi;
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

template std::pair<double,double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_benchmark_requests_locally
(const double &,
 const double &,
 const double &) const;
template std::pair<double,double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_benchmark_requests_locally
(const double &,
 const double &,
 const double &) const;

template std::pair<double,double>
BuoyantFluid::BuoyantFluidSolver<2>::compute_benchmark_requests
(const double &,
 const double &,
 const double &) const;
template std::pair<double,double>
BuoyantFluid::BuoyantFluidSolver<3>::compute_benchmark_requests
(const double &,
 const double &,
 const double &) const;

template double
BuoyantFluid::BuoyantFluidSolver<2>::compute_zero_of_radial_velocity
(const double        &,
 const double        &,
 const unsigned int  &) const;
template double
BuoyantFluid::BuoyantFluidSolver<3>::compute_zero_of_radial_velocity
(const double        &,
 const double        &,
 const unsigned int  &) const;


template void
BuoyantFluid::BuoyantFluidSolver<2>::update_benchmark_point();
template void
BuoyantFluid::BuoyantFluidSolver<3>::update_benchmark_point();
