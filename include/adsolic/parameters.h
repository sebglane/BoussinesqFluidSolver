/*
 * parameters.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <adsolic/gravity_field.h>
#include <adsolic/initial_values.h>
#include <adsolic/timestepping.h>

namespace BuoyantFluid {

using namespace dealii;
using namespace TimeStepping;

/*
 *
 * enumeration for the type of the weak form of the convective term
 *
 */
enum ConvectiveWeakForm
{
    Standard,
    DivergenceForm,
    SkewSymmetric,
    RotationalForm
};

/*
 *
 * enumeration for the type of discretization of the convective term
 *
 */
enum ConvectiveDiscretizationType
{
    LinearImplicit,
    Explicit
};

/*
 *
 * enumeration for the type of the pressure projection scheme
 *
 */
enum PressureUpdateType
{
    StandardForm,
    IrrotationalForm
};

struct Parameters
{
    Parameters(const std::string &parameter_filename);
    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    // runtime parameters
    unsigned int    dim;
    bool            resume_from_snapshot;

    bool            solve_temperature_equation;
    bool            solve_momentum_equation;

    // geometry parameters
    double          aspect_ratio;

    // mesh refinement parameters
    bool            adaptive_refinement;
    unsigned int    refinement_frequency;

    unsigned int    n_global_refinements;
    unsigned int    n_initial_refinements;
    unsigned int    n_boundary_refinements;
    unsigned int    n_max_levels;
    unsigned int    n_min_levels;

    // logging parameters
    unsigned int    vtk_frequency;
    unsigned int    global_avg_frequency;
    unsigned int    cfl_frequency;
    unsigned int    benchmark_frequency;
    unsigned int    snapshot_frequency;

    unsigned int    benchmark_start;

    bool            verbose;

    // physics parameters
    double  Pr;
    double  Ra;
    double  Ek;

    EquationData::GravityProfile    gravity_profile;

    bool    rotation;
    bool    buoyancy;

    // initial conditions
    EquationData::TemperaturePerturbation   temperature_perturbation;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;
    unsigned int    n_max_iter;

    // time stepping parameters
    TimeSteppingParameters timestepping;

    double          cfl_min;
    double          cfl_max;

    // discretization parameters
    PressureUpdateType              projection_scheme;
    ConvectiveWeakForm              convective_weak_form;
    ConvectiveDiscretizationType    convective_scheme;

    unsigned int temperature_degree;
    unsigned int velocity_degree;
};


}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS_H_ */
