/*
 * parameters.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include "initial_values.h"
#include "timestepping.h"

namespace BuoyantFluid {

using namespace dealii;

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
    unsigned int    n_steps;
    unsigned int    refinement_frequency;

    double          t_final;

    // logging parameters
    unsigned int    vtk_frequency;
    unsigned int    rms_frequency;
    unsigned int    cfl_frequency;

    bool            verbose;

    // physics parameters
    double  aspect_ratio;
    double  Pr;
    double  Ra;
    double  Ek;

    bool    rotation;

    // initial conditions
    EquationData::TemperaturePerturbation   temperature_perturbation;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;

    unsigned int    n_max_iter;

    // time stepping parameters
    TimeStepping::IMEXType  imex_scheme;

    double          initial_timestep;
    double          min_timestep;
    double          max_timestep;
    double          cfl_min;
    double          cfl_max;

    bool            adaptive_timestep;

    // discretization parameters
    PressureUpdateType              projection_scheme;
    ConvectiveWeakForm              convective_weak_form;

    unsigned int temperature_degree;
    unsigned int velocity_degree;

    // refinement parameters
    unsigned int n_global_refinements;
    unsigned int n_initial_refinements;
    unsigned int n_boundary_refinements;
    unsigned int n_max_levels;
};


}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS_H_ */
