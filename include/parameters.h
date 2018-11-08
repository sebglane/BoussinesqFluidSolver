/*
 * parameters.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include "timestepping.h"

namespace BuoyantFluid {

using namespace dealii;

struct Parameters
{
    Parameters(const std::string &parameter_filename);
    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    // runtime parameters
    bool    workstream_assembly;
    bool    assemble_schur_complement;

    // physics parameters
    double aspect_ratio;
    double Pr;
    double Ra;
    double Ek;

    bool         rotation;

    // linear solver parameters
    double rel_tol;
    double abs_tol;
    unsigned int n_max_iter;

    // time stepping parameters
    TimeStepping::IMEXType  imex_scheme;

    unsigned int    n_steps;

    double  initial_timestep;
    double  min_timestep;
    double  max_timestep;
    double  cfl_min;
    double  cfl_max;

    bool    adaptive_timestep;

    // discretization parameters
    unsigned int temperature_degree;
    unsigned int velocity_degree;

    // refinement parameters
    unsigned int n_global_refinements;
    unsigned int n_initial_refinements;
    unsigned int n_boundary_refinements;
    unsigned int n_max_levels;

    unsigned int refinement_frequency;

    // logging parameters
    unsigned int output_frequency;
};


}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS_H_ */
