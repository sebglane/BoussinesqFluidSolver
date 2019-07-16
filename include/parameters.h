/*
 * parameters.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <initial_values.h>
#include <timestepping.h>
#include <postprocessor.h>

namespace BuoyantFluid {

using namespace dealii;
/*
 *
 * enumeration for the type of the geometry
 *
 */
enum GeometryType
{
    SphericalShell,
    Cavity
};

/*
 *
 * enumeration for the type of a coordinate system
 *
 */
enum CoordinateSystem
{
    Cartesian,
    Spherical
};


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
    unsigned int    n_steps;
    unsigned int    refinement_frequency;

    double          t_final;

    bool            adaptive_refinement;
    bool            resume_from_snapshot;

    // output parameters
    OutputFlags     output_flags;
    bool            output_benchmark_results;
    bool            output_point_probe;

    // point probe parameters
    unsigned int        point_probe_frequency;
    std::vector<double> point_coordinates;
    CoordinateSystem    point_coordinate_system;
    bool                point_probe_spherical;

    // benchmark parameters
    unsigned int    benchmark_frequency;
    unsigned int    benchmark_start;

    // logging parameters
    unsigned int    vtk_frequency;
    unsigned int    global_avg_frequency;
    unsigned int    cfl_frequency;
    unsigned int    snapshot_frequency;

    bool            verbose;

    // geometry parameters
    GeometryType    geometry;
    double          aspect_ratio;

    // physics parameters
    double  Pr;
    double  Ra;
    double  Ek;
    double  Pm;

    EquationData::GravityProfile    gravity_profile;

    bool    rotation;
    bool    buoyancy;
    bool    magnetism;
    bool    magnetic_induction;

    // initial conditions
    EquationData::TemperaturePerturbation   temperature_perturbation;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;

    unsigned int    max_iter_navier_stokes;
    unsigned int    max_iter_temperature;
    unsigned int    max_iter_magnetic;
    // time stepping parameters
    TimeStepping::IMEXType  imex_scheme;

    double          initial_timestep;
    double          min_timestep;
    double          max_timestep;
    double          cfl_min;
    double          cfl_max;

    unsigned int    adaptive_timestep_barrier;

    bool            adaptive_timestep;

    // discretization parameters
    PressureUpdateType              projection_scheme;
    ConvectiveWeakForm              convective_weak_form;
    ConvectiveDiscretizationType    convective_scheme;

    PressureUpdateType              magnetic_projection_scheme;

    unsigned int temperature_degree;
    unsigned int velocity_degree;
    unsigned int magnetic_degree;

    // refinement parameters
    unsigned int n_global_refinements;
    unsigned int n_initial_refinements;
    unsigned int n_boundary_refinements;
    unsigned int n_max_levels;
    unsigned int n_min_levels;
};

template<int dim>
Point<dim>  probe_point(const Parameters  &parameters);

}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS_H_ */
