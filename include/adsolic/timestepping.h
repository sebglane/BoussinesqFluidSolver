/*
 * timestepping.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_TIMESTEPPING_H_
#define INCLUDE_TIMESTEPPING_H_

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/smartpointer.h>

#include <array>

using namespace dealii;

namespace TimeStepping {

/*
 *
 * enumeration for the type of the IMEX time stepping type
 *
 */
enum IMEXType
{
    Euler,
    CNAB,
    MCNAB,
    CNLF,
    SBDF
};

/*
 *
 * This class manages the time stepping parameters
 *
 */
struct TimeSteppingParameters
{
    TimeSteppingParameters();
    TimeSteppingParameters(const std::string &parameter_filename);

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    /*
     * function forwarding parameters to a stream object
     */
    template<typename Stream>
    void write(Stream &stream) const;


    // time stepping parameters
    TimeStepping::IMEXType  imex_scheme;

    unsigned int    n_steps;

    bool            adaptive_timestep;
    unsigned int    adaptive_timestep_barrier;

    double          initial_timestep;
    double          min_timestep;
    double          max_timestep;
    double          final_time;

    bool            verbose;
};

/*
 *
 * This class manages the IMEX time stepping schemes.
 *
 */
class IMEXTimeStepping : public Subscriptor
{
public:
    IMEXTimeStepping(const TimeSteppingParameters &prm);

    /*
     * function forwarding IMEX coefficients to a stream object
     */
    template<typename Stream>
    void write(Stream &stream) const;

    /*
     * function forwarding current time and step size to a stream object
     */
    template<typename Stream>
    void print_info(Stream &stream) const;

    void   set_time_step(double);
    double advance_in_time();

    void reset(const double initial_step = 0.0);

    /*
     * functions returning array of the IMEX coefficients
     */
    const std::array<double,3>& alpha() const;
    const std::array<double,2>& beta() const;
    const std::array<double,3>& gamma() const;

    /*
     * inline functions
     */
    double start() const;
    double final() const;
    double step_size() const;
    double min_step_size() const;
    double max_step_size() const;
    double old_step_size() const;
    double now() const;
    double previous() const;
    double pre_previous() const;
    bool   coefficients_have_changed() const;
    unsigned int step_no() const;
    bool    at_end() const;

    // Extrapolates two values from to the previous time levels to the current
    // time level
    template <typename Number>
    Number extrapolate(const Number &old_value,
                       const Number &old_old_value) const;

    std::string name() const;
    IMEXType    scheme() const;

private:
    const IMEXType      type;
    /*
     * IMEX coefficients
     */
    std::array<double,3>    alpha_array;
    std::array<double,2>    beta_array;
    std::array<double,3>    gamma_array;
    /*
     * start and end time
     */
    const double start_time;
    const double end_time;
    /*
     * time step variables
     */
    const double start_step_val;
    const double min_step_val;
    const double max_step_val;
    double step_val;
    double old_step_val;
    double old_old_step_val;
    double omega;
    /*
     * variables of current and previous time
     */
    double current_time;
    double previous_time;
    double pre_previous_time;
    /*
     * extrapolation factors
     */
    double old_extrapol_factor;
    double old_old_extrapol_factor;
    /*
     * adaptivity parameters
     */
    const bool adaptive_timestep;
    const unsigned int adaptive_barrier;
    /*
     * timestep counter
     */
    unsigned int step_no_val;
    unsigned int max_step_no;
    /*
     * boolean flags
     */
    bool at_end_time;
    bool coefficients_changed;
    const bool verbose;
    /*
     * update coefficients
     */
    void update_alpha();
    void update_beta();
    void update_gamma();
    void update_extrapol_factors();
    /*
     * set time step
     */
    void set_step_adaptive(const double &desired_step_size);
    void set_step_fixed();
};

inline double
IMEXTimeStepping::start() const
{
    return start_time;
}


inline double
IMEXTimeStepping::final() const
{
    return end_time;
}

inline double
IMEXTimeStepping::step_size() const
{
  return step_val;
}

inline double
IMEXTimeStepping::old_step_size() const
{
    return old_step_val;
}

inline double
IMEXTimeStepping::min_step_size() const
{
    return min_step_val;
}

inline double
IMEXTimeStepping::max_step_size() const
{
    return max_step_val;
}

inline double
IMEXTimeStepping::now() const
{
    return current_time;
}

inline double
IMEXTimeStepping::previous() const
{
    return previous_time;
}

inline double
IMEXTimeStepping::pre_previous() const
{
    return pre_previous_time;
}

inline unsigned int
IMEXTimeStepping::step_no() const
{
    return step_no_val;
}

inline bool
IMEXTimeStepping::coefficients_have_changed() const
{
    return coefficients_changed;
}

template <typename Number>
inline Number
IMEXTimeStepping::extrapolate
(const Number &old,
 const Number &old_old) const
{
    return old * old_extrapol_factor + old_old * old_old_extrapol_factor;
}

inline bool
IMEXTimeStepping::at_end() const
{
    return at_end_time || (step_no() == max_step_no);
}

inline void IMEXTimeStepping::reset(const double initial_step)
{
    at_end_time = false;
    step_no_val = 0;
    current_time = start();
    previous_time = 0.0;
    pre_previous_time = 0.0;
    if (!adaptive_timestep && initial_step > 0.0)
        step_val = initial_step;
    else
        step_val = start_step_val;
    old_step_val = 0.0;
    old_old_step_val = 0.0;
    omega = 1.0;
    old_extrapol_factor = 1.0;
    old_old_extrapol_factor = 0.0;
}

template<typename Stream>
inline void IMEXTimeStepping::print_info
(Stream &stream) const
{
    std::ostringstream ss;
    if (!at_end_time)
        ss << std::setprecision(8)
           << "time: " << std::setw(12) << std::right
           << (now() > 1e3 || (now() < 1e-8 && now() > 0.0)? std::scientific: std::fixed) << now()  << ", "
           << "step size: " << std::setw(12) << std::right
           << (step_size() > 1e3 || step_size() < 1e-8? std::scientific: std::fixed) << step_size()
           << std::endl;
    else
        ss << std::setprecision(8)
           << "time: " << std::setw(12) << std::right
           << (now() > 1e3 || (now() < 1e-8 && now() > 0.0)? std::scientific: std::fixed) << now()  << ", "
           << "finished time integration"
           << std::endl;

    stream << "Step No: "
           << Utilities::int_to_string(step_no(), 10) << ", "
           << ss.str().c_str()
           << std::flush;
}

}  // namespace IMEXTimeStepping

#endif /* INCLUDE_TIMESTEPPING_H_ */
