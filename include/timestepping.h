/*
 * timestepping.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_TIMESTEPPING_H_
#define INCLUDE_TIMESTEPPING_H_

#include <assert.h>
#include <iomanip>
#include <iostream>
#include <array>

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
 * These functions return the coefficients of the variable time step IMEX stepping
 * schemes.
 *
 */
class IMEXCoefficients
{
public:

    IMEXCoefficients(const IMEXType &type_);

    std::array<double,3> alpha(const double omega);
    std::array<double,2> beta(const double omega);
    std::array<double,3> gamma(const double omega);

    void write(std::ostream &stream) const;

private:

    void compute_alpha();
    void compute_beta();
    void compute_gamma();

    const IMEXType      type;

    std::array<double,3> alpha_;
    std::array<double,2> beta_;
    std::array<double,3> gamma_;

    bool    update_alpha;
    bool    update_beta;
    bool    update_gamma;

    double  omega;
};

}  // namespace TimeStepping

#endif /* INCLUDE_TIMESTEPPING_H_ */
