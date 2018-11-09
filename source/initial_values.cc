/*
 * initial_values.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#include "initial_values.templates.h"

namespace EquationData {

template class TemperatureInitialValues<1>;
template class TemperatureInitialValues<2>;
template class TemperatureInitialValues<3>;

template class GravityVector<1>;
template class GravityVector<2>;
template class GravityVector<3>;
}  // namespace EquationData


