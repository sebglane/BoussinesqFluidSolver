/*
 * postprocessor.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

//#include "buoyant_fluid_solver.h"
#include <postprocessor.templates.h>

namespace BuoyantFluid {

template class PostProcessor<2>;
template class PostProcessor<3>;

}  // namespace BuoyantFluid

