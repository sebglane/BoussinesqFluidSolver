/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include "buoyant_fluid_solver.templates.h"

namespace BuoyantFluid {

template class BuoyantFluidSolver<2>;
template class BuoyantFluidSolver<3>;

}  // namespace BouyantFluid
