/*
 * setup.cc
 *
 *  Created on: Jul 29, 2019
 *      Author: sg
 */

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/vector_tools.h>

#include <adsolic/navier_stokes_solver.h>

namespace adsolic
{


template<int dim>
void NavierStokesSolver<dim>::setup_dofs()
{
    this->pcout << "Setup dofs..." << std::endl;

    TimerOutput::Scope timer_section(*(this->computing_timer), "Nav.-St. setup dofs");

    /*
     * split boundary conditions into velocity and stokes part...
     */
    velocity.setup_dofs();
    pressure.setup_dofs();
}


// explicit instantiation
template void NavierStokesSolver<2>::setup_dofs();
template void NavierStokesSolver<3>::setup_dofs();

}  // namespace adsolic



