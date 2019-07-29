/*
 * solver_base.h
 *
 *  Created on: Jul 27, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_SOLVER_BASE_H_
#define INCLUDE_ADSOLIC_SOLVER_BASE_H_

#include <memory>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/distributed/tria.h>

#include <adsolic/linear_algebra.h>
#include <adsolic/timestepping.h>

namespace adsolic
{

using namespace dealii;

using namespace TimeStepping;

template <int dim,class VectorType=LA::Vector>
class SolverBase
{
public:
    SolverBase
    (const parallel::distributed::Triangulation<dim> &triangulation_in,
     const MappingQ<dim>   &mapping_in,
     const IMEXTimeStepping&timestepper_in,
     const std::shared_ptr<TimerOutput> external_timer =
             std::shared_ptr<TimerOutput>());

    virtual void advance_in_time() = 0;

    virtual void setup_problem() = 0;

    virtual void setup_initial_condition
    (const Function<dim,double> &initial_field) = 0;

    virtual const FiniteElement<dim> &get_fe() const = 0;

    virtual unsigned int fe_degree() const = 0;

    types::global_dof_index n_dofs() const;

    const DoFHandler<dim>  &get_dof_handler() const;

    const VectorType    &get_solution() const;
    std::vector<const VectorType*> get_old_solutions() const;

protected:
    // reference to common triangulation
    const parallel::distributed::Triangulation<dim>   &triangulation;

    // reference to common mapping
    const MappingQ<dim>&mapping;

    // reference to time stepper
    const IMEXTimeStepping   &timestepper;

    // parallel output
    ConditionalOStream  pcout;

    // pointer to monitor of computing times
    std::shared_ptr<TimerOutput> computing_timer;

    // FiniteElement and DoFHandler
    DoFHandler<dim> dof_handler;

    // vectors
    VectorType      solution;
    VectorType      old_solution;
    VectorType      old_old_solution;
    VectorType      rhs;

    virtual void advance_solution();

    virtual void extrapolate_solution();
};

}  // namespace adsolic

#endif /* INCLUDE_ADSOLIC_SOLVER_BASE_H_ */
