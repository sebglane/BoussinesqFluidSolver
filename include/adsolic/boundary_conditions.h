/*
 * boundary_conditions.h
 *
 *  Created on: Jul 24, 2019
 *      Author: sg
 */

#ifndef INCLUDE_BOUNDARY_CONDITIONS_H_
#define INCLUDE_BOUNDARY_CONDITIONS_H_

#include <array>
#include <map>
#include <memory.h>
#include <set>
#include <utility>

#include <deal.II/base/function.h>

namespace adsolic
{

using namespace dealii;

namespace BC
{

/**
 * Structure that keeps all information about boundary
 * conditions. Necessary to enable different classes to share the boundary
 * conditions.
 */
template <int dim>
struct ScalarBoundaryConditions
{
  ScalarBoundaryConditions();

  void clear_all_boundary_conditions();

  void set_dirichlet_bc(const types::boundary_id                boundary_id,
                        const std::shared_ptr<Function<dim>>   &dirichlet_function
                        = std::shared_ptr<Function<dim>>());

  void set_neumann_bc(const types::boundary_id              boundary_id,
                      const std::shared_ptr<Function<dim>> &neumann_function
                      = std::shared_ptr<Function<dim>>());

  void set_periodic_bc(const unsigned int       direction,
                       const types::boundary_id first_boundary_id,
                       const types::boundary_id second_boundary_id);

  std::map<types::boundary_id,std::shared_ptr<Function<dim>>>   dirichlet_bcs;
  std::map<types::boundary_id,std::shared_ptr<Function<dim>>>   neumann_bcs;

  std::array<std::pair<types::boundary_id,types::boundary_id>,dim>  periodic_bcs;

private:
  void check_boundary_id(const types::boundary_id boundary_id) const;
};

/**
 * Structure that keeps all information about boundary
 * conditions. Necessary to enable different classes to share the boundary
 * conditions.
 */
template <int dim>
struct NavierStokesBoundaryConditions
{
    NavierStokesBoundaryConditions();

  void clear_all_boundary_conditions();

  void set_dirichlet_bc_velocity(const types::boundary_id                boundary_id,
                                 const std::shared_ptr<Function<dim>>   &velocity_function
                                     = std::shared_ptr<Function<dim>>());

  void set_dirichlet_bc_pressure(const types::boundary_id              boundary_id,
                                 const std::shared_ptr<Function<dim>> &pressure_function
                                     = std::shared_ptr<Function<dim>>());

  void set_open_bc(const types::boundary_id              boundary_id,
                   const bool constrain_normal_flux = false,
                   const std::shared_ptr<Function<dim>> &pressure_function
                       = std::shared_ptr<Function<dim>>());

  void set_periodic_bc(const unsigned int       direction,
                       const types::boundary_id first_boundary_id,
                       const types::boundary_id second_boundary_id);

  std::map<types::boundary_id,std::shared_ptr<Function<dim>>>   dirichlet_bcs_velocity;
  std::map<types::boundary_id,std::shared_ptr<Function<dim>>>   dirichlet_bcs_pressure;
  std::map<types::boundary_id,std::shared_ptr<Function<dim>>>   open_bcs_pressure;

  std::set<types::boundary_id>  normal_flux;
  std::set<types::boundary_id>  no_slip;

  std::array<std::pair<types::boundary_id,types::boundary_id>,dim>  periodic_bcs;

private:
  void check_boundary_id(const types::boundary_id boundary_id) const;
};


}  // namespace BoundaryConditions

}  // namespace adsolic

#endif /* INCLUDE_BOUNDARY_CONDITIONS_H_ */
