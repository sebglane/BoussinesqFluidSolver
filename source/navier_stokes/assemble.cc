/*
 * assemble.cc
 *
 *  Created on: Jul 30, 2019
 *      Author: sg
 */

#include <adsolic/navier_stokes_solver.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/matrix_tools.h>

namespace adsolic
{

using namespace dealii;

using namespace NavierStokesAssembly;

template<int dim>
void NavierStokesSolver<dim>::local_assemble_diffusion_rhs
(const IteratorPair              &SI,
 Scratch::VelocityDiffusion<dim> &scratch,
 CopyData::RightHandSides<dim>   &data)
{
    scratch.fe_values_velocity.reinit(std::get<0>(*SI));
    scratch.fe_values_pressure.reinit(std::get<1>(*SI));

    std::get<0>(*SI)->get_dof_indices(data.local_dof_indices);

    scratch.fe_values_velocity[scratch.velocity].get_function_values
    (this->old_solution,
     scratch.old_velocity_values);

    scratch.fe_values_velocity[scratch.velocity].get_function_values
    (this->old_old_solution,
     scratch.old_old_velocity_values);

    scratch.fe_values_velocity[scratch.velocity].get_function_gradients
    (this->old_solution,
     scratch.old_velocity_gradients);

    scratch.fe_values_velocity[scratch.velocity].get_function_gradients
    (this->old_old_solution,
     scratch.old_old_velocity_gradients);

    scratch.fe_values_pressure.get_function_gradients
    (pressure.old_solution,
     scratch.old_pressure_values);

    if (parameters.pressure_projection_type == PressureProjectionType::Compact)
    {
        scratch.fe_values_pressure.get_function_gradients
        (pressure.update,
         scratch.pressure_update_values);
        scratch.fe_values_pressure.get_function_gradients
        (pressure.old_update,
         scratch.old_pressure_update_values);
    }

    data.local_matrix_for_bc = 0;
    data.local_rhs = 0;

    const double dt = this->timestepper.step_size();

    for (unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for (unsigned int k=0; k<data.dofs_per_cell; ++k)
        {
            scratch.phi_velocity[k]     = scratch.fe_values_velocity[scratch.velocity].value(k, q);
            scratch.grad_phi_velocity[k]= scratch.fe_values_velocity[scratch.velocity].gradient(k, q);
        }

        const double JxW = scratch.fe_values.JxW(q);

        double pressure_term;
        switch (parameters.pressure_projection_type)
        {
        case PressureProjectionType::Standard:
            pressure_term = scratch.old_pressure_values[q];
            break;
        case PressureProjectionType::Compact:
            pressure_term = scratch.old_pressure_values[q]
                          - alpha[1]/alpha[0] * scratch.pressure_update_values[q]
                          - alpha[2]/alpha[0] * scratch.old_pressure_update_values[q];
            break;
        case PressureProjectionType::NoneIncremental:
            pressure_term = 0;
            break;
        default:
            Assert(false, ExcNotImplemented());
            break;
        }

        const Tensor<1,dim> time_derivative_velocity
            = scratch.alpha[1] / dt  * scratch.old_velocity_values[q]
            + scratch.alpha[2] / dt * scratch.old_old_velocity_values[q];

        const Tensor<2,dim> linear_term_velocity
            = scratch.gamma[1] * scratch.old_velocity_gradients[q]
            + scratch.gamma[2] * scratch.old_old_velocity_gradients[q];

        Tensor<1,dim> nonlinear_term_velocity;
        switch (parameters.convective_weak_form)
        {
        case ConvectiveWeakForm::Standard:
            nonlinear_term_velocity
                = scratch.beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + scratch.beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];
            break;
        case ConvectiveWeakForm::DivergenceForm:
            nonlinear_term_velocity
                = scratch.beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + 0.5 * scratch.beta[0] * trace(scratch.old_velocity_gradients[q]) * scratch.old_velocity_values[q]
                + scratch.beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q]
                + 0.5 * scratch.beta[1] * trace(scratch.old_old_velocity_gradients[q]) * scratch.old_old_velocity_values[q];
            break;
        /*
        case ConvectiveWeakForm::SkewSymmetric:
            nonlinear_term_velocity
                = 0.5 * scratch.beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + 0.5 * scratch.beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];
            skew = true;
            break;
         */
        default:
            Assert(false, ExcNotImplemented());
            break;
        }

        for (unsigned int i=0; i<data.dofs_per_cell; ++i)
        {
            data.local_rhs(i)
                += (
                    - time_derivative_velocity * scratch.phi_velocity[i]
                    - nonlinear_term_velocity * scratch.phi_velocity[i]
                    - pressure_term * scratch.phi_velocity[i]
                    - equation_coefficient * scalar_product(linear_term_velocity, scratch.grad_phi_velocity[i])
                    ) * JxW;

            if (data.constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
            {
                for (unsigned int j=0; j<data.dofs_per_cell; ++j)
                    data.local_matrix_for_bc(j,i) +=
                            (
                              scratch.alpha[0] / dt *
                              scratch.phi_velocity[j] *
                              scratch.phi_velocity[i]
                            + scratch.gamma[0] * equation_coefficient  *
                              scalar_product(scratch.grad_phi_velocity[j], scratch.grad_phi_velocity[i])
                            ) * JxW;
            }
        }
    }
}

template<int dim>
void NavierStokesSolver<dim>::copy_local_to_global_diffusion_rhs
(const CopyData::RightHandSides<dim> &data)
{
    data.constraints.distribute_local_to_global
    (data.local_rhs,
     data.local_dof_indices,
     velocity.rhs,
     data.local_matrix_for_bc);
}


template<int dim>
void NavierStokesSolver<dim>::local_assemble_projection_rhs
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 Scratch::PressureProjection<dim> &scratch,
 CopyData::RightHandSides<dim> &data)
{
    scratch.fe_values_velocity.reinit(std::get<0>(*SI));
    scratch.fe_values_pressure.reinit(std::get<1>(*SI));

    std::get<1>(*SI)->get_dof_indices(data.local_dof_indices);

    scratch.fe_values_velocity[scratch.velocity].
    get_function_divergences(velocity.tentative_velocity,
                             scratch.velocity_divergences);

    data.local_matrix_for_bc = 0;
    data.local_rhs = 0.;

    for(unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for(unsigned int i=0; i<data.dofs_per_cell; ++i)
            scratch.phi_pressure[i] = scratch.fe_values_pressure.shape_value(i, q);

        const double JxW = scratch.fe_values_pressure.JxW(q);

        for (unsigned int i=0; i<data.dofs_per_cell; ++i)
        {
            data.local_rhs(i) -= scratch.velocity_divergences[q] *
                                 scratch.phi_pressure[i] *
                                 JxW;

            if (data.constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
            {
                for (unsigned int k=0; k<data.dofs_per_cell; ++k)
                    scratch.grad_phi_pressure[k] = scratch.fe_values_pressure.shape_grad(k, q);

                for (unsigned int j=0; j<data.dofs_per_cell; ++j)
                    data.local_matrix_for_bc(j,i) += scratch.grad_phi_pressure[j] *
                                                     scratch.grad_phi_pressure[i] *
                                                     JxW;
            }
        }
    }

}

template<int dim>
void NavierStokesSolver<dim>::copy_local_to_global_projection_rhs
(const CopyData::RightHandSides<dim> &data)
{
    data.constraints.distribute_local_to_global
    (data.local_rhs,
     data.local_dof_indices,
     pressure.rhs,
     data.local_matrix_for_bc);
}

template<int dim>
void NavierStokesSolver<dim>::local_assemble_correction_rhs
(const IteratorPair                 &SI,
 Scratch::VelocityCorrection<dim>   &scratch,
 CopyData::RightHandSides<dim>      &data)
{
    scratch.fe_values_velocity.reinit(std::get<0>(*SI));
    scratch.fe_values_pressure.reinit(std::get<1>(*SI));

    std::get<0>(*SI)->get_dof_indices(data.local_dof_indices);

    const double dt = timestepper.step_size();

    scratch.fe_values_velocity[scratch.velocity].
    get_function_values(velocity.tentative_velocity,
                        scratch.tentative_velocity_values);

    scratch.fe_values_pressure.
    get_function_gradients(pressure.update,
                           scratch.pressure_gradients);

    data.local_matrix_for_bc = 0;
    data.local_rhs = 0.;

    for(unsigned int q=0; q<scratch.n_q_points; ++q)
    {
        for(unsigned int i=0; i<data.dofs_per_cell; ++i)
            scratch.phi_velocity[i] = scratch.fe_values_velocity[scratch.velocity].
                                      value(i, q);

        for (unsigned int i=0; i<data.dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                      scratch.tentative_velocity_values[q] *
                      scratch.phi_velocity[i]
                    - dt / alpha[0] * scratch.pressure_gradients[q] *
                      scratch.phi_velocity[i]) *
                      scratch.fe_values_velocity.JxW(q);

            if (data.constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                for (unsigned int j=0; j<data.dofs_per_cell; ++j)
                    data.local_matrix_for_bc(j,i) += scratch.phi_velocity[j] *
                                                     scratch.phi_velocity[i] *
                                                     scratch.fe_values_velocity.JxW(q);
        }
    }
}

template<int dim>
void NavierStokesSolver<dim>::copy_local_to_global_correction_rhs
(const CopyData::RightHandSides<dim> &data)
{
    correction_constraints.distribute_local_to_global
    (data.local_rhs,
     data.local_dof_indices,
     velocity.rhs,
     data.local_matrix_for_bc);
}


// explicit instantiations
template void
NavierStokesSolver<2>::local_assemble_diffusion_rhs
(const IteratorPair            &,
 Scratch::VelocityDiffusion<2> &,
 CopyData::RightHandSides<2>   &);
template void
NavierStokesSolver<3>::local_assemble_diffusion_rhs
(const IteratorPair             &,
 Scratch::VelocityDiffusion<3>  &,
 CopyData::RightHandSides<3>    &);

template void
NavierStokesSolver<2>::copy_local_to_global_diffusion_rhs
(const CopyData::RightHandSides<2> &);
template void
NavierStokesSolver<3>::copy_local_to_global_diffusion_rhs
(const CopyData::RightHandSides<3> &);

template void
NavierStokesSolver<2>::local_assemble_projection_rhs
(const IteratorPair             &,
 Scratch::PressureProjection<2> &,
 CopyData::RightHandSides<2>    &);
template void
NavierStokesSolver<3>::local_assemble_projection_rhs
(const IteratorPair             &,
 Scratch::PressureProjection<3> &,
 CopyData::RightHandSides<3>    &);

template void
NavierStokesSolver<2>::copy_local_to_global_projection_rhs
(const NavierStokesAssembly::CopyData::RightHandSides<2> &);
template void
NavierStokesSolver<3>::copy_local_to_global_projection_rhs
(const NavierStokesAssembly::CopyData::RightHandSides<3> &);

template void
NavierStokesSolver<2>::local_assemble_correction_rhs
(const IteratorPair             &,
 Scratch::VelocityCorrection<2> &,
 CopyData::RightHandSides<2>    &);
template void
NavierStokesSolver<3>::local_assemble_correction_rhs
(const IteratorPair             &,
 Scratch::VelocityCorrection<3> &,
 CopyData::RightHandSides<3>    &);

template void
NavierStokesSolver<2>::copy_local_to_global_correction_rhs
(const CopyData::RightHandSides<2> &);
template void
NavierStokesSolver<3>::copy_local_to_global_correction_rhs
(const CopyData::RightHandSides<3> &);

}  // namespace adsolic



