/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/matrix_tools.h>

#include "buoyant_fluid_solver.h"


namespace BuoyantFluid {

template<int dim>
void BuoyantFluidSolver<dim>::assemble_temperature_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble temperature system");

    if (parameters.verbose)
        pcout << "      Assembling temperature system..." << std::endl;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // quadrature formula
    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    // time stepping coefficients
    const std::vector<double> alpha = (timestep_number != 0?
                                        imex_coefficients.alpha(timestep/old_timestep):
                                        std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                        imex_coefficients.gamma(timestep/old_timestep):
                                        std::vector<double>({1.0,0.0,0.0}));

    // assemble temperature matrices
    if (rebuild_temperature_matrices)
    {
        temperature_mass_matrix = 0;
        temperature_stiffness_matrix = 0;

        // assemble temperature right-hand side
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           temperature_dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           temperature_dof_handler.end()),
                std::bind(&BuoyantFluidSolver<dim>::local_assemble_temperature_matrix,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_temperature_matrix,
                          this,
                          std::placeholders::_1),
                TemperatureAssembly::Scratch::Matrix<dim>(temperature_fe,
                                                          mapping,
                                                          quadrature_formula,
                                                          update_values|
                                                          update_gradients|
                                                          update_JxW_values),
                TemperatureAssembly::CopyData::Matrix<dim>(temperature_fe));

        temperature_mass_matrix.compress(VectorOperation::add);
        temperature_stiffness_matrix.compress(VectorOperation::add);

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0] / timestep;
        temperature_matrix.add(gamma[0] * equation_coefficients[3],
                               temperature_stiffness_matrix);

        temperature_matrix.compress(VectorOperation::add);

        rebuild_temperature_matrices = false;
        rebuild_temperature_preconditioner = true;
    }

    if (timestep_number == 0 || timestep_number == 1 || timestep_modified)
    {
        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0] / timestep;
        temperature_matrix.add(gamma[0] * equation_coefficients[3],
                               temperature_stiffness_matrix);

        temperature_matrix.compress(VectorOperation::add);

        rebuild_temperature_preconditioner = true;
    }

    // reset all entries
    temperature_rhs = 0;

    // assemble temperature right-hand side
    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       temperature_dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       temperature_dof_handler.end()),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_temperature_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_temperature_rhs,
                      this,
                      std::placeholders::_1),
            TemperatureAssembly::Scratch::RightHandSide<dim>(
                    temperature_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_gradients|
                    update_JxW_values,
                    navier_stokes_fe,
                    update_values,
                    alpha,
                    (timestep_number != 0?
                            imex_coefficients.beta(timestep/old_timestep):
                            std::vector<double>({1.0,0.0})),
                    gamma),
            TemperatureAssembly::CopyData::RightHandSide<dim>(temperature_fe));

    temperature_rhs.compress(VectorOperation::add);
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_navier_stokes_matrices()
{
    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);

    // reset all entries
    navier_stokes_matrix = 0;
    navier_stokes_mass_matrix = 0;
    navier_stokes_laplace_matrix = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // assemble matrix
    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       navier_stokes_dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       navier_stokes_dof_handler.end()),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_matrix,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix,
                      this,
                      std::placeholders::_1),
            NavierStokesAssembly::Scratch::Matrix<dim>(
                    navier_stokes_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_gradients|
                    update_JxW_values),
            NavierStokesAssembly::CopyData::Matrix<dim>(navier_stokes_fe));

    navier_stokes_matrix.compress(VectorOperation::add);
    navier_stokes_mass_matrix.compress(VectorOperation::add);
    navier_stokes_laplace_matrix.compress(VectorOperation::add);

    // rebuild both preconditionerss
    rebuild_projection_preconditioner = true;
    rebuild_pressure_mass_preconditioner = true;

    // do not rebuild stokes matrices again
    rebuild_navier_stokes_matrices = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_magnetic_matrices()
{
    const QGauss<dim>   quadrature_formula(parameters.magnetic_degree + 1);

    // reset all entries
    magnetic_matrix= 0;
    magnetic_mass_matrix = 0;
    magnetic_laplace_matrix = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // assemble matrix
    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       magnetic_dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       magnetic_dof_handler.end()),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_magnetic_matrix,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_magnetic_matrix,
                      this,
                      std::placeholders::_1),
            MagneticAssembly::Scratch::Matrix<dim>(
                    magnetic_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_gradients|
                    update_JxW_values),
            MagneticAssembly::CopyData::Matrix<dim>(magnetic_fe));

    magnetic_matrix.compress(VectorOperation::add);
    magnetic_mass_matrix.compress(VectorOperation::add);
    magnetic_laplace_matrix.compress(VectorOperation::add);

    // rebuild both preconditionerss
    rebuild_projection_preconditioner = true;
    rebuild_pressure_mass_preconditioner = true;

    // do not rebuild stokes matrices again
    rebuild_magnetic_matrices = false;
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_diffusion_system()
{
    if (parameters.verbose)
        pcout << "      Assembling diffusion system..." << std::endl;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // quadrature formula
    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);

    // time stepping coefficients
    const std::vector<double> alpha = (timestep_number != 0?
                                        imex_coefficients.alpha(timestep/old_timestep):
                                        std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                        imex_coefficients.gamma(timestep/old_timestep):
                                        std::vector<double>({1.0,0.0,0.0}));

    {
    TimerOutput::Scope timer_section(computing_timer, "assemble diff. sys., part 1");

    if (parameters.convective_scheme == ConvectiveDiscretizationType::LinearImplicit)
    {
        navier_stokes_matrix.block(0,0) = 0;

        // assemble right-hand side function
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           navier_stokes_dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           navier_stokes_dof_handler.end()),
                std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_convection_matrix,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_convection_matrix,
                          this,
                          std::placeholders::_1),
                NavierStokesAssembly::Scratch::ConvectionMatrix<dim>(
                        navier_stokes_fe,
                        mapping,
                        quadrature_formula,
                        update_values|
                        update_gradients|
                        update_JxW_values),
                NavierStokesAssembly::CopyData::ConvectionMatrix<dim>(navier_stokes_fe));
    }

    if (parameters.convective_scheme == ConvectiveDiscretizationType::LinearImplicit)
    {
        if (rebuild_navier_stokes_matrices)
            assemble_navier_stokes_matrices();

        // correct (0,0)-block of navier stokes system by add-operations
        navier_stokes_matrix.block(0,0).add(alpha[0] / timestep,
                                            navier_stokes_mass_matrix.block(0,0));
        navier_stokes_matrix.block(0,0).add(equation_coefficients[1] * gamma[0],
                                            navier_stokes_laplace_matrix.block(0,0));

        navier_stokes_matrix.compress(VectorOperation::add);

        // rebuild the preconditioner of diffusion solve
        rebuild_diffusion_preconditioner = true;
    }
    else if (timestep_number == 0 || timestep_number == 1 || timestep_modified ||
             rebuild_navier_stokes_matrices)
    {
        if (rebuild_navier_stokes_matrices)
            assemble_navier_stokes_matrices();

        // correct (0,0)-block of navier stokes system by copy-operation
        navier_stokes_matrix.block(0,0).copy_from(navier_stokes_mass_matrix.block(0,0));
        navier_stokes_matrix.block(0,0) *= alpha[0] / timestep;
        navier_stokes_matrix.block(0,0).add(equation_coefficients[1] * gamma[0],
                                            navier_stokes_laplace_matrix.block(0,0));

        navier_stokes_matrix.compress(VectorOperation::add);

        // rebuild the preconditioner of diffusion solve
        rebuild_diffusion_preconditioner = true;
    }
    }
    {
    TimerOutput::Scope timer_section(computing_timer, "assemble diff. sys., part 2");

    // reset all entries
    navier_stokes_rhs = 0;

    // compute extrapolated pressure
    LA::Vector  extrapolated_pressure(navier_stokes_rhs.block(1));
    LA::Vector  aux_distributed_pressure(navier_stokes_rhs.block(1));
    switch (timestep_number)
    {
        case 0:
            break;
        case 1:
            aux_distributed_pressure = old_phi_pressure.block(1);
            extrapolated_pressure.add(-alpha[1], aux_distributed_pressure);
            break;
        default:
            aux_distributed_pressure = old_phi_pressure.block(1);
            extrapolated_pressure.add(-alpha[1]/alpha[0], aux_distributed_pressure);
            aux_distributed_pressure = old_old_phi_pressure.block(1);
            extrapolated_pressure.add(-alpha[2]/alpha[0], aux_distributed_pressure);
            break;
    }
    extrapolated_pressure *= -1.0;

    // add pressure gradient to right-hand side
    navier_stokes_matrix.block(0,1).vmult(navier_stokes_rhs.block(0),
                                          extrapolated_pressure);

    // assemble right-hand side function
    if (parameters.convective_scheme == ConvectiveDiscretizationType::Explicit)
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           navier_stokes_dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           navier_stokes_dof_handler.end()),
                std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_rhs_explicit,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs,
                          this,
                          std::placeholders::_1),
                NavierStokesAssembly::Scratch::RightHandSide<dim>(
                        navier_stokes_fe,
                        mapping,
                        quadrature_formula,
                        update_values|
                        update_quadrature_points|
                        update_JxW_values|
                        update_gradients,
                        temperature_fe,
                        update_values,
                        alpha,
                        (timestep_number != 0?
                                imex_coefficients.beta(timestep/old_timestep):
                                std::vector<double>({1.0,0.0})),
                        gamma,
                        parameters.gravity_profile),
                NavierStokesAssembly::CopyData::RightHandSide<dim>(navier_stokes_fe));
    else if (parameters.convective_scheme == ConvectiveDiscretizationType::LinearImplicit)
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           navier_stokes_dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           navier_stokes_dof_handler.end()),
                std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_rhs_implicit,
                          this,
                          std::placeholders::_1,
                          std::placeholders::_2,
                          std::placeholders::_3),
                std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs,
                          this,
                          std::placeholders::_1),
                NavierStokesAssembly::Scratch::RightHandSide<dim>(
                        navier_stokes_fe,
                        mapping,
                        quadrature_formula,
                        update_values|
                        update_quadrature_points|
                        update_JxW_values|
                        update_gradients,
                        temperature_fe,
                        update_values,
                        alpha,
                        (timestep_number != 0?
                                imex_coefficients.beta(timestep/old_timestep):
                                std::vector<double>({1.0,0.0})),
                        gamma,
                        parameters.gravity_profile),
                NavierStokesAssembly::CopyData::RightHandSide<dim>(navier_stokes_fe));
    else
        throw   ExcInternalError();

    navier_stokes_rhs.compress(VectorOperation::add);
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_projection_system()
{
    if (parameters.verbose)
        pcout << "      Assembling projection system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble projection system");

    if (rebuild_navier_stokes_matrices)
        assemble_navier_stokes_matrices();

    LA::Vector  distributed_velocity(navier_stokes_rhs.block(0));
    distributed_velocity = navier_stokes_solution.block(0);

    navier_stokes_matrix.block(1,0).vmult(navier_stokes_rhs.block(1),
                                          distributed_velocity);

    navier_stokes_rhs.compress(VectorOperation::add);
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_magnetic_diffusion_system()
{
    if (parameters.verbose)
        pcout << "      Assembling magnetic diffusion system..." << std::endl;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // quadrature formula
    const QGauss<dim>   quadrature_formula(parameters.magnetic_degree + 1);

    // time stepping coefficients
    const std::vector<double> alpha = (timestep_number != 0?
                                        imex_coefficients.alpha(timestep/old_timestep):
                                        std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                        imex_coefficients.gamma(timestep/old_timestep):
                                        std::vector<double>({1.0,0.0,0.0}));

    {
    TimerOutput::Scope timer_section(computing_timer, "assemble mag. diff. sys., part 1");

    if (timestep_number == 0 || timestep_number == 1 || timestep_modified ||
            rebuild_magnetic_matrices)
    {
        if (rebuild_magnetic_matrices)
            assemble_magnetic_matrices();

        // correct (0,0)-block of magnetic system by copy-operation
        magnetic_matrix.block(0,0).copy_from(magnetic_mass_matrix.block(0,0));
        magnetic_matrix.block(0,0) *= alpha[0] / timestep;
        magnetic_matrix.block(0,0).add(equation_coefficients[5] * gamma[0],
                                       magnetic_laplace_matrix.block(0,0));

        // add stabilization term to system matrix
        magnetic_matrix.block(0,0).add(1.0,
                                       magnetic_stabilization_matrix.block(0,0));


        magnetic_matrix.compress(VectorOperation::add);

        // TODO: separated preconditioner flags
        // rebuild the preconditioner of diffusion solve
        rebuild_magnetic_diffusion_preconditioner = true;
    }
    }
    {
    TimerOutput::Scope timer_section(computing_timer, "assemble mag. diff. sys., part 2");

    // reset all entries
    magnetic_rhs = 0;

    // compute extrapolated pressure
    LA::Vector  extrapolated_pressure(magnetic_rhs.block(1));
    LA::Vector  aux_distributed_pressure(magnetic_rhs.block(1));
    switch (timestep_number)
    {
        case 0:
            break;
        case 1:
            aux_distributed_pressure = old_phi_pseudo_pressure.block(1);
            extrapolated_pressure.add(-alpha[1], aux_distributed_pressure);
            break;
        default:
            aux_distributed_pressure = old_phi_pseudo_pressure.block(1);
            extrapolated_pressure.add(-alpha[1]/alpha[0], aux_distributed_pressure);
            aux_distributed_pressure = old_old_phi_pseudo_pressure.block(1);
            extrapolated_pressure.add(-alpha[2]/alpha[0], aux_distributed_pressure);
            break;
    }
    extrapolated_pressure *= -1.0;

    // TODO: check the signs
    // add pressure gradient to right-hand side
    magnetic_matrix.block(0,1).vmult(magnetic_rhs.block(0),
                                     extrapolated_pressure);

    // assemble right-hand side function
    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       magnetic_dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(),
                       magnetic_dof_handler.end()),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_magnetic_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_magnetic_rhs,
                      this,
                      std::placeholders::_1),
            MagneticAssembly::Scratch::RightHandSide<dim>(
                    magnetic_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_JxW_values|
                    update_gradients,
                    navier_stokes_fe,
                    update_values,
                    alpha,
                    (timestep_number != 0?
                            imex_coefficients.beta(timestep/old_timestep):
                            std::vector<double>({1.0,0.0})),
                    gamma),
                    MagneticAssembly::CopyData::RightHandSide<dim>(magnetic_fe));
    magnetic_rhs.compress(VectorOperation::add);
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_magnetic_projection_system()
{
    if (parameters.verbose)
        pcout << "      Assembling magnetic projection system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "assemble projection system");

    if (rebuild_magnetic_matrices)
        assemble_magnetic_matrices();

    // compute stiffness matrix with PSPG term
    if (timestep_number == 0 || timestep_modified)
    {
        magnetic_matrix.block(1,1).copy_from(magnetic_laplace_matrix.block(1,1));

        // compute stiffness matrix with PSPG term
        magnetic_matrix.block(1,1).add(1. / timestep,
                                       magnetic_stabilization_matrix.block(1,1));
    }

    LA::Vector  distributed_magnetic_field(magnetic_rhs.block(0));
    distributed_magnetic_field = magnetic_solution.block(0);

    magnetic_matrix.block(1,0).vmult(magnetic_rhs.block(1),
                                     distributed_magnetic_field);

    magnetic_rhs.compress(VectorOperation::add);
}

}  // namespace BuoyantFluid

// explicit instantiation

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_temperature_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_temperature_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_navier_stokes_matrices();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_navier_stokes_matrices();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_magnetic_matrices();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_magnetic_matrices();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_diffusion_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_diffusion_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_projection_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_projection_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_magnetic_diffusion_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_magnetic_diffusion_system();

template void BuoyantFluid::BuoyantFluidSolver<2>::assemble_magnetic_projection_system();
template void BuoyantFluid::BuoyantFluidSolver<3>::assemble_magnetic_projection_system();
