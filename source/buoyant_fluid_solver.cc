/*
 * buoyant_fluid_solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "buoyant_fluid_solver.h"
#include "initial_values.h"
#include "postprocessor.h"
#include "preconditioning.h"

namespace BuoyantFluid {

template<int dim>
BuoyantFluidSolver<dim>::BuoyantFluidSolver(Parameters &parameters_)
:
parameters(parameters_),
imex_coefficients(parameters.imex_scheme),
triangulation(),
mapping(4),
// temperature part
temperature_fe(parameters.temperature_degree),
temperature_dof_handler(triangulation),
// stokes part
stokes_fe(FE_Q<dim>(parameters.velocity_degree), dim,
          FE_Q<dim>(parameters.velocity_degree - 1), 1),
stokes_dof_handler(triangulation),
// coefficients
equation_coefficients{(parameters.rotation ? 2.0/parameters.Ek: 0.0),
                      (parameters.rotation ? 1.0 : std::sqrt(parameters.Pr/ parameters.Ra) ),
                      (parameters.rotation ? parameters.Ra / parameters.Pr  : 1.0 ),
                      (parameters.rotation ? 1.0 / parameters.Pr : 1.0 / std::sqrt(parameters.Ra * parameters.Pr) )},
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
// time stepping
timestep(parameters.initial_timestep),
old_timestep(parameters.initial_timestep)
{
    std::cout << "Boussinesq solver by S. Glane\n"
              << "This program solves the Navier-Stokes system with thermal convection.\n"
              << "The stable Taylor-Hood (P2-P1) element and an approximative Schur complement solver is used.\n\n"
              << "The governing equations are\n\n"
              << "\t-- Incompressibility constraint:\n\t\t div(v) = 0,\n\n"
              << "\t-- Navier-Stokes equation:\n\t\tdv/dt + v . grad(v) + C1 Omega .times. v\n"
              << "\t\t\t\t= - grad(p) + C2 div(grad(v)) - C3 T g,\n\n"
              << "\t-- Heat conduction equation:\n\t\tdT/dt + v . grad(T) = C4 div(grad(T)).\n\n"
              << "The coefficients C1 to C4 depend on the normalization as follows.\n\n";

    // generate a nice table of the equation coefficients
    std::cout << "\n\n"
              << "+-------------------+----------+---------------+----------+-------------------+\n"
              << "|       case        |    C1    |      C2       |    C3    |        C4         |\n"
              << "+-------------------+----------+---------------+----------+-------------------+\n"
              << "| Non-rotating case |    0     | sqrt(Pr / Ra) |    1     | 1 / sqrt(Ra * Pr) |\n"
              << "| Rotating case     |  2 / Ek  |      1        |  Ra / Pr | 1 /  Pr           |\n"
              << "+-------------------+----------+---------------+----------+-------------------+\n";

    std::cout << std::endl << "You have chosen ";

    std::stringstream ss;
    ss << "+----------+----------+----------+----------+----------+----------+----------+\n"
       << "|    Ek    |    Ra    |    Pr    |    C1    |    C2    |    C3    |    C4    |\n"
       << "+----------+----------+----------+----------+----------+----------+----------+\n";

    if (parameters.rotation)
    {
        rotation_vector[dim-1] = 1.0;

        std::cout << "the rotating case with the following parameters: "
                  << std::endl;
        ss << "| ";
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ek;
        ss << " | ";
    }
    else
    {
        std::cout << "the non-rotating case with the following parameters: "
                  << std::endl;
        ss << "|     0    | ";
    }

    ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Ra;
    ss << " | ";
    ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Pr;
    ss << " | ";


    for (unsigned int n=0; n<4; ++n)
    {
        ss << std::setw(8) << std::setprecision(1) << std::scientific << std::right << equation_coefficients[n];
        ss << " | ";
    }

    ss << "\n+----------+----------+----------+----------+----------+----------+----------+\n";

    std::cout << std::endl << ss.str() << std::endl;

    std::cout << std::endl << std::fixed << std::flush;
}



template<int dim>
void BuoyantFluidSolver<dim>::make_grid()
{
    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    const Point<dim> center;
    const double ri = parameters.aspect_ratio;
    const double ro = 1.0;

    GridGenerator::hyper_shell(triangulation, center, ri, ro, (dim==3) ? 96 : 12);

    std::cout << "   Number of initial cells: "
              << triangulation.n_active_cells()
              << std::endl;

    static SphericalManifold<dim>       manifold(center);

    triangulation.set_all_manifold_ids(0);
    triangulation.set_all_manifold_ids_on_boundary(1);

    triangulation.set_manifold (0, manifold);
    triangulation.set_manifold (1, manifold);

    // setting boundary ids on coarsest grid
    const double tol = 1e-12;
    for(auto cell: triangulation.active_cell_iterators())
      if (cell->at_boundary())
          for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face(f)->at_boundary())
              {
                  std::vector<double> dist(GeometryInfo<dim>::vertices_per_face);
                  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                      dist[v] = cell->face(f)->vertex(v).distance(center);
                  if (std::all_of(dist.begin(), dist.end(),
                          [&ri,&tol](double d){return std::abs(d - ri) < tol;}))
                      cell->face(f)->set_boundary_id(EquationData::BoundaryIds::ICB);
                  if (std::all_of(dist.begin(), dist.end(),
                          [&ro,&tol](double d){return std::abs(d - ro) < tol;}))
                      cell->face(f)->set_boundary_id(EquationData::BoundaryIds::CMB);
              }

    // initial global refinements
    if (parameters.n_global_refinements > 0)
    {
        triangulation.refine_global(parameters.n_global_refinements);
        std::cout << "      Number of cells after "
                  << parameters.n_global_refinements
                  << " global refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    // initial boundary refinements
    if (parameters.n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<parameters.n_boundary_refinements; ++step)
        {
            for (auto cell: triangulation.active_cell_iterators())
                if (cell->at_boundary())
                    cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells after "
                  << parameters.n_boundary_refinements
                  << " boundary refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_dofs()
{
    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

    // temperature part
    temperature_dof_handler.distribute_dofs(temperature_fe);

    DoFRenumbering::boost::king_ordering(temperature_dof_handler);

    // stokes part
    stokes_dof_handler.distribute_dofs(stokes_fe);

    DoFRenumbering::boost::king_ordering(stokes_dof_handler);

    std::vector<unsigned int> stokes_block_component(dim+1,0);
    stokes_block_component[dim] = 1;

    DoFRenumbering::component_wise(stokes_dof_handler, stokes_block_component);

    // IO
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(stokes_dof_handler,
                                   dofs_per_block,
                                   stokes_block_component);

    const unsigned int n_temperature_dofs = temperature_dof_handler.n_dofs();

    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "      Number of degrees of freedom: "
              << stokes_dof_handler.n_dofs()
              << std::endl
              << "      Number of velocity degrees of freedom: "
              << dofs_per_block[0]
              << std::endl
              << "      Number of pressure degrees of freedom: "
              << dofs_per_block[1]
              << std::endl
              << "      Number of temperature degrees of freedom: "
              << n_temperature_dofs
              << std::endl;

    // temperature constraints
    {
        temperature_constraints.clear();

        DoFTools::make_hanging_node_constraints(
                temperature_dof_handler,
                temperature_constraints);

        const Functions::ConstantFunction<dim> icb_temperature(0.5);
        const Functions::ConstantFunction<dim> cmb_temperature(-0.5);

        const std::map<typename types::boundary_id, const Function<dim>*>
        temperature_boundary_values = {{EquationData::BoundaryIds::ICB, &icb_temperature},
                                       {EquationData::BoundaryIds::CMB, &cmb_temperature}};

        VectorTools::interpolate_boundary_values(
                temperature_dof_handler,
                temperature_boundary_values,
                temperature_constraints);

        temperature_constraints.close();
    }
    // temperature matrix and vector setup
    setup_temperature_matrices(n_temperature_dofs);

    temperature_solution.reinit(n_temperature_dofs);
    old_temperature_solution.reinit(n_temperature_dofs);
    old_old_temperature_solution.reinit(n_temperature_dofs);

    temperature_rhs.reinit(n_temperature_dofs);

    // stokes constraints
    {
        stokes_constraints.clear();

        DoFTools::make_hanging_node_constraints(
                stokes_dof_handler,
                stokes_constraints);

        const Functions::ZeroFunction<dim> zero_function(dim+1);

        const std::map<typename types::boundary_id, const Function<dim>*>
        velocity_boundary_values = {{EquationData::BoundaryIds::ICB, &zero_function},
                                    {EquationData::BoundaryIds::CMB, &zero_function}};

        const FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(
                stokes_dof_handler,
                velocity_boundary_values,
                stokes_constraints,
                stokes_fe.component_mask(velocities));

        stokes_constraints.close();
    }
    // stokes constraints for pressure laplace matrix
    if (parameters.assemble_schur_complement)
    {
        stokes_laplace_constraints.clear();

        stokes_laplace_constraints.merge(stokes_constraints);

        // find pressure boundary dofs
        const FEValuesExtractors::Scalar pressure(dim);

        std::vector<bool> boundary_dofs(stokes_dof_handler.n_dofs(),
                                        false);
        DoFTools::extract_boundary_dofs(stokes_dof_handler,
                                        stokes_fe.component_mask(pressure),
                                        boundary_dofs);

        // find first unconstrained pressure boundary dof
        unsigned int first_boundary_dof = stokes_dof_handler.n_dofs();
        std::vector<bool>::const_iterator
        dof = boundary_dofs.begin(),
        end_dof = boundary_dofs.end();
        for (; dof != end_dof; ++dof)
            if (*dof)
                if (!stokes_laplace_constraints.is_constrained(dof - boundary_dofs.begin()))
                {
                    first_boundary_dof = dof - boundary_dofs.begin();
                    break;
                }
        Assert(first_boundary_dof < stokes_dof_handler.n_dofs(),
               ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        std::vector<bool>::const_iterator it = std::find(boundary_dofs.begin(),
                                                         boundary_dofs.end(),
                                                         true);
        Assert(first_boundary_dof >= it - boundary_dofs.begin(),
                ExcMessage(std::string("Pressure boundary dof is not well constrained.").c_str()));

        // set first pressure boundary dof to zero
        stokes_laplace_constraints.add_line(first_boundary_dof);

        stokes_laplace_constraints.close();
    }
    else
    {
        stokes_laplace_constraints.clear();
    }
    // stokes matrix and vector setup
    setup_stokes_matrix(dofs_per_block);
    stokes_solution.reinit(dofs_per_block);
    old_stokes_solution.reinit(dofs_per_block);
    old_old_stokes_solution.reinit(dofs_per_block);
    stokes_rhs.reinit(dofs_per_block);
}


template<int dim>
void BuoyantFluidSolver<dim>::setup_temperature_matrices(const types::global_dof_index n_temperature_dofs)
{
    preconditioner_T.reset();

    temperature_matrix.clear();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();

    DynamicSparsityPattern dsp(n_temperature_dofs, n_temperature_dofs);

    DoFTools::make_sparsity_pattern(temperature_dof_handler,
                                    dsp,
                                    temperature_constraints);

    temperature_sparsity_pattern.copy_from(dsp);

    temperature_matrix.reinit(temperature_sparsity_pattern);
    temperature_mass_matrix.reinit(temperature_sparsity_pattern);
    temperature_stiffness_matrix.reinit(temperature_sparsity_pattern);

    rebuild_temperature_matrices = true;
}

template<int dim>
void BuoyantFluidSolver<dim>::setup_stokes_matrix(
        const std::vector<types::global_dof_index> dofs_per_block)
{
    preconditioner_A.reset();
    preconditioner_Kp.reset();
    preconditioner_Mp.reset();

    stokes_matrix.clear();
    stokes_laplace_matrix.clear();

    pressure_mass_matrix.clear();
    velocity_mass_matrix.clear();

    {
        TimerOutput::Scope timer_section(computing_timer, "setup stokes matrix");

        Table<2,DoFTools::Coupling> stokes_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d || c<dim || d<dim)
                    stokes_coupling[c][d] = DoFTools::always;
                else
                    stokes_coupling[c][d] = DoFTools::none;

        BlockDynamicSparsityPattern dsp(dofs_per_block,
                                        dofs_per_block);

        DoFTools::make_sparsity_pattern(
                stokes_dof_handler,
                stokes_coupling,
                dsp,
                stokes_constraints);

        stokes_sparsity_pattern.copy_from(dsp);

        stokes_matrix.reinit(stokes_sparsity_pattern);
    }
    {
        TimerOutput::Scope timer_section(computing_timer, "setup stokes laplace matrix");

        Table<2,DoFTools::Coupling> stokes_laplace_coupling(dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c==d && c==dim && d==dim)
                    stokes_laplace_coupling[c][d] = DoFTools::always;
                else
                    stokes_laplace_coupling[c][d] = DoFTools::none;

        BlockDynamicSparsityPattern laplace_dsp(dofs_per_block, dofs_per_block);

        const ConstraintMatrix &constraints_used =
                (parameters.assemble_schur_complement?
                                stokes_laplace_constraints: stokes_constraints);

        DoFTools::make_sparsity_pattern(stokes_dof_handler,
                                        stokes_laplace_coupling,
                                        laplace_dsp,
                                        constraints_used);

        if (!parameters.assemble_schur_complement)
            laplace_dsp.block(1,1)
            .compute_mmult_pattern(stokes_sparsity_pattern.block(1,0),
                                   stokes_sparsity_pattern.block(0,1));


        auxiliary_stokes_sparsity_pattern.copy_from(laplace_dsp);

        stokes_laplace_matrix.reinit(auxiliary_stokes_sparsity_pattern);

        stokes_laplace_matrix.block(0,0).reinit(stokes_sparsity_pattern.block(0,0));
    }
}


template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_temperature_matrix(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        TemperatureAssembly::Scratch::Matrix<dim> &scratch,
        TemperatureAssembly::CopyData::Matrix<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    scratch.temperature_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.grad_phi_T[k] = scratch.temperature_fe_values.shape_grad(k,q);
            scratch.phi_T[k]      = scratch.temperature_fe_values.shape_value(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_mass_matrix(i,j)
                    += scratch.phi_T[i] * scratch.phi_T[j] * scratch.temperature_fe_values.JxW(q);
                data.local_stiffness_matrix(i,j)
                    += scratch.grad_phi_T[i] * scratch.grad_phi_T[j] * scratch.temperature_fe_values.JxW(q);
            }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_mass_matrix(i,j) = data.local_mass_matrix(j,i);
            data.local_stiffness_matrix(i,j) = data.local_stiffness_matrix(j,i);
        }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_temperature_matrix(
        const TemperatureAssembly::CopyData::Matrix<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_mass_matrix,
            data.local_dof_indices,
            temperature_mass_matrix);
    temperature_constraints.distribute_local_to_global(
            data.local_stiffness_matrix,
            data.local_dof_indices,
            temperature_stiffness_matrix);
}


template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_temperature_rhs(
        const typename DoFHandler<dim>::active_cell_iterator    &cell,
        TemperatureAssembly::Scratch::RightHandSide<dim>        &scratch,
        TemperatureAssembly::CopyData::RightHandSide<dim>       &data)
{
    const std::vector<double> alpha = (timestep_number != 0?
                                            imex_coefficients.alpha(timestep/old_timestep):
                                            std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> beta = (timestep_number != 0?
                                            imex_coefficients.beta(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                            imex_coefficients.gamma(timestep/old_timestep):
                                            std::vector<double>({1.0,0.0,0.0}));

    const FEValuesExtractors::Vector    velocity(0);

    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    data.matrix_for_bc = 0;
    data.local_rhs = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.temperature_fe_values.reinit (cell);

    typename DoFHandler<dim>::active_cell_iterator
    stokes_cell(&triangulation,
                cell->level(),
                cell->index(),
                &stokes_dof_handler);
    scratch.stokes_fe_values.reinit(stokes_cell);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);
    scratch.temperature_fe_values.get_function_gradients(old_temperature_solution,
                                                         scratch.old_temperature_gradients);
    scratch.temperature_fe_values.get_function_gradients(old_old_temperature_solution,
                                                         scratch.old_old_temperature_gradients);

    scratch.stokes_fe_values[velocity].get_function_values(old_stokes_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_stokes_solution,
                                                           scratch.old_old_velocity_values);


    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            scratch.phi_T[i]      = scratch.temperature_fe_values.shape_value(i, q);
            scratch.grad_phi_T[i] = scratch.temperature_fe_values.shape_grad(i, q);
        }

        const double time_derivative_temperature =
                alpha[1] * scratch.old_temperature_values[q]
                    + alpha[2] * scratch.old_old_temperature_values[q];

        const double nonlinear_term_temperature =
                beta[0] * scratch.old_temperature_gradients[q] * scratch.old_velocity_values[q]
                    + beta[1] * scratch.old_old_temperature_gradients[q] * scratch.old_old_velocity_values[q];

        const Tensor<1,dim> linear_term_temperature =
                gamma[1] * scratch.old_temperature_gradients[q]
                    + gamma[2] * scratch.old_old_temperature_gradients[q];

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                    - time_derivative_temperature * scratch.phi_T[i]
                    - timestep * nonlinear_term_temperature * scratch.phi_T[i]
                    - timestep * equation_coefficients[3] * linear_term_temperature * scratch.grad_phi_T[i]
                    ) * scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    data.matrix_for_bc(j,i) += (
                                  alpha[0] * scratch.phi_T[i] * scratch.phi_T[j]
                                + gamma[0] * timestep * equation_coefficients[3] * scratch.grad_phi_T[i] * scratch.grad_phi_T[j]
                                ) * scratch.temperature_fe_values.JxW(q);

        }

    }
}

template <int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_temperature_rhs(
        const TemperatureAssembly::CopyData::RightHandSide<dim> &data)
{
    temperature_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            temperature_rhs,
            data.matrix_for_bc);
}

template<int dim>
void BuoyantFluidSolver<dim>::assemble_temperature_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble temperature system");

    std::cout << "   Assembling temperature system..." << std::endl;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    // assemble temperature matrices
    if (rebuild_temperature_matrices)
    {
        temperature_mass_matrix = 0;
        temperature_stiffness_matrix = 0;

        WorkStream::run(
                temperature_dof_handler.begin_active(),
                temperature_dof_handler.end(),
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
                                                          quadrature_formula),
                TemperatureAssembly::CopyData::Matrix<dim>(temperature_fe));

        const std::vector<double> alpha = (timestep_number != 0?
                                                imex_coefficients.alpha(timestep/old_timestep):
                                                std::vector<double>({1.0,-1.0,0.0}));
        const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0];
        temperature_matrix.add(timestep * gamma[0] * equation_coefficients[3],
                               temperature_stiffness_matrix);

        rebuild_temperature_matrices = false;
        rebuild_temperature_preconditioner = true;
    }
    else if (timestep_number == 1 || timestep_modified)
    {
        Assert(timestep_number != 0, ExcInternalError());

        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        temperature_matrix.copy_from(temperature_mass_matrix);
        temperature_matrix *= alpha[0];
        temperature_matrix.add(timestep * gamma[0] * equation_coefficients[3],
                               temperature_stiffness_matrix);

        rebuild_temperature_preconditioner = true;
    }
    // reset all entries
    temperature_rhs = 0;

    // assemble temperature right-hand side
    WorkStream::run(
            temperature_dof_handler.begin_active(),
            temperature_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_temperature_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_temperature_rhs,
                      this,
                      std::placeholders::_1),
            TemperatureAssembly::Scratch::RightHandSide<dim>(temperature_fe,
                                                             mapping,
                                                             quadrature_formula,
                                                             update_values|
                                                             update_gradients|
                                                             update_JxW_values,
                                                             stokes_fe,
                                                             update_values),
            TemperatureAssembly::CopyData::RightHandSide<dim>(temperature_fe));
}

template<int dim>
void BuoyantFluidSolver<dim>::build_temperature_preconditioner()
{
    if (!rebuild_temperature_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build temperature preconditioner");

    preconditioner_T.reset(new PreconditionerTypeT());

    PreconditionerTypeT::AdditionalData     data;
    data.relaxation = 0.6;

    preconditioner_T->initialize(temperature_matrix,
                                 data);

    rebuild_temperature_preconditioner = false;
}

template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_stokes_matrix(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        StokesAssembly::Scratch::Matrix<dim> &scratch,
        StokesAssembly::CopyData::Matrix<dim> &data)
{
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    scratch.stokes_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_v[k] = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_v[k] = scratch.stokes_fe_values[velocity].gradient(k, q);
            scratch.div_phi_v[k] = scratch.stokes_fe_values[velocity].divergence(k, q);
            scratch.phi_p[k] = scratch.stokes_fe_values[pressure].value(k, q);
            if (parameters.assemble_schur_complement)
            scratch.grad_phi_p[k] = scratch.stokes_fe_values[pressure].gradient(k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<=i; ++j)
            {
                data.local_matrix(i,j)
                    += (
                          scratch.phi_v[i] * scratch.phi_v[j]
                        - scratch.phi_p[i] * scratch.div_phi_v[j]
                        - scratch.div_phi_v[i] * scratch.phi_p[j]
                        + scratch.phi_p[i] * scratch.phi_p[j]
                        ) * scratch.stokes_fe_values.JxW(q);
                data.local_stiffness_matrix(i,j)
                    += (
                          scalar_product(scratch.grad_phi_v[i], scratch.grad_phi_v[j])
                        + (parameters.assemble_schur_complement?
                                scratch.grad_phi_p[i] * scratch.grad_phi_p[j] : 0)
                        ) * scratch.stokes_fe_values.JxW(q);
            }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        {
            data.local_matrix(i,j) = data.local_matrix(j,i);
            data.local_stiffness_matrix(i,j) = data.local_stiffness_matrix(j,i);
        }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix(
        const StokesAssembly::CopyData::Matrix<dim> &data)
{
    stokes_constraints.distribute_local_to_global(
            data.local_matrix,
            data.local_dof_indices,
            stokes_matrix);

    const ConstraintMatrix &constraints_used
    = (parameters.assemble_schur_complement?
            stokes_laplace_constraints: stokes_constraints);

    constraints_used.distribute_local_to_global(
            data.local_stiffness_matrix,
            data.local_dof_indices,
            stokes_laplace_matrix);
}


template <int dim>
void BuoyantFluidSolver<dim>::local_assemble_stokes_rhs(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        StokesAssembly::Scratch::RightHandSide<dim> &scratch,
        StokesAssembly::CopyData::RightHandSide<dim> &data)
{
    const std::vector<double> alpha = (timestep_number != 0?
                                        imex_coefficients.alpha(timestep/old_timestep):
                                        std::vector<double>({1.0,-1.0,0.0}));
    const std::vector<double> beta = (timestep_number != 0?
                                        imex_coefficients.beta(timestep/old_timestep):
                                        std::vector<double>({1.0,0.0}));
    const std::vector<double> gamma = (timestep_number != 0?
                                        imex_coefficients.gamma(timestep/old_timestep):
                                        std::vector<double>({1.0,0.0,0.0}));

    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    scratch.stokes_fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&triangulation,
                     cell->level(),
                     cell->index(),
                     &temperature_dof_handler);
    scratch.temperature_fe_values.reinit(temperature_cell);

    data.local_rhs = 0;

    scratch.stokes_fe_values[velocity].get_function_values(old_stokes_solution,
                                                           scratch.old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_values(old_old_stokes_solution,
                                                           scratch.old_old_velocity_values);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_stokes_solution,
                                                             scratch.old_velocity_gradients);
    scratch.stokes_fe_values[velocity].get_function_gradients(old_old_stokes_solution,
                                                             scratch.old_old_velocity_gradients);

    scratch.temperature_fe_values.get_function_values(old_temperature_solution,
                                                      scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values(old_old_temperature_solution,
                                                      scratch.old_old_temperature_values);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            scratch.phi_v[k] = scratch.stokes_fe_values[velocity].value(k, q);
            scratch.grad_phi_v[k] = scratch.stokes_fe_values[velocity].gradient(k, q);
        }

        const Tensor<1,dim> time_derivative_velocity
            = alpha[1] * scratch.old_velocity_values[q]
                + alpha[2] * scratch.old_old_velocity_values[q];

        const Tensor<1,dim> nonlinear_term_velocity
            = beta[0] * scratch.old_velocity_gradients[q] * scratch.old_velocity_values[q]
                + beta[1] * scratch.old_old_velocity_gradients[q] * scratch.old_old_velocity_values[q];

        const Tensor<2,dim> linear_term_velocity
            = gamma[1] * scratch.old_velocity_gradients[q]
                + gamma[2] * scratch.old_old_velocity_gradients[q];

        const Tensor<1,dim> extrapolated_velocity
            = (timestep != 0 ?
                (scratch.old_velocity_values[q] * (1 + timestep/old_timestep)
                        - scratch.old_old_velocity_values[q] * timestep/old_timestep)
                        : scratch.old_velocity_values[q]);
        const double extrapolated_temperature
            = (timestep != 0 ?
                (scratch.old_temperature_values[q] * (1 + timestep/old_timestep)
                        - scratch.old_old_temperature_values[q] * timestep/old_timestep)
                        : scratch.old_temperature_values[q]);

        const Tensor<1,dim> gravity_vector = EquationData::GravityVector<dim>().value(scratch.stokes_fe_values.quadrature_point(q));

        Tensor<1,dim>   coriolis_term;
        if (parameters.rotation)
        {
            if (dim == 2)
                coriolis_term = cross_product_2d(extrapolated_velocity);
            else if (dim == 3)
                coriolis_term = cross_product_3d(rotation_vector,
                                                 extrapolated_velocity);
            else
            {
                Assert(false, ExcInternalError());
            }
        }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            data.local_rhs(i)
                += (
                    - time_derivative_velocity * scratch.phi_v[i]
                    - timestep * nonlinear_term_velocity * scratch.phi_v[i]
                    - timestep * equation_coefficients[1] * scalar_product(linear_term_velocity, scratch.grad_phi_v[i])
                    - timestep * (parameters.rotation ? equation_coefficients[0] * coriolis_term * scratch.phi_v[i]: 0)
                    - timestep * equation_coefficients[2] * extrapolated_temperature * gravity_vector * scratch.phi_v[i]
                    ) * scratch.stokes_fe_values.JxW(q);
    }
}

template<int dim>
void BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs(
        const StokesAssembly::CopyData::RightHandSide<dim> &data)
{
    stokes_constraints.distribute_local_to_global(
            data.local_rhs,
            data.local_dof_indices,
            stokes_rhs);
}



template<int dim>
void BuoyantFluidSolver<dim>::assemble_stokes_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assemble stokes system");

    std::cout << "   Assembling stokes system..." << std::endl;

    const QGauss<dim> quadrature_formula(parameters.velocity_degree + 1);

    if (rebuild_stokes_matrices)
    {
        // reset all entries
        stokes_matrix = 0;
        stokes_laplace_matrix = 0;

            // assemble matrix
            WorkStream::run(
                    stokes_dof_handler.begin_active(),
                    stokes_dof_handler.end(),
                    std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_matrix,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_matrix,
                              this,
                              std::placeholders::_1),
                    StokesAssembly::Scratch::Matrix<dim>(
                            stokes_fe,
                            mapping,
                            quadrature_formula,
                            update_values|
                            update_gradients|
                            update_JxW_values),
                    StokesAssembly::CopyData::Matrix<dim>(stokes_fe));

            // copy velocity mass matrix
            velocity_mass_matrix.reinit(stokes_sparsity_pattern.block(0,0));
            velocity_mass_matrix.copy_from(stokes_matrix.block(0,0));

            // copy pressure mass matrix
            pressure_mass_matrix.reinit(stokes_sparsity_pattern.block(1,1));
            pressure_mass_matrix.copy_from(stokes_matrix.block(1,1));
            stokes_matrix.block(1,1) = 0;

            // time stepping coefficients
            const std::vector<double> alpha = (timestep_number != 0?
                                                imex_coefficients.alpha(timestep/old_timestep):
                                                std::vector<double>({1.0,-1.0,0.0}));
            const std::vector<double> gamma = (timestep_number != 0?
                                                imex_coefficients.gamma(timestep/old_timestep):
                                                std::vector<double>({1.0,0.0,0.0}));
            // correct (0,0)-block of stokes system
            stokes_matrix.block(0,0) *= alpha[0];
            stokes_matrix.block(0,0).add(timestep * equation_coefficients[1] * gamma[0],
                                         stokes_laplace_matrix.block(0,0));

            // adjust factors in the pressure matrices
            factor_Kp = alpha[0];
            factor_Mp = timestep * gamma[0] * equation_coefficients[1];

            // rebuilding pressure stiffness matrix preconditioner
            if (parameters.assemble_schur_complement == false)
            {
                Vector<double> tmp1(velocity_mass_matrix.m()), tmp2(tmp1);
                tmp1 = 1.0;
                tmp2 = 0.0;

                velocity_mass_matrix.precondition_Jacobi(tmp2, tmp1);
                stokes_matrix.block(1,0).mmult(stokes_laplace_matrix.block(1,1),
                                               stokes_matrix.block(0,1),
                                               tmp2,
                                               false);
            }

            preconditioner_Kp = std::shared_ptr<PreconditionerTypeKp>
            (new PreconditionerTypeKp());

            PreconditionerTypeKp::AdditionalData preconditioner_Kp_data;
            preconditioner_Kp->initialize(stokes_laplace_matrix.block(1,1),
                                          preconditioner_Kp_data);

            // rebuilding pressure mass matrix preconditioner
            preconditioner_Mp = std::shared_ptr<PreconditionerTypeMp>(new PreconditionerTypeMp());
            PreconditionerTypeMp::AdditionalData preconditioner_Mp_data;
            preconditioner_Mp_data.relaxation = 0.75;

            preconditioner_Mp->initialize(pressure_mass_matrix,
                                          preconditioner_Mp_data);

            // rebuild the preconditioner of the velocity block
            rebuild_stokes_preconditioner = true;

        // do not rebuild stokes matrices
        rebuild_stokes_matrices = false;
    }
    else if (timestep_number == 1 || timestep_modified)
    {
        Assert(timestep_number != 0, ExcInternalError());

        // time stepping coefficients
        const std::vector<double> alpha = imex_coefficients.alpha(timestep/old_timestep);
        const std::vector<double> gamma = imex_coefficients.gamma(timestep/old_timestep);

        // correct (0,0)-block of stokes system
        stokes_matrix.block(0,0).copy_from(velocity_mass_matrix);
        stokes_matrix.block(0,0) *= alpha[0];
        stokes_matrix.block(0,0).add(timestep * equation_coefficients[1] * gamma[0],
                                     stokes_laplace_matrix.block(0,0));

        // adjust factors in the pressure matrices
        factor_Kp = alpha[0];
        factor_Mp = timestep * gamma[0] * equation_coefficients[1];

        // rebuild the preconditioner of the velocity block
        rebuild_stokes_preconditioner = true;
    }
    // reset all entries
    stokes_rhs = 0;

    // assemble right-hand side function
    WorkStream::run(
            stokes_dof_handler.begin_active(),
            stokes_dof_handler.end(),
            std::bind(&BuoyantFluidSolver<dim>::local_assemble_stokes_rhs,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&BuoyantFluidSolver<dim>::copy_local_to_global_stokes_rhs,
                      this,
                      std::placeholders::_1),
            StokesAssembly::Scratch::RightHandSide<dim>(
                    stokes_fe,
                    mapping,
                    quadrature_formula,
                    update_values|
                    update_quadrature_points|
                    update_JxW_values|
                    update_gradients,
                    temperature_fe,
                    update_values),
            StokesAssembly::CopyData::RightHandSide<dim>(stokes_fe));
}
template<int dim>
void BuoyantFluidSolver<dim>::build_stokes_preconditioner()
{
    if (!rebuild_stokes_preconditioner)
        return;

    TimerOutput::Scope timer_section(computing_timer, "build stokes preconditioner");

    Assert(!rebuild_stokes_matrices, ExcInternalError());

    std::cout << "   Building stokes preconditioner..." << std::endl;

    preconditioner_A = std::shared_ptr<PreconditionerTypeA>
                       (new PreconditionerTypeA());

    std::vector<std::vector<bool>>  constant_modes;
    FEValuesExtractors::Vector      velocity_components(0);
    DoFTools::extract_constant_modes(stokes_dof_handler,
                                     stokes_fe.component_mask(velocity_components),
                                     constant_modes);

    PreconditionerTypeA::AdditionalData preconditioner_A_data;
    preconditioner_A_data.constant_modes = constant_modes;
    preconditioner_A_data.elliptic = true;
    preconditioner_A_data.higher_order_elements = true;
    preconditioner_A_data.smoother_sweeps = 2;
    preconditioner_A_data.aggregation_threshold = 0.02;

    preconditioner_A->initialize(stokes_matrix.block(0,0),
                                 preconditioner_A_data);

    rebuild_stokes_preconditioner = false;
}


template<int dim>
std::pair<double, double> BuoyantFluidSolver<dim>::compute_rms_values() const
{
    const QGauss<dim> quadrature_formula(parameters.velocity_degree + 1);

    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> stokes_fe_values(mapping,
                                   stokes_fe,
                                   quadrature_formula,
                                   update_values|update_JxW_values);

    FEValues<dim> temperature_fe_values(mapping,
                                        temperature_fe,
                                        quadrature_formula,
                                        update_values);

    std::vector<double>         temperature_values(n_q_points);
    std::vector<Tensor<1,dim>>  velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double rms_velocity = 0;
    double rms_temperature = 0;
    double volume = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    temperature_cell = temperature_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();

    for (; cell != endc; ++cell, ++temperature_cell)
    {
        stokes_fe_values.reinit(cell);
        temperature_fe_values.reinit(temperature_cell);

        temperature_fe_values.get_function_values(temperature_solution,
                                                  temperature_values);
        stokes_fe_values[velocities].get_function_values(stokes_solution,
                                                         velocity_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            rms_velocity += velocity_values[q] * velocity_values[q] * stokes_fe_values.JxW(q);
            rms_temperature += temperature_values[q] * temperature_values[q] * stokes_fe_values.JxW(q);
            volume += stokes_fe_values.JxW(q);
        }
    }

    rms_velocity /= volume;
    AssertIsFinite(rms_velocity);
    Assert(rms_velocity >= 0, ExcLowerRangeType<double>(rms_velocity, 0));

    rms_temperature /= volume;
    AssertIsFinite(rms_temperature);
    Assert(rms_temperature >= 0, ExcLowerRangeType<double>(rms_temperature, 0));

    return std::pair<double,double>(std::sqrt(rms_velocity), std::sqrt(rms_temperature));
}

template <int dim>
double BuoyantFluidSolver<dim>::compute_cfl_number() const
{
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping,
                            stokes_fe,
                            quadrature_formula,
                            update_values);

    std::vector<Tensor<1,dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double max_cfl = 0;

    for (auto cell : stokes_dof_handler.active_cell_iterators())
    {
        fe_values.reinit (cell);
        fe_values[velocities].get_function_values(stokes_solution,
                                                  velocity_values);
        double max_cell_velocity = 0;
        for (unsigned int q=0; q<n_q_points; ++q)
            max_cell_velocity = std::max(max_cell_velocity,
                                         velocity_values[q].norm());
        max_cfl = std::max(max_cfl,
                           max_cell_velocity / cell->diameter());
    }
    return max_cfl * timestep;
}

template<int dim>
void BuoyantFluidSolver<dim>::update_timestep(const double current_cfl_number)
{
    TimerOutput::Scope  timer_section(computing_timer, "update time step");

    std::cout << "   Updating time step..." << std::endl;

    old_timestep = timestep;
    timestep_modified = false;

    if (current_cfl_number > parameters.cfl_max || current_cfl_number < parameters.cfl_min)
    {
        timestep = 0.5 * (parameters.cfl_min + parameters.cfl_max)
                        * old_timestep / current_cfl_number;
        if (timestep == old_timestep)
            return;
        else if (timestep > parameters.max_timestep
                && old_timestep == parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
            return;
        }
        else if (timestep > parameters.max_timestep
                && old_timestep != parameters.max_timestep)
        {
            timestep = parameters.max_timestep;
            timestep_modified = true;
        }
        else if (timestep < parameters.min_timestep)
        {
            ExcLowerRangeType<double>(timestep, parameters.min_timestep);
        }
        else if (timestep < parameters.max_timestep)
        {
            timestep_modified = true;
        }
    }
    if (timestep_modified)
        std::cout << "      time step changed from "
                  << std::setw(6) << std::setprecision(2) << std::scientific << old_timestep
                  << " to "
                  << std::setw(6) << std::setprecision(2) << std::scientific << timestep
                  << std::endl;
}

template<int dim>
void BuoyantFluidSolver<dim>::output_results() const
{
    std::cout << "   Output results..." << std::endl;

    // create joint finite element
    const FESystem<dim> joint_fe(stokes_fe, 1,
                                 temperature_fe, 1);

    // create joint dof handler
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    Assert(joint_dof_handler.n_dofs() ==
           stokes_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
           ExcInternalError());

    // create joint solution
    Vector<double>      joint_solution;
    joint_solution.reinit(joint_dof_handler.n_dofs());

    {
        std::vector<types::global_dof_index> local_joint_dof_indices(joint_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_stokes_dof_indices(stokes_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_temperature_dof_indices(temperature_fe.dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
        joint_cell       = joint_dof_handler.begin_active(),
        joint_endc       = joint_dof_handler.end(),
        stokes_cell      = stokes_dof_handler.begin_active(),
        temperature_cell = temperature_dof_handler.begin_active();
        for (; joint_cell!=joint_endc; ++joint_cell, ++stokes_cell, ++temperature_cell)
        {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            stokes_cell->get_dof_indices(local_stokes_dof_indices);
            temperature_cell->get_dof_indices(local_temperature_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    Assert (joint_fe.system_to_base_index(i).second < local_stokes_dof_indices.size(),
                            ExcInternalError());
                    joint_solution(local_joint_dof_indices[i])
                    = stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
                else
                {
                    Assert (joint_fe.system_to_base_index(i).first.first == 1,
                            ExcInternalError());
                    Assert (joint_fe.system_to_base_index(i).second < local_temperature_dof_indices.size(),
                            ExcInternalError());
                    joint_solution(local_joint_dof_indices[i])
                    = temperature_solution(local_temperature_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
        }
    }

    // create post processor
    PostProcessor<dim>   postprocessor;

    // prepare data out object
    DataOut<dim>    data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    data_out.add_data_vector(joint_solution, postprocessor);
    data_out.build_patches();

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(timestep_number, 5) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

template<int dim>
void BuoyantFluidSolver<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    // error estimation based on temperature
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(temperature_dof_handler,
                                       QGauss<dim-1>(parameters.temperature_degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       temperature_solution,
                                       estimated_error_per_cell);
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.7, 0.3);

    // clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > parameters.n_max_levels)
        for (auto cell: triangulation.active_cell_iterators_on_level(parameters.n_max_levels))
            cell->clear_refine_flag();


    // preparing temperature solution transfer
    std::vector<Vector<double>> x_temperature(3);
    x_temperature[0] = temperature_solution;
    x_temperature[1] = old_temperature_solution;
    x_temperature[2] = old_old_temperature_solution;
    SolutionTransfer<dim,Vector<double>> temperature_transfer(temperature_dof_handler);

    // preparing temperature stokes transfer
    std::vector<BlockVector<double>> x_stokes(3);
    x_stokes[0] = stokes_solution;
    x_stokes[1] = old_stokes_solution;
    x_stokes[2] = old_old_stokes_solution;
    SolutionTransfer<dim,BlockVector<double>> stokes_transfer(stokes_dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    temperature_transfer.prepare_for_coarsening_and_refinement(x_temperature);
    stokes_transfer.prepare_for_coarsening_and_refinement(x_stokes);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of temperature solution
    {
        std::vector<Vector<double>> tmp_temperature(3);
        tmp_temperature[0].reinit(temperature_solution);
        tmp_temperature[1].reinit(temperature_solution);
        tmp_temperature[2].reinit(temperature_solution);
        temperature_transfer.interpolate(x_temperature, tmp_temperature);

        temperature_solution = tmp_temperature[0];
        old_temperature_solution = tmp_temperature[1];
        old_old_temperature_solution = tmp_temperature[2];

        temperature_constraints.distribute(temperature_solution);
        temperature_constraints.distribute(old_temperature_solution);
        temperature_constraints.distribute(old_old_temperature_solution);
    }
    // transfer of stokes solution
    {
        std::vector<BlockVector<double>>    tmp_stokes(3);
        tmp_stokes[0].reinit(stokes_solution);
        tmp_stokes[1].reinit(stokes_solution);
        tmp_stokes[2].reinit(stokes_solution);
        stokes_transfer.interpolate(x_stokes, tmp_stokes);

        stokes_solution = tmp_stokes[0];
        old_stokes_solution = tmp_stokes[1];
        old_old_stokes_solution = tmp_stokes[2];

        stokes_constraints.distribute(stokes_solution);
        stokes_constraints.distribute(old_stokes_solution);
        stokes_constraints.distribute(old_old_stokes_solution);
    }
    // set rebuild flags
    rebuild_stokes_matrices = true;
    rebuild_temperature_matrices = true;
}


template <int dim>
void BuoyantFluidSolver<dim>::solve()
{
    {
        std::cout << "   Solving temperature system..." << std::endl;
        TimerOutput::Scope  timer_section(computing_timer, "temperature solve");

        temperature_constraints.set_zero(temperature_solution);

        SolverControl solver_control(temperature_matrix.m(),
                1e-12 * temperature_rhs.l2_norm());

        SolverCG<>   cg(solver_control);
        cg.solve(temperature_matrix,
                 temperature_solution,
                 temperature_rhs,
                 *preconditioner_T);

        temperature_constraints.distribute(temperature_solution);

        std::cout << "      "
                << solver_control.last_step()
                << " CG iterations for temperature"
                << std::endl;
    }
    {
        std::cout << "   Solving stokes system..." << std::endl;

        TimerOutput::Scope  timer_section(computing_timer, "stokes solve");

        stokes_constraints.set_zero(stokes_solution);

        PrimitiveVectorMemory<BlockVector<double>> vector_memory;

        SolverControl solver_control(parameters.n_max_iter,
                                     std::max(parameters.abs_tol,
                                              parameters.rel_tol * stokes_rhs.l2_norm()));

        SolverFGMRES<BlockVector<double>>
        solver(solver_control,
               vector_memory,
               SolverFGMRES<BlockVector<double>>::AdditionalData(30, true));

        const Preconditioning::BlockSchurPreconditioner
        <PreconditionerTypeA, PreconditionerTypeMp, PreconditionerTypeKp>
        preconditioner(stokes_matrix,
                       pressure_mass_matrix,
                       stokes_laplace_matrix.block(1,1),
                       *preconditioner_A,
                       *preconditioner_Kp,
                       factor_Kp,
                       *preconditioner_Mp,
                       factor_Mp,
                       false);

        solver.solve(stokes_matrix,
                     stokes_solution,
                     stokes_rhs,
                     preconditioner);

        std::cout << "      "
                  << solver_control.last_step()
                  << " GMRES iterations for stokes system, "
                  << " (A: " << preconditioner.n_iterations_A()
                  << ", Kp: " << preconditioner.n_iterations_Kp()
                  << ", Mp: " << preconditioner.n_iterations_Mp()
                  << ")"
                  << std::endl;

        stokes_constraints.distribute(stokes_solution);
    }
}


template<int dim>
void BuoyantFluidSolver<dim>::run()
{
    make_grid();

    setup_dofs();

    const EquationData::TemperatureInitialValues<dim>
    initial_temperature(parameters.aspect_ratio,
                        1.0,
                        0.5,
                        -0.5);

    VectorTools::interpolate(mapping,
                             temperature_dof_handler,
                             initial_temperature,
                             old_temperature_solution);


    VectorTools::interpolate(mapping,
                             temperature_dof_handler,
                             Functions::ZeroFunction<dim>(1),
                             old_temperature_solution);

    temperature_constraints.distribute(old_temperature_solution);

    temperature_solution = old_temperature_solution;

    output_results();

    double time = 0;
    double cfl_number = 0;

    do
    {
        std::cout << "step: " << Utilities::int_to_string(timestep_number, 8) << ", "
                  << "time: " << time << ", "
                  << "time step: " << timestep
                  << std::endl;

        assemble_stokes_system();
        build_stokes_preconditioner();

        assemble_temperature_system();
        build_temperature_preconditioner();

        solve();
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute rms values");

            const std::pair<double,double> rms_values = compute_rms_values();

            std::cout << "   velocity rms value: "
                      << rms_values.first
                      << std::endl
                      << "   temperature rms value: "
                      << rms_values.second
                      << std::endl;
        }
        {
            TimerOutput::Scope  timer_section(computing_timer, "compute cfl number");

            cfl_number = compute_cfl_number();

            std::cout << "   current cfl number: "
                      << cfl_number
                      << std::endl;
        }
        if (timestep_number % parameters.output_frequency == 0
                && timestep_number != 0)
        {
            TimerOutput::Scope  timer_section(computing_timer, "output results");
            output_results();
        }
        // mesh refinement
        if ((timestep_number > 0)
                && (timestep_number % parameters.refinement_frequency == 0))
            refine_mesh();
        // adjust time step
        if (parameters.adaptive_timestep && timestep_number > 1)
            update_timestep(cfl_number);

        // copy temperature solution
        old_old_temperature_solution = old_temperature_solution;
        old_temperature_solution = temperature_solution;

        // extrapolate temperature solution
        temperature_solution.sadd(1. + timestep / old_timestep,
                                  timestep / old_timestep,
                                  old_old_temperature_solution);

        // extrapolate stokes solution
        old_old_stokes_solution = old_stokes_solution;
        old_stokes_solution = stokes_solution;

        // extrapolate stokes solution
        stokes_solution.sadd(1. + timestep / old_timestep,
                             timestep / old_timestep,
                             old_old_stokes_solution);
        // advance in time
        time += timestep;
        ++timestep_number;

    } while (timestep_number <= parameters.n_steps);

    if (parameters.n_steps % parameters.output_frequency != 0)
        output_results();

    std::cout << std::fixed;

    computing_timer.print_summary();
    computing_timer.reset();

    std::cout << std::endl;
}

}  // namespace BouyantFluid

// explicit instantiation
template class BuoyantFluid::BuoyantFluidSolver<2>
template class BuoyantFluid::BuoyantFluidSolver<3>
