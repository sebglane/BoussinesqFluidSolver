/*
 * navier_stokes_solver.cc
 *
 *  Created on: Jul 29, 2019
 *      Author: sg
 */
#include <deal.II/base/parameter_handler.h>

#include <adsolic/navier_stokes_solver.h>

namespace adsolic {

using namespace dealii;

NavierStokesParameters::NavierStokesParameters()
:
linear_solver_parameters(),
fe_degree_velocity(2),
fe_degree_pressure(1),
projection_scheme(PressureUpdateType::StandardForm),
convective_weak_form(ConvectiveWeakForm::SkewSymmetric),
verbose(false)
{}

NavierStokesParameters::NavierStokesParameters
(const std::string &parameter_filename)
:
NavierStokesParameters()
{
    ParameterHandler prm;
    declare_parameters(prm);
    linear_solver_parameters.declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename.c_str());

    if (!parameter_file)
    {
        parameter_file.close();

        std::ostringstream message;
        message << "Input parameter file <"
                << parameter_filename << "> not found. Creating a"
                << std::endl
                << "template file of the same name."
                << std::endl;

        std::ofstream parameter_out(parameter_filename.c_str());
        prm.print_parameters(parameter_out,
                             ParameterHandler::OutputStyle::Text);

        AssertThrow(false, ExcMessage(message.str().c_str()));
    }

    prm.parse_input(parameter_file);

    parse_parameters(prm);
    linear_solver_parameters.parse_parameters(prm);
}

void NavierStokesParameters::declare_parameters
(ParameterHandler &prm)
{
    prm.enter_subsection("navier_stokes_parameters");
    {
        prm.declare_entry("fe_degree_velocity",
                "0",
                Patterns::Integer(),
                "Polynomial degree of the velocity discretization. The polynomial "
                "degree of the pressure is automatically set to one less than the "
                "one of the velocity if it is not specified.");

        prm.declare_entry("fe_degree_pressure",
                "0",
                Patterns::Integer(),
                "Polynomial degree of the pressure discretization. The polynomial "
                "degree of the velocity is automatically set to one larger than the "
                "one of the pressure.");

        prm.declare_entry("pressure_update_type",
                "Standard",
                Patterns::Selection("Standard|Irrotational"),
                "Type of pressure projection scheme applied (Standard|Irrotational).");

        prm.declare_entry("convective_weak_form",
                "Standard",
                Patterns::Selection("Standard|DivergenceForm|"
                                    "SkewSymmetric|RotationalForm"),
                "Type of weak form of convective term (Standard|DivergenceForm"
                "|SkewSymmetric|RotationalForm).");

        prm.declare_entry("verbose",
                "false",
                Patterns::Bool(),
                "Flag to activate output of subroutines.");
    }
    prm.leave_subsection();
}

void NavierStokesParameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Discretization parameters");
    {
        fe_degree_velocity = prm.get_integer("p_degree_velocity");
        fe_degree_pressure = prm.get_integer("p_degree_temperature");

        if ((fe_degree_pressure == 1) && (fe_degree_velocity == 0))
            fe_degree_velocity = 2;
        else if (fe_degree_pressure == 0 && fe_degree_velocity == 2)
            fe_degree_pressure = 1;
        else
            Assert(false, ExcMessage("Incorrect specification of polynomial "
                                     "degrees."));

        const std::string projection_type_str
        = prm.get("pressure_update_type");

        if (projection_type_str == "Standard")
            projection_scheme = PressureUpdateType::StandardForm;
        else if (projection_type_str == "Irrotational")
            projection_scheme = PressureUpdateType::IrrotationalForm;
        else
            AssertThrow(false,
                        ExcMessage("Unexpected string for pressure update scheme."));

        const std::string convective_weak_form_str
        = prm.get("convective_weak_form");

        if (convective_weak_form_str == "Standard")
            convective_weak_form = ConvectiveWeakForm::Standard;
        else if (convective_weak_form_str == "DivergenceForm")
            convective_weak_form = ConvectiveWeakForm::DivergenceForm;
        else if (convective_weak_form_str == "SkewSymmetric")
            convective_weak_form = ConvectiveWeakForm::SkewSymmetric;
        else if (convective_weak_form_str == "RotationalForm")
            convective_weak_form = ConvectiveWeakForm::RotationalForm;
        else
            AssertThrow(false,
                        ExcMessage("Unexpected string for convective weak form."));

        verbose = prm.get_bool("verbose");
    }
    prm.leave_subsection();
}

template<int dim>
NavierStokesSolver<dim>::NavierStokesSolver
(const NavierStokesParameters &parameters_in,
 const parallel::distributed::Triangulation<dim> &triangulation_in,
 const MappingQ<dim>   &mapping_in,
 const IMEXTimeStepping&timestepper_in,
 const std::shared_ptr<const BC::NavierStokesBoundaryConditions<dim>> boundary_descriptor,
 const std::shared_ptr<TimerOutput> external_timer)
:
SolverBase<dim,LA::BlockVector>
(triangulation_in,
 mapping_in,
 timestepper_in,
 external_timer),
parameters(parameters_in),
equation_coefficients(parameters.equation_coefficients),
fe(FE_Q<dim>(parameters.fe_degree_velocity), dim,
   FE_Q<dim>(parameters.fe_degree_pressure), 1)
{
   if (boundary_descriptor.get() != 0)
       boundary_conditions = boundary_descriptor;
   else
       boundary_conditions = std::make_shared<const BC::NavierStokesBoundaryConditions<dim>>();
}

template<int dim>
const FiniteElement<dim> &
NavierStokesSolver<dim>::get_fe() const
{
    return fe;
}

template<int dim>
types::global_dof_index
NavierStokesSolver<dim>::n_dofs_velocity() const
{
    std::vector<unsigned int> block_component(2,0);
    block_component[1] = 1;
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(this->dof_handler,
                                   dofs_per_block,
                                   block_component);

    return dofs_per_block[0];

}

template<int dim>
types::global_dof_index
NavierStokesSolver<dim>::n_dofs_pressure() const
{
   std::vector<unsigned int> block_component(2,0);
   block_component[1] = 1;
   std::vector<types::global_dof_index> dofs_per_block(2);
   DoFTools::count_dofs_per_block(this->dof_handler,
                                  dofs_per_block,
                                  block_component);

   return dofs_per_block[1];
}

template<int dim>
void NavierStokesSolver<dim>::advance_in_time()
{
    if (parameters.verbose)
        this->pcout << "   Navier Stokes step..." << std::endl;

    this->computing_timer->enter_subsection("Nav.-St.");

    // extrapolate from old solutions
    this->extrapolate_solution();

    // assemble right-hand side (and system if necessary)
    assemble_system();

    // update solution vectors
    this->advance_solution();

    this->computing_timer->leave_subsection();
}

// explicit instantiation
template class NavierStokesSolver<2>;
template class NavierStokesSolver<3>;

}  // namespace adsolic
