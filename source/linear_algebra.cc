/*
 * linear_algebra.cc
 *
 *  Created on: Jul 29, 2019
 *      Author: sg
 */

#include <deal.II/base/conditional_ostream.h>

#include <adsolic/linear_algebra.h>

namespace adsolic {

using namespace dealii;

LinearSolverParameters::LinearSolverParameters()
:
rel_tol(1e-6),
abs_tol(1e-9),
n_max_iter(200)
{}

LinearSolverParameters::LinearSolverParameters
(const std::string &parameter_filename)
:
LinearSolverParameters()
{
    ParameterHandler prm;
    declare_parameters(prm);

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
}

void LinearSolverParameters::declare_parameters
(ParameterHandler &prm)
{
    prm.enter_subsection("Linear solver parameters");

    prm.declare_entry("tol_rel",
            "1e-6",
            Patterns::Double(1.0e-15,1.0),
            "Relative tolerance of the linear solver.");

    prm.declare_entry("tol_abs",
            "1e-9",
            Patterns::Double(1.0e-15,1.0),
            "Absolute tolerance of the linear solver.");

    prm.declare_entry("n_max_iter",
            "200",
            Patterns::Integer(1,1000),
            "Maximum number of iterations of the linear solver.");

    prm.leave_subsection();
}

void LinearSolverParameters::parse_parameters
(ParameterHandler &prm)
{
    prm.enter_subsection("Convection diffusion parameters");

    rel_tol = prm.get_double("tol_rel");
    Assert(rel_tol > 0, ExcLowerRangeType<double>(rel_tol, 0));

    abs_tol = prm.get_double("tol_abs");
    Assert(abs_tol > 0, ExcLowerRangeType<double>(abs_tol, 0));

    n_max_iter = prm.get_integer("n_max_iter");
    Assert(n_max_iter > 0, ExcLowerRange(n_max_iter, 0));

    prm.leave_subsection();
}

template<typename Stream>
void LinearSolverParameters::write(Stream &stream) const
{
    stream << "Linear solver parameters" << std::endl
           << "   rel_tol: " << rel_tol << std::endl
           << "   abs_tol: " << abs_tol << std::endl
           << "   n_max_iter: " << n_max_iter << std::endl;
}

// explicit instantiation
template void LinearSolverParameters::write(std::ostream &) const;
template void LinearSolverParameters::write(ConditionalOStream &) const;

}  // namespace adsolic
