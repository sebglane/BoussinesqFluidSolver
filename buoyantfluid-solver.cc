#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace BuoyantFluid {

using namespace dealii;

/*
 *
 * enumeration for boundary identifiers
 *
 */
enum BoundaryIds
{
    ICB,
    CMB,
};


namespace TimeStepping {
/*
 *
 * enumeration for the type of the IMEX time stepping type
 *
 */
enum IMEXType
{
    CNAB,
    MCNAB,
    CNLF,
    SBDF
};

/*
 *
 * These functions return the coefficients of the variable time step IMEX stepping
 * schemes.
 *
 */
class IMEXCoefficients
{
public:

    IMEXCoefficients(const IMEXType &type_);

    std::vector<double> alpha(const double omega);
    std::vector<double> beta(const double omega);
    std::vector<double> gamma(const double omega);

    void write(std::ostream &stream) const;

private:

    void compute_alpha();
    void compute_beta();
    void compute_gamma();

    const IMEXType      type;

    std::vector<double> alpha_;
    std::vector<double> beta_;
    std::vector<double> gamma_;

    bool    update_alpha;
    bool    update_beta;
    bool    update_gamma;

    double      omega;

};

IMEXCoefficients::IMEXCoefficients(const IMEXType &type_)
:
type(type_),
alpha_(3,0),
beta_(2,0),
gamma_(3,0),
update_alpha(true),
update_beta(true),
update_gamma(true),
omega(0)
{}

std::vector<double> IMEXCoefficients::alpha(const double    timestep_ratio)
{
    if (timestep_ratio != omega)
    {
        omega = timestep_ratio;

        update_alpha = true;
        update_beta = true;
        update_gamma = true;

    }

    compute_alpha();

    return alpha_;
}

std::vector<double> IMEXCoefficients::beta(const double timestep_ratio)
{
    if (timestep_ratio != omega)
    {
        omega = timestep_ratio;

        update_alpha = true;
        update_beta = true;
        update_gamma = true;

    }

    compute_beta();

    return beta_;
}

std::vector<double> IMEXCoefficients::gamma(const double    timestep_ratio)
{
    if (timestep_ratio != omega)
    {
        omega = timestep_ratio;

        update_alpha = true;
        update_beta = true;
        update_gamma = true;

    }

    compute_gamma();

    return gamma_;
}

void IMEXCoefficients::compute_alpha()
{
    if (!update_alpha)
        return;

    if (type == IMEXType::SBDF)
    {
        alpha_[0] = (1. + 2. * omega) / (1. + omega);
        alpha_[1] = -(1. + omega);
        alpha_[2] = (omega * omega) / (1. + omega);
    }
    else if (type == IMEXType::CNAB || type == IMEXType::MCNAB)
    {
        alpha_[0] = 1.0;
        alpha_[1] = -1.0;
    }
    else if (type == IMEXType::CNLF)
    {
        alpha_[0] = 1. / (1. + omega);
        alpha_[1] = omega - 1.;
        alpha_[2] = -(omega * omega) / (1. + omega);
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }
    update_alpha = false;
}

void IMEXCoefficients::compute_beta()
{
    if (!update_beta)
        return;

    if (type == IMEXType::SBDF)
    {
        beta_[0] = (1. + omega);
        beta_[1] = -omega;
    }
    else if (type == IMEXType::CNAB ||  type == IMEXType::MCNAB)
    {
        beta_[0] = (1. + 0.5 * omega);
        beta_[1] = -0.5 * omega;
    }
    else if (type == IMEXType::CNLF)
    {
        beta_[0] = 1.;
        beta_[1] = 0.;
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }

    update_beta = false;

}
void IMEXCoefficients::compute_gamma()
{
    if (!update_gamma)
        return;

    if (type == IMEXType::SBDF)
    {
        gamma_[0] = 1.0;
        gamma_[1] = 0.0;
    }
    else if (type == IMEXType::CNAB)
    {
        gamma_[0] = 0.5;
        gamma_[1] = 0.5;
    }
    else if (type == IMEXType::MCNAB)
    {
        gamma_[0] = (8. * omega + 1.)/ (16. * omega);
        gamma_[1] = (7. * omega - 1.)/ (16. * omega);
        gamma_[2] = omega / (16. * omega);
    }
    else if (type == IMEXType::CNLF)
    {
        gamma_[0] = 0.5 / omega;
        gamma_[1] = 0.5 * (1. - 1./omega);
        gamma_[2] = 0.5;
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }
    update_gamma = false;
}

void IMEXCoefficients::write(std::ostream &stream) const
{
    stream << std::endl
           << "+-----------+----------+----------+----------+\n"
           << "|   Index   |    n+1   |    n     |    n-1   |\n"
           << "+-----------+----------+----------+----------+\n"
           << "|   alpha   | ";
    for (const auto it: alpha_)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "|   beta    |    0     | ";
    for (const auto it: beta_)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }

    stream << std::endl << "|   gamma   | ";
    for (const auto it: gamma_)
    {
        stream << std::setw(8)
               << std::setprecision(1)
               << std::scientific
               << std::right
               << it;
        stream << " | ";
    }
    stream << std::endl
            << "+-----------+----------+----------+----------+\n";
}

}  // namespace TimeStepping


namespace EquationData {

template<int dim>
class TemperatureInitialValues : public Function<dim>
{
public:
    TemperatureInitialValues(const double inner_radius,
                             const double outer_radius,
                             const double inner_temperature,
                             const double outer_temperature);

    virtual double value(const Point<dim>   &point,
                         const unsigned int component = 0) const;

    virtual void vector_value(const Point<dim>  &point,
                              Vector<double>    &values) const;

private:
    const double ri;
    const double ro;
    const double Ti;
    const double To;
};

template <int dim>
TemperatureInitialValues<dim>::TemperatureInitialValues(
        const double inner_radius,
        const double outer_radius,
        const double inner_temperature,
        const double outer_temperature)
:
Function<dim>(1),
ri(inner_radius),
ro(outer_radius),
Ti(inner_temperature),
To(outer_temperature)
{
    Assert(To < Ti, ExcLowerRangeType<double>(To, Ti));
    Assert(ri < ro, ExcLowerRangeType<double>(ri, ro));
}

template <int dim>
double TemperatureInitialValues<dim>::value(
        const Point<dim>    &point,
        const unsigned int component) const
{
    const double radius = point.distance(Point<dim>());
    const double value = Ti + (To - Ti) / (ro - ri) * (radius - ri);
    return value;
}

template <int dim>
void TemperatureInitialValues<dim>::vector_value(
        const Point<dim> &p,
        Vector<double>   &values) const
{
    for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = value(p, c);
}

}  // namespace EquationData



namespace Assembly {

namespace Scratch {

template<int dim>
struct TemperatureMatrix
{
    TemperatureMatrix(const FiniteElement<dim> &temperature_fe,
                      const Mapping<dim>       &mapping,
                      const Quadrature<dim>    &temperature_quadrature);

    TemperatureMatrix(const TemperatureMatrix<dim>  &scratch);

    FEValues<dim>               temperature_fe_values;

    std::vector<double>         phi_T;
    std::vector<Tensor<1,dim>>  grad_phi_T;
};


template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(
        const FiniteElement<dim> &temperature_fe,
        const Mapping<dim>       &mapping,
        const Quadrature<dim>    &temperature_quadrature)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      update_values|
                      update_gradients|
                      update_JxW_values),
phi_T(temperature_fe.dofs_per_cell),
grad_phi_T(temperature_fe.dofs_per_cell)
{}

template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(const TemperatureMatrix &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_T(scratch.phi_T),
grad_phi_T(scratch.grad_phi_T)
{}

template<int dim>
struct TemperatureRightHandSide
{
    TemperatureRightHandSide(
            const FiniteElement<dim> &temperature_fe,
            const Mapping<dim>       &mapping,
            const Quadrature<dim>    &temperature_quadrature,
            const UpdateFlags         temperature_update_flags);

    TemperatureRightHandSide(const TemperatureRightHandSide<dim> &scratch);

    FEValues<dim>               temperature_fe_values;
    std::vector<double>         phi_T;
    std::vector<Tensor<1,dim>>  grad_phi_T;
    std::vector<double>         old_temperature_values;
    std::vector<double>         old_old_temperature_values;
    std::vector<Tensor<1,dim>>  old_temperature_gradients;
    std::vector<Tensor<1,dim>>  old_old_temperature_gradients;
};

template<int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
        const FiniteElement<dim>    &temperature_fe,
        const Mapping<dim>          &mapping,
        const Quadrature<dim>       &temperature_quadrature,
        const UpdateFlags            temperature_update_flags)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      temperature_quadrature,
                      temperature_update_flags),
phi_T(temperature_fe.dofs_per_cell),
grad_phi_T(temperature_fe.dofs_per_cell),
old_temperature_values(temperature_quadrature.size()),
old_old_temperature_values(temperature_quadrature.size()),
old_temperature_gradients(temperature_quadrature.size()),
old_old_temperature_gradients(temperature_quadrature.size())
{}

template<int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
        const TemperatureRightHandSide<dim> &scratch)
:
temperature_fe_values(scratch.temperature_fe_values.get_mapping(),
                      scratch.temperature_fe_values.get_fe(),
                      scratch.temperature_fe_values.get_quadrature(),
                      scratch.temperature_fe_values.get_update_flags()),
phi_T(scratch.phi_T),
grad_phi_T(scratch.grad_phi_T),
old_temperature_values(scratch.old_temperature_values),
old_old_temperature_values(scratch.old_old_temperature_values),
old_temperature_gradients(scratch.old_temperature_gradients),
old_old_temperature_gradients(scratch.old_old_temperature_gradients)
{}

}  // namespace Scratch

namespace CopyData {

template <int dim>
struct TemperatureMatrix
{
    TemperatureMatrix(const FiniteElement<dim> &temperature_fe);
    TemperatureMatrix(const TemperatureMatrix<dim> &data);

    FullMatrix<double>                      local_mass_matrix;
    FullMatrix<double>                      local_stiffness_matrix;

    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(const FiniteElement<dim> &temperature_fe)
:
local_mass_matrix(temperature_fe.dofs_per_cell),
local_stiffness_matrix(temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
TemperatureMatrix<dim>::TemperatureMatrix(const TemperatureMatrix<dim> &data)
:
local_mass_matrix(data.local_mass_matrix),
local_stiffness_matrix(data.local_stiffness_matrix),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
struct TemperatureRightHandSide
{
    TemperatureRightHandSide(const FiniteElement<dim>               &temperature_fe);
    TemperatureRightHandSide(const TemperatureRightHandSide<dim>    &data);

    Vector<double>                          local_rhs;
    FullMatrix<double>                      matrix_for_bc;
    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
    const FiniteElement<dim> &temperature_fe)
:
local_rhs(temperature_fe.dofs_per_cell),
matrix_for_bc(temperature_fe.dofs_per_cell,
              temperature_fe.dofs_per_cell),
local_dof_indices(temperature_fe.dofs_per_cell)
{}

template <int dim>
TemperatureRightHandSide<dim>::TemperatureRightHandSide(
    const TemperatureRightHandSide<dim> &data)
:
local_rhs(data.local_rhs),
matrix_for_bc(data.matrix_for_bc),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace CopyData

}  // namespace Assembly



}  // namespace BuoyantFluid
