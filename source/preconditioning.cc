/*
 * preconditioning.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */


#include "preconditioning.h"

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/precondition.h>

namespace Preconditioning
{

template<class PreconditionerTypeA, class PreconditionerTypeMp, class PreconditionerTypeKp>
BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeMp, PreconditionerTypeKp>
::BlockSchurPreconditioner(
        const BlockSparseMatrix<double> &system_matrix_,
        const SparseMatrix<double>      &pressure_mass_matrix_,
        const SparseMatrix<double>      &pressure_laplace_matrix_,
        const PreconditionerTypeA       &preconditioner_A,
        const PreconditionerTypeKp      &preconditioner_Kp,
        const double                    factor_Kp,
        const PreconditionerTypeMp      &preconditioner_Mp,
        const double                    factor_Mp,
        const bool                      do_solve_A)
:
system_matrix(&system_matrix_),
pressure_mass_matrix(&pressure_mass_matrix_),
pressure_laplace_matrix(&pressure_laplace_matrix_),
preconditioner_A(preconditioner_A),
preconditioner_Mp(preconditioner_Mp),
preconditioner_Kp(preconditioner_Kp),
factor_Kp(factor_Kp),
factor_Mp(factor_Mp),
do_solve_A(do_solve_A),
n_iterations_A_(0),
n_iterations_Kp_(0),
n_iterations_Mp_(0)
{}

template<class PreconditionerTypeA, class PreconditionerTypeMp, class PreconditionerTypeKp>
void BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeMp, PreconditionerTypeKp>::
vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
{
    {
        SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());
        SolverCG<> solver(solver_control);

        dst.block(1) = 0;
        solver.solve(*pressure_mass_matrix,
                dst.block(1),
                src.block(1),
                preconditioner_Mp);
        n_iterations_Mp_ += solver_control.last_step();

        dst.block(1) *= -factor_Mp;
    }
    {
        SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());
        SolverCG<> solver(solver_control);

        Vector<double> tmp_pressure(dst.block(1).size());

        tmp_pressure = 0;
        solver.solve(*pressure_laplace_matrix,
                     tmp_pressure,
                     src.block(1),
                     preconditioner_Kp);
        n_iterations_Kp_ += solver_control.last_step();

        tmp_pressure *= -factor_Kp;

        dst.block(1) += tmp_pressure;
    }
    /*
     *
    VectorTools::subtract_mean_value(dst.block(1));
     *
     */
    Vector<double> tmp_vel(src.block(0).size());
    {
        system_matrix->block(0,1).vmult(tmp_vel, dst.block(1));
        tmp_vel *= -1.0;
        tmp_vel += src.block(0);
    }
    if (do_solve_A)
    {
        SolverControl solver_control(30,
                                     1e-3 * tmp_vel.l2_norm());
        SolverCG<> solver(solver_control);

        dst.block(0) = 0;
        solver.solve(system_matrix->block(0,0),
                     dst.block(0),
                     tmp_vel,
                     preconditioner_A);
        n_iterations_A_ += solver_control.last_step();
    }
    else
    {
        preconditioner_A.vmult(dst.block(0), tmp_vel);
        n_iterations_A_ += 1;
    }
}

template<class PreconditionerTypeA, class PreconditionerTypeMp, class PreconditionerTypeKp>
unsigned int BlockSchurPreconditioner<PreconditionerTypeA,PreconditionerTypeMp,PreconditionerTypeKp>::n_iterations_A() const
{
    return n_iterations_A_;
}

template<class PreconditionerTypeA, class PreconditionerTypeMp, class PreconditionerTypeKp>
unsigned int BlockSchurPreconditioner<PreconditionerTypeA,PreconditionerTypeMp,PreconditionerTypeKp>::n_iterations_Kp() const
{
    return n_iterations_Kp_;
}

template<class PreconditionerTypeA, class PreconditionerTypeMp, class PreconditionerTypeKp>
unsigned int BlockSchurPreconditioner<PreconditionerTypeA,PreconditionerTypeMp,PreconditionerTypeKp>::n_iterations_Mp() const
{
    return n_iterations_Mp_;
}

template class BlockSchurPreconditioner
<TrilinosWrappers::PreconditionAMG,
 SparseILU<double>,
 PreconditionSSOR<SparseMatrix<double>>>;

template class BlockSchurPreconditioner
<TrilinosWrappers::PreconditionAMG,
 SparseILU<double>,
 PreconditionJacobi<SparseMatrix<double>>>;

template class BlockSchurPreconditioner
<TrilinosWrappers::PreconditionAMG,
 PreconditionJacobi<SparseMatrix<double>>,
 SparseILU<double>>;

template class BlockSchurPreconditioner
<TrilinosWrappers::PreconditionAMG,
 PreconditionSSOR<SparseMatrix<double>>,
 SparseILU<double>>;

}  // namespace Preconditioning
