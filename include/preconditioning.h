/*
 * preconditioning.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PRECONDITIONING_H_
#define INCLUDE_PRECONDITIONING_H_

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

namespace Preconditioning
{

using namespace dealii;

template<class PreconditionerTypeA, class PreconditionerTypeMp, class PreconditionerTypeKp>
class BlockSchurPreconditioner : public Subscriptor
{
public:
    BlockSchurPreconditioner(
            const BlockSparseMatrix<double> &system_matrix,
            const SparseMatrix<double>      &pressure_mass_matrix,
            const SparseMatrix<double>      &pressure_laplace_matrix,
            const PreconditionerTypeA       &preconditioner_A,
            const PreconditionerTypeKp      &preconditioner_Kp,
            const double                    factor_Kp,
            const PreconditionerTypeMp      &preconditioner_Mp,
            const double                    factor_Mp,
            const bool                      do_solve_A);

    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

    unsigned int n_iterations_A() const;
    unsigned int n_iterations_Kp() const;
    unsigned int n_iterations_Mp() const;

private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<const SparseMatrix<double>>      pressure_mass_matrix;
    const SmartPointer<const SparseMatrix<double>>      pressure_laplace_matrix;

    const PreconditionerTypeA       &preconditioner_A;
    const PreconditionerTypeMp      &preconditioner_Mp;
    const PreconditionerTypeKp      &preconditioner_Kp;

    const double    factor_Kp;
    const double    factor_Mp;


    const bool      do_solve_A;

    mutable unsigned int    n_iterations_A_;
    mutable unsigned int    n_iterations_Kp_;
    mutable unsigned int    n_iterations_Mp_;
};

}  // namespace Preconditioning

#endif /* INCLUDE_PRECONDITIONING_H_ */
