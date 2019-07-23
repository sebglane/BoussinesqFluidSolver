/*
 * linear_algebra.h
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_LINEAR_ALGEBRA_H_
#define INCLUDE_ADSOLIC_LINEAR_ALGEBRA_H_

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

/**
 * A namespace that contains typedefs for classes used in the linear algebra
 * description.
 */
namespace LA
{
    /**
     * Typedef for the vector type used.
     */
    typedef dealii::TrilinosWrappers::MPI::Vector Vector;

    /**
     * Typedef for the type used to describe vectors that consist of multiple
     * blocks.
     */
    typedef dealii::TrilinosWrappers::MPI::BlockVector BlockVector;

    /**
     * Typedef for the sparse matrix type used.
     */
    typedef dealii::TrilinosWrappers::SparseMatrix SparseMatrix;

    /**
     * Typedef for the type used to describe sparse matrices that consist of
     * multiple blocks.
     */
    typedef dealii::TrilinosWrappers::BlockSparseMatrix BlockSparseMatrix;

    /**
     * Typedef for the base class for all preconditioners.
     */
    typedef dealii::TrilinosWrappers::PreconditionBase PreconditionBase;

    /**
     * Typedef for the AMG preconditioner type used for the top left block of
     * the Stokes matrix.
     */
    typedef dealii::TrilinosWrappers::PreconditionAMG PreconditionAMG;

    /**
     * Typedef for the Incomplete Cholesky preconditioner used for other
     * blocks of the system matrix.
     */
    typedef dealii::TrilinosWrappers::PreconditionIC PreconditionIC;

    /**
     * Typedef for the Incomplete LU decomposition preconditioner used for
     * other blocks of the system matrix.
     */
    typedef dealii::TrilinosWrappers::PreconditionILU PreconditionILU;

    /**
     * Typedef for the Jacobi preconditioner.
     */
    typedef dealii::TrilinosWrappers::PreconditionJacobi PreconditionJacobi;

    /**
     * Typedef for the SSOR preconditioner.
     */
    typedef dealii::TrilinosWrappers::PreconditionBlockSSOR PreconditionSSOR;

    /**
     * Typedef for the SSOR preconditioner.
     */
    typedef dealii::TrilinosWrappers::PreconditionSOR PreconditionSOR;


    /**
     * Typedef for the block compressed sparsity pattern type.
     */
    typedef dealii::TrilinosWrappers::BlockSparsityPattern BlockDynamicSparsityPattern;

    /**
     * Typedef for the compressed sparsity pattern type.
     */
    typedef dealii::TrilinosWrappers::SparsityPattern DynamicSparsityPattern;

    /**
     * Typedef for conjugate gradient linear solver.
     */
    typedef dealii::TrilinosWrappers::SolverCG  SolverCG;

    /**
     * Typedef for gmres method linear solver.
     */
    typedef dealii::TrilinosWrappers::SolverGMRES  SolverGMRES;

}

#endif /* INCLUDE_ADSOLIC_LINEAR_ALGEBRA_H_ */
