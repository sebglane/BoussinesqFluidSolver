/*
 * linear_algebra.h
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */

#ifndef INCLUDE_ADSOLIC_LINEAR_ALGEBRA_H_
#define INCLUDE_ADSOLIC_LINEAR_ALGEBRA_H_

#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

namespace adsolic {

using namespace dealii;

/**
 * A namespace that contains typedefs for classes used in the linear algebra
 * description.
 */
namespace LA
{
    /**
     * Typedef for the vector type used.
     */
    typedef TrilinosWrappers::MPI::Vector Vector;

    /**
     * Typedef for the type used to describe vectors that consist of multiple
     * blocks.
     */
    typedef TrilinosWrappers::MPI::BlockVector BlockVector;

    /**
     * Typedef for the sparse matrix type used.
     */
    typedef TrilinosWrappers::SparseMatrix SparseMatrix;

    /**
     * Typedef for the type used to describe sparse matrices that consist of
     * multiple blocks.
     */
    typedef TrilinosWrappers::BlockSparseMatrix BlockSparseMatrix;

    /**
     * Typedef for the base class for all preconditioners.
     */
    typedef TrilinosWrappers::PreconditionBase PreconditionBase;

    /**
     * Typedef for the AMG preconditioner type used for the top left block of
     * the Stokes matrix.
     */
    typedef TrilinosWrappers::PreconditionAMG PreconditionAMG;

    /**
     * Typedef for the Incomplete Cholesky preconditioner used for other
     * blocks of the system matrix.
     */
    typedef TrilinosWrappers::PreconditionIC PreconditionIC;

    /**
     * Typedef for the Incomplete LU decomposition preconditioner used for
     * other blocks of the system matrix.
     */
    typedef TrilinosWrappers::PreconditionILU PreconditionILU;

    /**
     * Typedef for the Jacobi preconditioner.
     */
    typedef TrilinosWrappers::PreconditionJacobi PreconditionJacobi;

    /**
     * Typedef for the SSOR preconditioner.
     */
    typedef TrilinosWrappers::PreconditionBlockSSOR PreconditionSSOR;

    /**
     * Typedef for the SSOR preconditioner.
     */
    typedef TrilinosWrappers::PreconditionSOR PreconditionSOR;


    /**
     * Typedef for the block compressed sparsity pattern type.
     */
    typedef TrilinosWrappers::BlockSparsityPattern BlockDynamicSparsityPattern;

    /**
     * Typedef for the compressed sparsity pattern type.
     */
    typedef TrilinosWrappers::SparsityPattern DynamicSparsityPattern;

    /**
     * Typedef for conjugate gradient linear solver.
     */
    typedef TrilinosWrappers::SolverCG  SolverCG;

    /**
     * Typedef for gmres method linear solver.
     */
    typedef TrilinosWrappers::SolverGMRES  SolverGMRES;

}

/**
 * A structure containing parameters relevant for the solution of linear systems.
 */
struct  LinearSolverParameters
{
    LinearSolverParameters();
    LinearSolverParameters(const std::string &parameter_filename);

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    /*
     * function forwarding parameters to a stream object
     */
    template<typename Stream>
    void write(Stream &stream) const;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;
    unsigned int    n_max_iter;
};

}  // namespace adsolic

#endif /* INCLUDE_ADSOLIC_LINEAR_ALGEBRA_H_ */
