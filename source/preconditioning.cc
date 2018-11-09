/*
 * preconditioning.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */


#include "preconditioning.templates.h"

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/precondition.h>

namespace Preconditioning {

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

}  // namespace


