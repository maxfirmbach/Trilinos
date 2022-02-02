/*
 * MueLu_SPAI_decl.hpp
 *
 *  Created on: Jan 18, 2022
 *      Author: max
 */

#ifndef MUELU_SPAI_DECL_HPP_
#define MUELU_SPAI_DECL_HPP_

#include <Xpetra_MultiVector_fwd.hpp>
#include <Xpetra_Matrix_fwd.hpp>
#include <Xpetra_CrsGraph_fwd.hpp>
#include <Xpetra_Vector_fwd.hpp>
#include <Xpetra_MultiVectorFactory_fwd.hpp>
#include <Xpetra_VectorFactory_fwd.hpp>
#include <Xpetra_CrsMatrixWrap_fwd.hpp>
#include <Xpetra_Export_fwd.hpp>
#include <Xpetra_ExportFactory_fwd.hpp>
#include <Xpetra_Import_fwd.hpp>
#include <Xpetra_ImportFactory_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_BaseClass.hpp"

namespace MueLu {

  template <class Scalar = DefaultScalar,
            class LocalOrdinal = DefaultLocalOrdinal,
            class GlobalOrdinal = DefaultGlobalOrdinal,
            class Node = DefaultNode>
  class SPAI : public BaseClass {

#undef MUELU_SPAI_SHORT
#include "MueLu_UseShortNames.hpp"

    public:


     //! build sparse approximate inverse
     /*
      * \param A: input matrix (input)
      * \param sparsityPattern: sparsity pattern of the inverse (input)
      * \param Ainv: sparse approximate inverse of A with given sparsity pattern (output)
      */
     RCP<Matrix> BuildInverse(const RCP<Matrix> & A, const RCP<const CrsGraph> & sparsityPattern);

    private:


  };

} // namespace MueLu

#define MUELU_SPAI_SHORT

#endif /* MUELU_SPAI_DECL_HPP_ */
