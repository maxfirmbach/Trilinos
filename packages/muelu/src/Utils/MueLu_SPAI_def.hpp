/*
 * MueLu_SPAI_def.hpp
 *
 *  Created on: Jan 18, 2022
 *      Author: max
 */

#ifndef MUELU_SPAI_DEF_CPP_
#define MUELU_SPAI_DEF_CPP_

#include <queue>

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialQRDenseSolver.hpp>
#include <Teuchos_PtrDecl.hpp>

#include <Xpetra_MultiVector.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_Export.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_Import.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>

// for io operation
#include <Xpetra_IO.hpp>

#include "MueLu_Utilities.hpp"

#include "MueLu_SPAI_decl.hpp"

namespace MueLu {

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  SPAI<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BuildInverse(
    const RCP<Matrix> & A, const RCP<const CrsGraph> & graphAinv)
  {
    A->fillComplete();

    // Construct the inverse matrix with the given sparsity pattern
    RCP<Matrix> Ainv = MatrixFactory::Build(graphAinv);
    Ainv->resumeFill();

    // loop over all rows of the inverse sparsity pattern (this can be done in parallel)
    for(size_t k=0; k<Ainv->getNodeNumRows(); k++) {

      // 1. get sparsity pattern Ik of local row k (done)
      ArrayView<const LO> Ik;
      graphAinv->getLocalRowView(k, Ik);

      // 2. get all local A(Ik,:) rows (done)
      Array<ArrayView<const GO>> J(Ik.size());
      Array<ArrayView<const SC>> Ak(Ik.size());
      Array<GO> Jk;
      for (size_t i = 0; i < Ik.size(); i++) {
        A->getLocalRowView(Ik[i], J[i], Ak[i]); // getGlobalRowView?? // Export/Import
        for (size_t j = 0; j < J[i].size(); j++) Jk.append(J[i][j]);
      }
      // set of unique j indices
      std::sort(Jk.begin(), Jk.end());
      Jk.erase(std::unique(Jk.begin(), Jk.end()), Jk.end());
      // create map
      std::map<GO, GO> G;
      for (size_t i = 0; i < Jk.size(); i++) G.insert(std::pair<GO, GO>(Jk[i], i));

      // 3. merge rows together (done ?)
      Teuchos::SerialDenseMatrix<LO, SC> localA(Jk.size(), Ik.size(), true);
      for (size_t i = 0; i < Ik.size(); i++) {
        for (size_t j = 0; j < J[i].size(); j++) {
          localA(G.at(J[i][j]), i) = Ak[i][j];
        }
      }

      // 4. get direction-vector (done)
      Teuchos::SerialDenseVector<LO, SC> ek(Jk.size(), true); // Jk.size()
      ek[std::find(Ik.begin(), Ik.end(), k) - Ik.begin()] = 1.0;

      // 5. solve linear system for x (done)
      Teuchos::SerialDenseVector<LO, SC> localX(Ik.size());
      Teuchos::SerialQRDenseSolver<LO, SC> qrSolver;
      qrSolver.setMatrix(Teuchos::rcp(&localA, false));
      qrSolver.setVectors(Teuchos::rcp(&localX, false), Teuchos::rcp(&ek, false));
      qrSolver.solve();

      // 6. set calculated row into Ainv (done)
      ArrayView<const SC> Mk(localX.values(), localX.length());
      Ainv->replaceLocalValues(k, Ik, Mk);

    }

    // generate output for checking
    Ainv->fillComplete();
    RCP<Matrix> AinvT = MueLu::Utilities<SC, LO, GO, NO>::Transpose(*Ainv);
    Xpetra::IO<SC, LO, GO, NO>::Write("Ainv", *AinvT);

    // return the inverse operator of A
    return AinvT;
  }

} // namespace MueLu

#endif /* MUELU_SPAI_DEF_CPP_ */
