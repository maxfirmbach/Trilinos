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

#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

#include <Xpetra_MapFactory.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_Export.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Import.hpp>

#include "MueLu_Utilities.hpp"
#include "MueLu_Spai_decl.hpp"

namespace MueLu {

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
  Spai<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BuildInverse(
    const RCP<Matrix> & A, const RCP<const CrsGraph> & graphAinv)
  {
    A->fillComplete();

    // Construct the inverse matrix with the given sparsity pattern
    RCP<Matrix> Ainv = MatrixFactory::Build(graphAinv);
    Ainv->resumeFill();

    // loop over all rows of the inverse sparsity pattern (this can be done in parallel)
    for(size_t k=0; k<Ainv->getLocalNumRows(); k++) {

      // 1. get sparsity pattern Ik of local row k (done)
      ArrayView<const LO> Ik;
      graphAinv->getLocalRowView(k, Ik);

      // 2. gather missing rows
      /*
      RCP<Map> map = MapFactory::Build(A->getRowMap()->lib(), -1, Ik.size(), 0, A->getRowMap()->getComm());
      RCP<Matrix> AIk = MatrixFactory::Build(map, A->getGlobalMaxNumRowEntries());
      RCP<Import> scatter = ImportFactory::Build(A->getRowMap(), map);
      AIk->doImport(*A, *scatter, Xpetra::INSERT);
      */
 
      // 3. get all local A(Ik,:) rows (done)
      Array<ArrayView<const GO>> J(Ik.size());
      Array<ArrayView<const SC>> Ak(Ik.size());
      Array<GO> Jk;
      for (size_t i = 0; i < Ik.size(); i++) {
        //AIk->getLocalRowView(Ik[i], J[i], Ak[i]);
        A->getLocalRowView(Ik[i], J[i], Ak[i]);
        for (size_t j = 0; j < J[i].size(); j++)
          Jk.append(J[i][j]);
      }
      // set of unique column indices Jk
      std::sort(Jk.begin(), Jk.end());
      Jk.erase(std::unique(Jk.begin(), Jk.end()), Jk.end());
      // create map
      std::map<GO, GO> G;
      for (size_t i = 0; i < Jk.size(); i++) G.insert(std::pair<GO, GO>(Jk[i], i));

      // 4. merge rows together (done)
      Teuchos::SerialDenseMatrix<LO, SC> localA(Jk.size(), Ik.size(), true);
      for (size_t i = 0; i < Ik.size(); i++) {
        for (size_t j = 0; j < J[i].size(); j++) {
          localA(G.at(J[i][j]), i) = Ak[i][j];
        }
      }

      // 5. get direction-vector (done)
      Teuchos::SerialDenseVector<LO, SC> ek(Jk.size(), true);
      ek[std::find(Jk.begin(), Jk.end(), k) - Jk.begin()] = 1.0;

      // 6. solve linear system for x (done)
      Teuchos::SerialDenseVector<LO, SC> localX(Ik.size());
      Teuchos::SerialQRDenseSolver<LO, SC> qrSolver;
      qrSolver.setMatrix(Teuchos::rcp(&localA, false));
      qrSolver.setVectors(Teuchos::rcp(&localX, false), Teuchos::rcp(&ek, false));
      qrSolver.solve();

      // 7. set calculated row into Ainv (done)
      ArrayView<const SC> Mk(localX.values(), localX.length());
      Ainv->replaceLocalValues(k, Ik, Mk);

    }

    // generate output for checking
    Ainv->fillComplete();
    RCP<Matrix> AinvT = MueLu::Utilities<SC, LO, GO, NO>::Transpose(*Ainv);

    // return the inverse operator of A
    return AinvT;
  }

} // namespace MueLu

#endif /* MUELU_SPAI_DEF_CPP_ */
