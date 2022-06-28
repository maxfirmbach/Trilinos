// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_
#define MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_TripleMatrixMultiply.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>

#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialQRDenseSolver.hpp>

#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_InverseApproximationFactory_decl.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set<RCP<const FactoryBase> >("A", NoFactory::getRCP(), "Matrix to build the approximate inverse on.\n");

    validParamList->set<std::string>            ("inverse: approximation type",  "diagonal", "Method used to approximate the inverse.");
    validParamList->set<bool>                   ("inverse: fixing",              false     , "Fix diagonal by replacing small entries with 1.0");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
    Input(currentLevel, "A");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
    FactoryMonitor m(*this, "Build", currentLevel);

    using STS = Teuchos::ScalarTraits<SC>;
    const SC one = STS::one();

    const ParameterList& pL = GetParameterList();
    const bool fixing = pL.get<bool>("inverse: fixing");

    // check which approximation type to use
    const std::string method = pL.get<std::string>("inverse: approximation type");
    TEUCHOS_TEST_FOR_EXCEPTION(method != "diagonal" && method != "lumping" && method != "spai", Exceptions::RuntimeError,
                               "MueLu::InverseApproximationFactory::Build: Approximation type can be 'diagonal' or 'lumping'.");

    RCP<Matrix> A = Get<RCP<Matrix> >(currentLevel, "A");
    RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A);
    const bool isBlocked = (bA == Teuchos::null ? false : true);

    // if blocked operator is used, defaults to A(0,0)
    if(isBlocked) A = bA->getMatrix(0,0);

    RCP<Matrix> Ainv = Teuchos::null;
    if(method=="diagonal") {
      const auto diag = VectorFactory::Build(A->getRangeMap(), true);
      A->getLocalDiagCopy(*diag);
      const RCP<const Vector> D = (!fixing ? Utilities::GetInverse(diag) : Utilities::GetInverse(diag, 1e-4, one));
      Ainv = MatrixFactory::Build(D);
    }
    else if(method=="lumping") {
      const auto diag = Utilities::GetLumpedMatrixDiagonal(*A);
      const RCP<const Vector> D = (!fixing ? Utilities::GetInverse(diag) : Utilities::GetInverse(diag, 1e-4, one));
      Ainv = MatrixFactory::Build(D);
    }
    else if(method=="spai") {
      Ainv = GetSparseInverse(A, A->getCrsGraph());
    }

    GetOStream(Statistics1) << "Approximate inverse calculated by: " << method << "." << std::endl;
    GetOStream(Statistics1) << "Ainv has " << Ainv->getGlobalNumRows() << "x" << Ainv->getGlobalNumCols() << " rows and columns." << std::endl;

    Set(currentLevel, "Ainv", Ainv);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetSparseInverse(const RCP<Matrix>& Aorg, const RCP<const CrsGraph>& sparsityPattern) const {

    // construct the inverse matrix with the given sparsity pattern
    RCP<Matrix> Ainv = MatrixFactory::Build(sparsityPattern);
    Ainv->resumeFill();

    // gather missing rows from other procs to generate an overlapping map
    RCP<Import> rowImport = ImportFactory::Build(sparsityPattern->getRowMap(), sparsityPattern->getColMap());
    RCP<Matrix> A = MatrixFactory::Build(Aorg, *rowImport);

    // loop over all rows of the inverse sparsity pattern (this can be done in parallel)
    for(size_t k=0; k<sparsityPattern->getLocalNumRows(); k++) {

      // 1. get column indices Ik of local row k
      ArrayView<const LO> Ik;
      sparsityPattern->getLocalRowView(k, Ik);

      // 2. get all local A(Ik,:) rows
      Array<ArrayView<const GO>> J(Ik.size());
      Array<ArrayView<const SC>> Ak(Ik.size());
      Array<GO> Jk;
      for (size_t i = 0; i < Ik.size(); i++) {
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

      // 3. merge rows together
      Teuchos::SerialDenseMatrix<LO, SC> localA(Jk.size(), Ik.size(), true);
      for (size_t i = 0; i < Ik.size(); i++) {
        for (size_t j = 0; j < J[i].size(); j++) {
          localA(G.at(J[i][j]), i) = Ak[i][j];
        }
      }

      // 4. get direction-vector
      // diagonal needs an entry!
      Teuchos::SerialDenseVector<LO, SC> ek(Jk.size(), true);
      ek[std::find(Jk.begin(), Jk.end(), k) - Jk.begin()] = 1.0;

      // 5. solve linear system for x
      Teuchos::SerialDenseVector<LO, SC> localX(Ik.size());
      Teuchos::SerialQRDenseSolver<LO, SC> qrSolver;
      qrSolver.setMatrix(Teuchos::rcp(&localA, false));
      qrSolver.setVectors(Teuchos::rcp(&localX, false), Teuchos::rcp(&ek, false));
      qrSolver.solve();

      // 6. set calculated row into Ainv
      ArrayView<const SC> Mk(localX.values(), localX.length());
      Ainv->replaceLocalValues(k, Ik, Mk);

    }
    Ainv->fillComplete();

    return Ainv;
  }

} // namespace MueLu

#endif /* MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_ */
