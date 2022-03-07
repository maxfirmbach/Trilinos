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
#ifndef MUELU_SCHURCOMPLEMENTFACTORY_DEF_HPP_
#define MUELU_SCHURCOMPLEMENTFACTORY_DEF_HPP_

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

#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_SchurComplementFactory.hpp"
//#include "MueLu_HierarchyHelpers.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> SchurComplementFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    SC one = Teuchos::ScalarTraits<SC>::one();

    validParamList->set<RCP<const FactoryBase> >("A", NoFactory::getRCP(), "Generating factory of the matrix A used for building Schur complement\n"
                                                                                           "(must be a 2x2 blocked operator)");
    validParamList->set<RCP<const FactoryBase> >("M", NoFactory::getRCP(), "Generating factory of the inverse matrix used in the Schur complement");

    validParamList->set<int>                    ("block row",                           1, "Block row of subblock matrix A");
    validParamList->set<int>                    ("block col",                           1, "Block column of subblock matrix A");

    validParamList->set<SC>                     ("schur: damping factor",             one, "Scaling parameter in S = A(1,1) - 1/omega A(1,0) M^{-1} A(0,1)");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void SchurComplementFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
    Input(currentLevel, "A");
    Input(currentLevel, "M");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void SchurComplementFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
    FactoryMonitor m(*this, "Build", currentLevel);

    typedef Teuchos::ScalarTraits<SC> STS;
    SC zero = STS::zero(), one = STS::one();

    RCP<Matrix>            A = Get<RCP<Matrix> >(currentLevel, "A");
    RCP<Matrix>            M = Get<RCP<Matrix> >(currentLevel, "M");
    RCP<BlockedCrsMatrix> bA = rcp_dynamic_cast<BlockedCrsMatrix>(A);
    TEUCHOS_TEST_FOR_EXCEPTION(bA.is_null(), Exceptions::BadCast,
                               "MueLu::SchurComplementFactory::Build: input matrix A is not of type BlockedCrsMatrix!");

    TEUCHOS_TEST_FOR_EXCEPTION(bA->Rows() != 2 || bA->Cols() != 2, Exceptions::RuntimeError,
                               "MueLu::SchurComplementFactory::Build: input matrix A is a " << bA->Rows() << "x" << bA->Cols() << " block matrix. We expect a 2x2 blocked operator.");

    const ParameterList& pL = GetParameterList();

    // Check on which block the Schur complement should be calculated  default is S(1,1)
    auto row = Teuchos::as<size_t>(pL.get<int>("block row"));
    auto col = Teuchos::as<size_t>(pL.get<int>("block col"));

    RCP<Matrix> A01, A10, A11;
    if(row == 0 && col == 0)
    {
      A01 = bA->getMatrix(1,0);
      A10 = bA->getMatrix(0,1);
      A11 = bA->getMatrix(0,0); 
    }
    else
    {
      A01 = bA->getMatrix(0,1);
      A10 = bA->getMatrix(1,0);
      A11 = bA->getMatrix(1,1);
    }

    SC omega = pL.get<Scalar>("schur: damping factor");

    TEUCHOS_TEST_FOR_EXCEPTION(omega == zero, Exceptions::RuntimeError,
                               "MueLu::SchurComplementFactory::Build: Scaling parameter omega must not be zero to avoid division by zero.");

    RCP<Matrix> A10MA01 = Teuchos::null;
    RCP<Matrix> S = Teuchos::null;
    // only if the off-diagonal blocks A10 and A01 are non-zero we have to do the MM multiplication
    if(A01.is_null() == false && A10.is_null() == false) {

      // scale with -1/omega
      M->scale(Teuchos::as<Scalar>(-one/omega));

      // -1/omega*M*A_01
      RCP<Matrix> MA01 = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*M, false, *A01, false, GetOStream(Statistics2), true, true, std::string("SchurComplementFactory"));

      // -1/omega*A_10*M*A_01;
      RCP<Matrix> A10MA01 = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*A10, false, *MA01, false, GetOStream(Statistics2), true, true, std::string("SchurComplementFactory"));

      if (!A11.is_null()) {
        Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::TwoMatrixAdd(*A11, false, one, *A10MA01, false, one, S, GetOStream(Statistics2));
        S->fillComplete();

        TEUCHOS_TEST_FOR_EXCEPTION(A11->getRangeMap()->isSameAs(*(S->getRangeMap())) == false, Exceptions::RuntimeError,
                                   "MueLu::SchurComplementFactory::Build: RangeMap of A11 and S are not the same.");
        TEUCHOS_TEST_FOR_EXCEPTION(A11->getDomainMap()->isSameAs(*(S->getDomainMap())) == false, Exceptions::RuntimeError,
                                   "MueLu::SchurComplementFactory::Build: DomainMap of A11 and S are not the same.");
      }
      else {
        S = A10MA01;
        S->fillComplete();
      }
    }
    else {
      if (!A11.is_null()) {
        S = MatrixFactory::BuildCopy(A11);
      } else {
        S = MatrixFactory::Build(A11->getRowMap(), 10 /*A11->getLocalMaxNumRowEntries()*/);
        S->fillComplete(A11->getDomainMap(),A11->getRangeMap());
      }
    }

    GetOStream(Statistics1) << "S(" << row << "," << col << ") has " << S->getGlobalNumRows() << "x" 
                            << S->getGlobalNumCols() << " rows and columns." << std::endl;

    // NOTE: "A" generated by this factory is actually the Schur complement
    // matrix, but it is required as all smoothers expect "A"
    Set(currentLevel, "A", S);

  }
} // namespace MueLu

#endif /* MUELU_SCHURCOMPLEMENTFACTORY_DEF_HPP_ */
