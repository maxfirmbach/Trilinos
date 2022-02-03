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

#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_SPAI.hpp"
#include "MueLu_InverseApproximationFactory_decl.hpp"
//#include "MueLu_HierarchyHelpers.hpp"

// for I/O
#include <Xpetra_IO.hpp>

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set<RCP<const FactoryBase> >("A", NoFactory::getRCP(), "Matrix to build the approximate inverse on.\n");

    validParamList->set<std::string>            ("inverse: approximation type",  "Diagonal", "Method used to approximate the inverse.");
    validParamList->set<double>                 ("inverse: absolute threshold", 1e-16, "Treshold for dropping entries inside sparse inverse.");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
    Input(currentLevel, "A");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
    FactoryMonitor m(*this, "Build", currentLevel);

    typedef Teuchos::ScalarTraits<SC> STS;
    SC zero = STS::zero(), one = STS::one();
    const ParameterList& pL = GetParameterList();

    // check which approximation type to use
    std::string method = pL.get<std::string>("inverse: approximation type");

    TEUCHOS_TEST_FOR_EXCEPTION((method=="Diagonal")==false && (method=="Lumping")==false &&
                               (method=="AbsRowSum")==false && (method=="SPAI")==false, Exceptions::RuntimeError,
                               "MueLu::InverseApproximationFactory::Build: Approximation type can be 'Diagonal', 'Lumping', 'AbsRowSum' or 'SPAI'.");

    GetOStream(Statistics1) << "Approximate inverse calculated by " << method << std::endl;

    double threshold = pL.get<double>("inverse: absolute threshold");

    RCP<Matrix> A = Get<RCP<Matrix> >(currentLevel, "A");
    RCP<Matrix> M;

    if(method=="Diagonal")
    {
      RCP<Vector> diag;
      diag = VectorFactory::Build(A->getRangeMap(), true);
      A->getLocalDiagCopy(*diag);
      const RCP<const Vector> D = Utilities::GetInverse(diag);
      M = MatrixFactory::Build(D);
    }
    else if(method=="Lumping")
    {
      RCP<Vector> diag;
      diag = Utilities::GetLumpedMatrixDiagonal(*A);
      const RCP<const Vector> D = Utilities::GetInverse(diag);
      M = MatrixFactory::Build(D);
    }
    else if(method=="AbsRowSum")
    {
      // Not implemented yet!
      TEUCHOS_TEST_FOR_EXCEPTION((method=="AbsRowSum")==true,
                                 MueLu::Exceptions::RuntimeError,"MueLu::SchurComplementFactory::Build: AbsRowSum approximation not implemented yet!");
    }
    else if(method=="SPAI")
    {
      // Implementation ongoing!
      RCP<SPAI> spai = Teuchos::rcp(new SPAI());

      RCP<Matrix> AA = MatrixFactory::Build(A->getRowMap(), Teuchos::as<LO>(0));
      MatrixMatrix::Multiply(*A, false, *A, false, *AA);

      RCP<Matrix> AAA = MatrixFactory::Build(A->getRowMap(), Teuchos::as<LO>(0));
      MatrixMatrix::Multiply(*AA, false, *AA, false, *AAA);

      RCP<const CrsGraph> pattern = AAA->getCrsGraph();

      // actually we want to return the inverse here ... otherwise we got everything we want argumentwise
      // patternwise different things are possible, given above is just the same pattern as input matrix???
      M = spai->BuildInverse(A, pattern);
    }

    GetOStream(Statistics1) << "M has " << M->getGlobalNumRows() << "x" << M->getGlobalNumCols() << " rows and columns." << std::endl;

    // NOTE: "M" generated by this factory is actually the approximate inverse matrix
    Set(currentLevel, "M", M);

  }
} // namespace MueLu

#endif /* MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_ */
