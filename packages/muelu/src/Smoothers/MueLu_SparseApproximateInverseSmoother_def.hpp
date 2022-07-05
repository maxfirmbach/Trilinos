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
#ifndef MUELU_SPARSEAPPROXIMATEINVERSESMOOTHER_DEF_HPP
#define MUELU_SPARSEAPPROXIMATEINVERSESMOOTHER_DEF_HPP

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include "MueLu_ConfigDefs.hpp"

#include "MueLu_SparseApproximateInverseSmoother_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SparseApproximateInverseSmoother(const Teuchos::ParameterList & paramList)
    : A_(Teuchos::null), Ainv_(Teuchos::null)
  {
    this->SetParameterList(paramList);
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set< RCP<const FactoryBase> >("A",    null, "Factory of the coarse matrix");
    validParamList->set< RCP<const FactoryBase> >("Ainv", null, "Factory of the approximate inverse matrix");

    validParamList->set<int>   ("relaxation: sweeps",           1, "Number of smoother sweeps.");
    validParamList->set<double>("relaxation: damping factor", 1.0, "Damping factor of the smoother.");

    return validParamList;
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &currentLevel) const {
    this->Input(currentLevel, "A");
    this->Input(currentLevel, "Ainv");
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level &currentLevel) {
    FactoryMonitor monitor(*this, "Setup Sparse Approximate Inverse Smoother", currentLevel);

    if (SmootherPrototype::IsSetup() == true)
      this->GetOStream(Warnings0) << "MueLu::ProjectorSmoother::Setup(): Setup() has already been called" << std::endl;

    A_    = Factory::Get< RCP<Matrix> >(currentLevel, "A");
    Ainv_ = Factory::Get< RCP<Matrix> >(currentLevel, "Ainv");

    ParameterList pL = this->GetParameterList();
    sweeps_ = pL.get<int>("relaxation: sweeps");
    omega_ = pL.get<double>("relaxation: damping factor");

    SmootherPrototype::IsSetup(true);
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const {
    SC zero = Teuchos::ScalarTraits<SC>::zero(), one = Teuchos::ScalarTraits<SC>::one();

    for (LO run = 0; run < sweeps_; ++run) {

      // 1) calculate current residual
      RCP<MultiVector> residual = MultiVectorFactory::Build(B.getMap(), B.getNumVectors());
      residual->update(one, B, zero); // residual = B
      A_->apply(X, *residual, Teuchos::NO_TRANS, -one, one);

      // 2) apply inverse operator
      Ainv_->apply(*residual, X, Teuchos::NO_TRANS, -one, one);

    }
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node> > SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
    return rcp( new SparseApproximateInverseSmoother(*this) );
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
    std::ostringstream out;
    out << SmootherPrototype::description();
    return out.str();
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void SparseApproximateInverseSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream &out, const VerbLevel verbLevel) const {
    MUELU_DESCRIBE;
    out0 << "";
  }

} // namespace MueLu

#endif // MUELU_SPARSEAPPROXIMATEINVERSESMOOTHER_DEF_HPP
