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
#ifndef MUELU_SPAISMOOTHER_DEF_HPP
#define MUELU_SPAISMOOTHER_DEF_HPP

#include "MueLu_SpaiSmoother_decl.hpp"
#include "MueLu_ParameterListInterpreter_decl.hpp"

#include "MueLu_Hierarchy.hpp"
#include "MueLu_HierarchyManager.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVector.hpp>

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SpaiSmoother(const std::string& type, const Teuchos::ParameterList& paramList)
    : type_(type)
  {
    SetParameterList(paramList);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetParameterList(const Teuchos::ParameterList& paramList)
  {
    Factory::SetParameterList(paramList);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const
  {
    this->Input(currentLevel, "M");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level& currentLevel)
  {
    FactoryMonitor m(*this, "Setup Smoother", currentLevel);

    M_ = Factory::Get<RCP<Matrix>>(currentLevel, "M");

    const ParameterList& pL = this->GetParameterList();
    sweeps_ = pL.get<int>("spai: sweeps");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(
      MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const
  {
    SC one = Teuchos::ScalarTraits<SC>::one();

    // X = intermediate solution
    // B = residual
    for(int i=0; i<sweeps_; i++)
    {
      // x = x - M*res
      M_->apply(B, X, Teuchos::NO_TRANS, -one, one);
    }
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
    RCP<SpaiSmoother> smoother = rcp(new SpaiSmoother(*this));
    smoother->SetParameterList(this->GetParameterList());
    return Teuchos::rcp_dynamic_cast<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(smoother);
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
    std::ostringstream out;

    if (SmootherPrototype::IsSetup() == true) {
      out << M_->description();

    } else {
      out << SmootherPrototype::description();
      out << "{type = " << type_ << "}";
    }
    return out.str();
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream &out, const VerbLevel verbLevel) const {
    MUELU_DESCRIBE;

    if (verbLevel & Parameters0)
      out0 << "Prec. type: " << type_ << std::endl;

    if (verbLevel & Parameters1) {
      out0 << "Parameter list: " << std::endl;
      Teuchos::OSTab tab2(out);
      out << this->GetParameterList();
    }

    if (verbLevel & External) {
      Teuchos::OSTab tab2(out);
      out << *M_ << std::endl;
    }

    if (verbLevel & Debug)
      out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl
           << "-" << std::endl
           << "RCP<M_>: " << *M_ << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t SpaiSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNodeSmootherComplexity() const
  {
    // ToDo: Does it make sense to return the operator complexity?
    return Teuchos::OrdinalTraits<size_t>::invalid();
  }

} // namespace MueLu

#endif // MUELU_SPAISMOOTHER_DEF_HPP
