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
#ifndef MUELU_MULTIGRIDSMOOTHER_DEF_HPP
#define MUELU_MULTIGRIDSMOOTHER_DEF_HPP

#include "MueLu_MultigridSmoother_decl.hpp"
#include "MueLu_ParameterListInterpreter_decl.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_HierarchyManager.hpp"
#include "MueLu_Level.hpp"
// #include "MueLu_ParameterListInterpreter.hpp"
// #include "MueLu_CreateXpetraPreconditioner.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVector.hpp>

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MultigridSmoother(const std::string type, const Teuchos::ParameterList& paramList)
   : type_(type)
  {
    // right now force the user to use "MULTIGRID", if a muelu smoother should be used
    bool isSupported = false;
    isSupported = type_ == "MULTIGRID";
    this->declareConstructionOutcome(!isSupported, "MueLu does not provide the smoother '" + type_ + "'.");
    if (isSupported)
      SetParameterList(paramList); 
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetParameterList(const Teuchos::ParameterList& paramList) {
    Factory::SetParameterList(paramList);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const
  {
    this->Input(currentLevel, "A");
    this->Input(currentLevel, "Nullspace");

  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level& currentLevel)
  {
#include "MueLu_UseShortNames.hpp"

    A_         = Factory::Get<RCP<Matrix>>(currentLevel, "A");
    Nullspace_ = Factory::Get<RCP<MultiVector>>(currentLevel, "Nullspace");

    const ParameterList& pL = this->GetParameterList();
    std::string xmlFile = pL.get<std::string>("muelu: xml file"); //TODO exit if xmlFile.null() ?!
 
    ParameterListInterpreter mueLuFactory(xmlFile, *(A_->getDomainMap()->getComm()));

    const std::string label = "MueLu Smoother";
    H_ = mueLuFactory.CreateHierarchy(label);

    H_->setlib(A_->getDomainMap()->lib());
    H_->GetLevel(0)->Set("A", A_);
    H_->GetLevel(0)->Set("Nullspace", Nullspace_);
    H_->SetProcRankVerbose(A_->getDomainMap()->getComm()->getRank());

    mueLuFactory.SetupHierarchy(*H_);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(
      MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const
  {
    //H_->IsPreconditioner(false);
    H_->Iterate(B, X); 
  }


  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
    RCP<MultigridSmoother> smoother = rcp(new MultigridSmoother(*this));
    smoother->SetParameterList(this->GetParameterList());
    return Teuchos::rcp_dynamic_cast<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(smoother);
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
    std::ostringstream out;

    if (SmootherPrototype::IsSetup() == true) {
      out << H_->description();

    } else {
      out << SmootherPrototype::description();
      out << "MUELU {type = " << type_ << "}";
    }
    return out.str();
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream &out, const VerbLevel verbLevel) const {
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
      out << *H_ << std::endl;
    }

    if (verbLevel & Debug)
      out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl
           << "-" << std::endl
           << "RCP<hierarchy_>: " << H_ << std::endl;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t MultigridSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNodeSmootherComplexity() const
  {
    // ToDo: Does it make sense to return the operator complexity of the underlying MueLu hierarchy?
    return Teuchos::OrdinalTraits<size_t>::invalid();
  }

} // namespace MueLu

#endif // MUELU_MUELUSMOOTHER_DEF_HPP
