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
#ifndef MUELU_MUELUSMOOTHER_DEF_HPP
#define MUELU_MUELUSMOOTHER_DEF_HPP

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <Kokkos_DefaultNode.hpp>

#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVector.hpp>

#include "MueLu_HierarchyFactory.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_HierarchyManager.hpp"
#include "MueLu_HierarchyUtils.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_FactoryManagerBase.hpp"
#include "MueLu_ParameterListInterpreter.hpp"
#include "MueLu_MueLuSmoother_decl.hpp"

namespace MueLu {

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MueLuSmoother(const Teuchos::ParameterList& paramList)
    {
      this->SetParameterList(paramList);
    }

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetParameterList(const Teuchos::ParameterList& paramList) {
      Factory::SetParameterList(paramList);
    }

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const
    {
      this->Input(currentLevel, "A");
      this->Input(currentLevel, "Nullspace");
    }

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level& currentLevel)
    {
#include "MueLu_UseShortNames.hpp"

      A_         = Factory::Get<RCP<Matrix>>(currentLevel, "A");
      nullspace_ = Factory::Get<RCP<MultiVector>>(currentLevel, "Nullspace"); // TODO: make this work properly

      const ParameterList& pL = this->GetParameterList();
      const std::string xmlFileName = pL.get<std::string>("xml file");
      TEUCHOS_TEST_FOR_EXCEPTION(xmlFileName.empty(), Exceptions::RuntimeError, "MueLu::MueLuSmoother::Setup: xml file name is empty.");

      RCP<ParameterList> mueluParams = Teuchos::rcp(new ParameterList());
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, mueluParams.ptr(), *(A_->getRowMap()->getComm()));

      std::string timerName = "MueLu setup time";
      RCP<Teuchos::Time> tm = Teuchos::TimeMonitor::getNewTimer(timerName);
      tm->start();

      RCP<HierarchyManager> mueLuFactory = rcp(new ParameterListInterpreter(*mueluParams, A_->getDomainMap()->getComm()));

      // Create Hierarchy
      H_ = mueLuFactory->CreateHierarchy();

      H_->setlib(A_->getDomainMap()->lib());
      H_->IsPreconditioner(true);
      H_->GetLevel(0)->Set("A", A_);
      H_->GetLevel(0)->Set("Nullspace", nullspace_);
      H_->SetProcRankVerbose(A_->getDomainMap()->getComm()->getRank());

      mueLuFactory->SetupHierarchy(*H_); // TODO: Here linking error? Seems like not o_o

      tm->stop();
      tm->incrementNumCalls();

      if (H_->GetVerbLevel() & Statistics0) {
        const bool alwaysWriteLocal = true;
        const bool writeGlobalStats = true;
        const bool writeZeroTimers  = false;
        const bool ignoreZeroTimers = true;
        const std::string filter    = timerName;
        Teuchos::TimeMonitor::summarize(A_->getRowMap()->getComm().ptr(), H_->GetOStream(Statistics0), alwaysWriteLocal, writeGlobalStats,
                                        writeZeroTimers, Teuchos::Union, filter, ignoreZeroTimers);
      }

      tm->reset();

    }

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(
            MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const
    {
      H_->Iterate(B, X);
    }

    template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
    RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
      RCP<MueLuSmoother> smoother = rcp(new MueLuSmoother(*this));
      smoother->SetParameterList(this->GetParameterList());

      return Teuchos::rcp_dynamic_cast<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(smoother);
    }

    template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
    std::string MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
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
    void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream &out, const VerbLevel verbLevel) const {
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
    size_t MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNodeSmootherComplexity() const
    {
      // ToDo: Does it make sense to return the operator complexity of the underlying MueLu hierarchy?
      return Teuchos::OrdinalTraits<size_t>::invalid();
    }

} // namespace MueLu

#endif // MUELU_MUELUSMOOTHER_DEF_HPP