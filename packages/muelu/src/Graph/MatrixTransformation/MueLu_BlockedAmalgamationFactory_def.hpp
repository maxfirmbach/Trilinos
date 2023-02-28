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
#ifndef MUELU_BLOCKEDAMALGAMATIONFACTORY_DEF_HPP
#define MUELU_BLOCKEDAMALGAMATIONFACTORY_DEF_HPP

#include "MueLu_BlockedAmalgamationFactory.hpp"

#include "MueLu_Level.hpp"
#include "MueLu_AmalgamationInfo.hpp"
#include "MueLu_AmalgamationFactory.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> BlockedAmalgamationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const
  {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set< RCP<const FactoryBase> >("A", Teuchos::null, "Generating factory of the matrix A");
    validParamList->set< RCP<const FactoryBase> >("UnAmalgamationInfo", Teuchos::null, "Generating factory of previous UnAmalgamationInfo. (must be set by user!).");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void BlockedAmalgamationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &currentLevel) const
  {
    this->Input(currentLevel, "A"); // sub-block from blocked A

    // Get AmalgamationInfo from previously defined block
    RCP<const FactoryBase> prevAmalgamationFact = this->GetFactory("UnAmalgamationInfo");
    TEUCHOS_TEST_FOR_EXCEPTION(prevAmalgamationFact==Teuchos::null, Exceptions::RuntimeError, "MueLu::BlockedAmalgamationFactory::BuildUnAmalgamationData: user did not specify UnAmalgamationInfo of previous block. Do not forget to set the Amalgamation factory.");
    currentLevel.DeclareInput("UnAmalgamationInfo", prevAmalgamationFact.get(), this);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void BlockedAmalgamationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level &currentLevel) const
  {
    FactoryMonitor m(*this, "Build", currentLevel);

    RCP<const FactoryBase> prevAmalgamationFact = this->GetFactory("UnAmalgamationInfo");
    RCP<AmalgamationInfo<LO, GO, NO>> subPAmalgamationInfo = currentLevel.Get<RCP<AmalgamationInfo<LO, GO, NO>>>("UnAmalgamationInfo", prevAmalgamationFact.get());

    GO offsetNodeGID = subPAmalgamationInfo->getNodeRowMap()->getMaxAllGlobalIndex()+1;
    this->GetOStream(Runtime1) << "minimalNodeGID: " << offsetNodeGID << "\n";

    AmalgamationFactory::BuildUnAmalgamationData(currentLevel, offsetNodeGID);
  }
} //namespace MueLu

#endif /* MUELU_BLOCKEDAMALGAMATIONFACTORY_DEF_HPP */

