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
//                    Tobias Wiesner    (tawiesn@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef PACKAGES_MUELU_SRC_MISC_MUELU_FINELEVELINPUTDATAFACTORY_DEF_HPP_
#define PACKAGES_MUELU_SRC_MISC_MUELU_FINELEVELINPUTDATAFACTORY_DEF_HPP_

#include "Xpetra_Matrix.hpp"

#include "MueLu_FineLevelInputDataFactory_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> FineLevelInputDataFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());

  // Variable name (e.g. A or P or Coordinates)
  validParamList->set<std::string>("Fine level variable", std::string("A"), "Variable name on finest level.");
  validParamList->set<std::string>("Variable", std::string("A"), "Variable name on all coarse levels (except the finest level).");

  // Names of generating factories (on finest level and coarse levels)
  validParamList->set<RCP<const FactoryBase> >("Fine level factory", Teuchos::null, "Generating factory of the fine level variable");
  validParamList->set<RCP<const FactoryBase> >("Coarse level factory", Teuchos::null, "Generating factory for data on all coarse levels (except the finest)");

  // Type of variable (see source code for a complete list of all available types)
  validParamList->set<std::string>("Variable type", std::string("Matrix"), "Type of variable");

  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FineLevelInputDataFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  const ParameterList& pL = GetParameterList();

  std::string variableName = "";
  if (currentLevel.GetLevelID() == 0 && pL.isParameter("Fine level variable")) {
      variableName = pL.get<std::string>("Fine level variable");
  } else if (pL.isParameter("Variable")) {
      variableName = pL.get<std::string>("Variable");
  }

  std::string factoryName = "NoFactory";
  if (currentLevel.GetLevelID() == 0) {
    factoryName = "Fine level factory";
  } else {
    factoryName = "Coarse level factory";
  }

  TEUCHOS_TEST_FOR_EXCEPTION(variableName == "", MueLu::Exceptions::RuntimeError, "FineLevelInputDataFactory: no variable name provided. Please set \'Variable\' parameter in your input deck.");

  // data must be specified in factory! (not in factory manager)
  RCP<const FactoryBase> fact = GetFactory(factoryName);
  currentLevel.DeclareInput(variableName, fact.get(), this);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FineLevelInputDataFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
  FactoryMonitor m(*this, "InputUserData", currentLevel);

  const ParameterList& pL = GetParameterList();

  std::string finestVariableName = "";
  if (currentLevel.GetLevelID() == 0 && pL.isParameter("Fine level variable")) {
    finestVariableName = pL.get<std::string>("Fine level variable");
  }

  std::string variableName = "";
  if (pL.isParameter("Variable"))
    variableName = pL.get<std::string>("Variable");

  std::string variableType = "";
  if (pL.isParameter("Variable type"))
    variableType = pL.get<std::string>("Variable type");

  std::string factoryName = "NoFactory";
  if (currentLevel.GetLevelID() == 0) {
    factoryName = "Fine level factory";
  } else {
    factoryName = "Coarse level factory";
  }
  RCP<const FactoryBase> fact = GetFactory(factoryName);

  if (currentLevel.GetLevelID() == 0)
    GetOStream(Runtime1) << "Use " << finestVariableName << " of type " << variableType << " from user input." << std::endl;
  else
    GetOStream(Runtime1) << "Use " << variableName << " of type " << variableType << " from " << factoryName << "(" << fact.get() << ")" << std::endl;

  // check data type
  // std::string strType = currentLevel.GetTypeName(variableName, fact.get());
  if (variableType == "int") {
    int data;
    if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
      data = currentLevel.Get<int>(finestVariableName, NoFactory::get());
    } else {
      data = currentLevel.Get<int>(variableName, fact.get());
    }
    Set(currentLevel, variableName, data);
  } else if (variableType == "double") {
    double data;
    if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
      data = currentLevel.Get<double>(finestVariableName, NoFactory::get());
    } else {
      data = currentLevel.Get<double>(variableName, fact.get());
    }
    Set(currentLevel, variableName, data);
  } else if (variableType == "string") {
    std::string data;
    if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
      std::string data = currentLevel.Get<std::string>(finestVariableName, NoFactory::get());
    } else {
      std::string data = currentLevel.Get<std::string>(variableName, fact.get());
    }
    Set(currentLevel, variableName, data);
  } else {
    size_t npos = std::string::npos;

    if (variableType.find("Aggregates") != npos) {
      RCP<Aggregates> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<Aggregates> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<Aggregates> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("SmootherBase") != npos) {
      RCP<SmootherBase> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<SmootherBase> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<SmootherBase> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("SmootherPrototype") != npos) {
      RCP<SmootherPrototype> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<SmootherPrototype> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<SmootherPrototype> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("Export") != npos) {
      RCP<Export> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<Export> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<Export> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("Import") != npos) {
      RCP<Import> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<Import> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<Import> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("Map") != npos) {
      RCP<Map> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<Map> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<Map> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("Matrix") != npos) {
      RCP<Matrix> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<Matrix> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<Matrix> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("MultiVector") != npos) {
      RCP<MultiVector> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<MultiVector> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<MultiVector> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else if (variableType.find("Operator") != npos) {
      RCP<Operator> data;
      if (currentLevel.GetLevelID() == 0 && !finestVariableName.empty()) {
        data = currentLevel.Get<RCP<Operator> >(finestVariableName, NoFactory::get());
      } else {
        data = currentLevel.Get<RCP<Operator> >(variableName, fact.get());
      }
      Set(currentLevel, variableName, data);
    } else {
      // TAW: is this working with empty procs?
      TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::RuntimeError, "FineLevelInputDataFactory: cannot detect type of variable " << variableName << " generated by " << fact.get() << ". User provided type " << variableType);
    }
  }
}

}  // namespace MueLu

#endif /* PACKAGES_MUELU_SRC_MISC_MUELU_FINELEVELINPUTDATAFACTORY_DEF_HPP_ */
