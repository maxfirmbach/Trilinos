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

#include <Xpetra_Map.hpp>
#include <Xpetra_Matrix.hpp>

#include "MueLu_MueluSmoother_decl.hpp"

#include "MueLu_Level.hpp"
#include "MueLu_SpaiSmoother.hpp"
#include "MueLu_MultigridSmoother.hpp"
#include "MueLu_Exceptions.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MueluSmoother(const std::string& type, const Teuchos::ParameterList& paramListIn)
  : type_(type)
  {
    sSpai_ = Teuchos::null;

    ParameterList paramList = paramListIn;

    triedSpai_ = false;
    if(type_=="SPAI")
    {
      sSpai_ = rcp(new SpaiSmoother(type_, paramList));
      triedSpai_ = true;
    }
    else if(type_=="MULTIGRID")
    {
      sMultigrid_ = rcp(new MultigridSmoother(type_, paramList));
      triedMultigrid_ = true;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!triedSpai_, Exceptions::RuntimeError, "Unable to construct any smoother. Please choose SPAI");
    }

    this->SetParameterList(paramList);
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetFactory(const std::string& varName, const RCP<const FactoryBase>& factory) {
    // We need to propagate SetFactory to proper place
    if (!sSpai_.is_null())      sSpai_->SetFactory(varName, factory);
    if (!sMultigrid_.is_null()) sMultigrid_->SetFactory(varName, factory);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const
  {
    if (!sSpai_.is_null())      s_ = sSpai_;
    if (!sMultigrid_.is_null()) s_ = sMultigrid_;

    s_->DeclareInput(currentLevel);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level& currentLevel)
  {
    if (SmootherPrototype::IsSetup() == true)
      this->GetOStream(Warnings0) << "MueLu::MueluSmoother::Setup(): Setup() has already been called" << std::endl;

    int oldRank = s_->SetProcRankVerbose(this->GetProcRankVerbose());

    s_->Setup(currentLevel);

    s_->SetProcRankVerbose(oldRank);

    SmootherPrototype::IsSetup(true);

    this->SetParameterList(s_->GetParameterList());
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(SmootherPrototype::IsSetup() == false, Exceptions::RuntimeError, "MueLu::MueluSmoother::Apply(): Setup() has not been called");

    s_->Apply(X, B, InitialGuessIsZero);
  }


  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
    RCP<MueluSmoother> newSmoo = rcp(new MueluSmoother(type_, this->GetParameterList()));

    // We need to be quite careful with Copy
    // We still want TrilinosSmoother to follow Prototype Pattern, so we need to hide the fact that we do have some state
    if (!sSpai_.is_null())      newSmoo->sSpai_      = sSpai_->Copy();
    if (!sMultigrid_.is_null()) newSmoo->sMultigrid_ = sMultigrid_->Copy();

    // Copy the default mode
    if (s_.get() == sSpai_.get())      newSmoo->s_ = newSmoo->sSpai_;
    if (s_.get() == sMultigrid_.get()) newSmoo->s_ = newSmoo->sMultigrid_;
    newSmoo->SetParameterList(this->GetParameterList());

    return newSmoo;
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
    std::ostringstream out;
    if (s_ != Teuchos::null) {
      out << s_->description();
    } else {
      out << SmootherPrototype::description();
      out << "{type = " << type_ << "}";
    }
    return out.str();
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueluSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream &out, const VerbLevel verbLevel) const {
    MUELU_DESCRIBE;

    if (verbLevel & Parameters0)
      out0 << "Prec. type: " << type_ << std::endl;

    if (verbLevel & Parameters1) {
      out0 << "PrecType: " << type_ << std::endl;
      out0 << "Parameter list: " << std::endl;
      Teuchos::OSTab tab2(out);
      out << this->GetParameterList();
    }

    if (verbLevel & Debug) {
      out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl
           << "-" << std::endl
           << "Epetra PrecType: " << type_ << std::endl
           << "Epetra Parameter list: " << std::endl;
    }
  }

} // namespace MueLu

#endif // MUELU_MUELUSMOOTHER_DEF_HPP
