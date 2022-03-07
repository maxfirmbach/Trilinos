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
#ifndef MUELU_MUELUSMOOTHER_DECL_HPP
#define MUELU_MUELUSMOOTHER_DECL_HPP

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_Matrix_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"
#include "MueLu_MueluSmoother_fwd.hpp"
#include "MueLu_SmootherPrototype.hpp"

#include "MueLu_FactoryBase_fwd.hpp"
#include "MueLu_SpaiSmoother_fwd.hpp"
#include "MueLu_MultigridSmoother_fwd.hpp"

namespace MueLu {

  /*!
    @class MueluSmoother
    @ingroup MueLuSmootherClasses
    @brief Class that encapsulates smoothers build in Muelu.

  */

  template <class Scalar = SmootherPrototype<>::scalar_type,
            class LocalOrdinal = typename SmootherPrototype<Scalar>::local_ordinal_type,
            class GlobalOrdinal = typename SmootherPrototype<Scalar, LocalOrdinal>::global_ordinal_type,
            class Node = typename SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
  class MueluSmoother : public SmootherPrototype<Scalar,LocalOrdinal,GlobalOrdinal,Node> {
#undef MUELU_MUELUSMOOTHER_SHORT
#include "MueLu_UseShortNames.hpp"

   public:

    //! @name Constructors / destructors
    //@{

    /*! @brief Constructor

          @param[in] type Smoother type.  Can currently be
           - SPAI for sparse approximate inverse smoothing
           - MULTIGRID for amg smoothing

          @param[in] paramList  A list holding parameters understood by the smoother setup.
    */
    MueluSmoother(const std::string& type = "", const Teuchos::ParameterList& paramList = Teuchos::ParameterList());

    //! Destructor
    virtual ~MueluSmoother() { };

    //@}

    //! Input
    //! @{

    void DeclareInput(Level& currentLevel) const;

    // void SetParameterList(const Teuchos::ParameterList& paramList);

    //!@}

    //! @name Setup and Apply methods.
    //@{

    //! TrilinosSmoother cannot be turned into a smoother using Setup(). Setup() always returns a RuntimeError exception.
    void Setup(Level& currentLevel);

    //! TrilinosSmoother cannot be applied. Apply() always returns a RuntimeError exception.
    void Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero = false) const;

    //@}

    //! Custom SetFactory
    void SetFactory(const std::string& varName, const RCP<const FactoryBase>& factory);

    //! When this prototype is cloned using Copy(), the clone is an Ifpack or an Ifpack2 smoother.
    RCP<SmootherPrototype> Copy() const;

        //! @name Overridden from Teuchos::Describable
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const;

    //! Print the object with some verbosity level to an FancyOStream object.
    //using MueLu::Describable::describe; // overloading, not hiding
    //void describe(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const {
    void print(Teuchos::FancyOStream& out, const VerbLevel verbLevel = Default) const;

    //! For diagnostic purposes
    RCP<SmootherPrototype> getSmoother() {return s_;}

    //! Get a rough estimate of cost per iteration
    size_t getNodeSmootherComplexity() const {return s_->getNodeSmootherComplexity();}

    //@}

  private:

    //! key phrase that denote smoother type
    std::string type_;

    //! A Factory
    RCP<FactoryBase> AFact_;

    //! Underlying Smoother
    RCP<SmootherPrototype> sSpai_, sMultigrid_;
    mutable
      RCP<SmootherPrototype> s_;

    // Records for the case if something goes wrong
    bool        triedSpai_, triedMultigrid_;
    std::string errorSpai_, errorMultigrid_;

  }; // class MueLuSmoother

} // namespace MueLu

#define MUELU_MUELUSMOOTHER_SHORT
#endif // MUELU_MUELUSMOOTHER_DECL_HPP
