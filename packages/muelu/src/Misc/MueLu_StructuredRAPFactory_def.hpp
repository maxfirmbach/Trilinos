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
#ifndef MUELU_STRUCTUREDRAPFACTORY_DEF_HPP
#define MUELU_STRUCTUREDRAPFACTORY_DEF_HPP


#include <sstream>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_MatrixUtils.hpp>
#include <Xpetra_TripleMatrixMultiply.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_CrsGraph.hpp>

#include "MueLu_StructuredRAPFactory_decl.hpp"

#include "MueLu_MasterList.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_PerfUtils.hpp"  
#include "MueLu_StructuredRAPFactory_decl.hpp"
//#include "MueLu_Utilities.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::StructuredRAPFactory()
    : hasDeclaredInput_(false) { }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

#define SET_VALID_ENTRY(name) validParamList->setEntry(name, MasterList::getEntry(name))
    SET_VALID_ENTRY("transpose: use implicit");
    SET_VALID_ENTRY("rap: triple product");
    SET_VALID_ENTRY("rap: fix zero diagonals");
    SET_VALID_ENTRY("rap: fix zero diagonals threshold");
    SET_VALID_ENTRY("rap: fix zero diagonals replacement");
    SET_VALID_ENTRY("rap: relative diagonal floor");  
    SET_VALID_ENTRY("rap: structure type"); // set by user to define matrix structure (e.g. Laplace2D)
#undef  SET_VALID_ENTRY
    validParamList->set< RCP<const FactoryBase> >("A", null, "Generating factory of the matrix A used during the prolongator smoothing process");
    validParamList->set< RCP<const FactoryBase> >("P", null, "Prolongator factory");
    validParamList->set< RCP<const FactoryBase> >("R", null, "Restrictor factory");
    
    validParamList->set<RCP<const FactoryBase> >("numDimensions",                Teuchos::null,
                                                 "Number of spacial dimensions in the problem.");
    validParamList->set<RCP<const FactoryBase> >("lCoarseNodesPerDim",           Teuchos::null,
                                                 "Number of nodes per spatial dimension on the coarse grid.");     

    validParamList->set< bool >                  ("CheckMainDiagonal",  false, "Check main diagonal for zeros");
    validParamList->set< bool >                  ("RepairMainDiagonal", false, "Repair zeros on main diagonal");

    // Make sure we don't recursively validate options for the matrixmatrix kernels
    ParameterList norecurse;
    norecurse.disableRecursiveValidation();
    validParamList->set<ParameterList> ("matrixmatrix: kernel params", norecurse, "MatrixMatrix kernel parameters");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &fineLevel, Level &coarseLevel) const {
    const Teuchos::ParameterList& pL = GetParameterList();
    if (pL.get<bool>("transpose: use implicit") == false)
      Input(coarseLevel, "R");

    Input(fineLevel,   "A");
    Input(coarseLevel, "P");

    // get structured information
    Input(fineLevel, "numDimensions");
    Input(fineLevel, "lCoarseNodesPerDim");
    
    // call DeclareInput of all user-given transfer factories
    for (std::vector<RCP<const FactoryBase> >::const_iterator it = transferFacts_.begin(); it != transferFacts_.end(); ++it)
      (*it)->CallDeclareInput(coarseLevel);

    hasDeclaredInput_ = true;
  }
   
  // Here do the pattern determination ...
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetLaplace1D(RCP<Matrix>& Ac, RCP<Matrix> P,
                                                                              	     Teuchos::Array<LocalOrdinal> lCoarseNodesPerDim) const
  {         
    // Define some containers for the compressed row storage
    int ncoarse = lCoarseNodesPerDim[0];
    int maxNnzOnRow = 3; // nnzOnRow is here ~3 for Laplace1D
    const RCP<ParameterList> paramList = Teuchos::null;
  
    // get graph container  
    auto rowMap = P->getDomainMap();
    auto colMap = P->getColMap();
    RCP<CrsGraph> myGraph = CrsGraphFactory::Build(rowMap, colMap, maxNnzOnRow, paramList);
    
    int end = 4+(ncoarse-2)*3;
    const ArrayRCP<LO> colind(end);
    const ArrayRCP<size_t> rowptr(ncoarse+1);
    rowptr[0] = 0;

    // set the crs pattern into the graph with local Indices
    rowptr[1] = rowptr[0]+2;    
    colind[0] = 0;
    colind[1] = 1;
    for(int rowIdx=1; rowIdx<ncoarse-1; rowIdx++) {
      int k = rowptr[rowIdx];
      rowptr[rowIdx+1] = rowptr[rowIdx]+3;
      colind[k]   = rowIdx-1;
      colind[k+1] = rowIdx;
      colind[k+2] = rowIdx+1;
    }
    rowptr[ncoarse] = rowptr[ncoarse-1]+2;
    colind[end-2] = ncoarse-2;
    colind[end-1] = ncoarse-1;
    
    std::cout << "Graph indices created!" << std::endl;
    myGraph->setAllIndices(rowptr, colind);

    std::cout << "Graph is created and filled!" << std::endl;
    myGraph->fillComplete();
 
    // build Ac with static graph pattern ...
    Ac = MatrixFactory::Build(myGraph, paramList); 
  }
  
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetLaplace2D(RCP<Matrix>& Ac, RCP<Matrix> P,
                                                                              	     Teuchos::Array<LocalOrdinal> lCoarseNodesPerDim) const
  {
    // Define some containers for the compressed row storage
    int ncoarse = lCoarseNodesPerDim[0];
    int maxNnzOnRow = 5; // nnzOnRow is here ~5 for Laplace2D
    const RCP<ParameterList> paramList = Teuchos::null;
  
    // get graph container  
    auto rowMap = P->getDomainMap();
    auto colMap = P->getColMap();
    RCP<CrsGraph> myGraph = CrsGraphFactory::Build(rowMap, colMap, maxNnzOnRow, paramList);

    const ArrayRCP<LO> colind(4*3+2*(ncoarse-2)*4+(ncoarse-2)*2*4+(ncoarse-2)*(ncoarse-2)*5); // TODO: is this correct?
    const ArrayRCP<size_t> rowptr(ncoarse*ncoarse+1);
    rowptr[0] = 0;
    
    // set the crs pattern into the graph
    for(int rowIdx=0; rowIdx<ncoarse; rowIdx++) {
      if(rowIdx==0) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+3;
        colind[k]   = rowIdx;
        colind[k+1] = rowIdx+1;
        colind[k+2] = rowIdx+ncoarse;
      }      
      else if(rowIdx==ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+3;
        colind[k]   = rowIdx-1;
        colind[k+1] = rowIdx;
        colind[k+2] = rowIdx+ncoarse;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+4;
        colind[k]   = rowIdx-1;
        colind[k+1] = rowIdx;
        colind[k+2] = rowIdx+1;
        colind[k+3] = rowIdx+ncoarse;
      }
    }
    for(int i=1; i<ncoarse-1; i++) { // loop over blocks
      for(int j=0; j<ncoarse; j++) { // loop over inside
        int rowIdx = i*ncoarse+j;
        if(j==0) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+4;
          colind[k]   = rowIdx-ncoarse;
          colind[k+1] = rowIdx;
          colind[k+2] = rowIdx+1;
          colind[k+3] = rowIdx+ncoarse;
        }
        else if(j==ncoarse-1) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+4;
          colind[k]   = rowIdx-ncoarse;
          colind[k+1] = rowIdx-1;
          colind[k+2] = rowIdx;
          colind[k+3] = rowIdx+ncoarse;
        }
        else {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+5;
          colind[k]   = rowIdx-ncoarse;
          colind[k+1] = rowIdx-1;
          colind[k+2] = rowIdx;
          colind[k+3] = rowIdx+1;
          colind[k+4] = rowIdx+ncoarse; 
        }
      }
    }
    for(int rowIdx=ncoarse*ncoarse-ncoarse; rowIdx<ncoarse*ncoarse; rowIdx++) {
      if(rowIdx==ncoarse*ncoarse-ncoarse) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+3;
        colind[k]   = rowIdx-ncoarse;
        colind[k+1] = rowIdx;
        colind[k+2] = rowIdx+1;
      }
      else if(rowIdx==ncoarse*ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+3;
        colind[k]   = rowIdx-ncoarse;
        colind[k+1] = rowIdx-1;
        colind[k+2] = rowIdx;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+4;
        colind[k]   = rowIdx-ncoarse;
        colind[k+1] = rowIdx-1;
        colind[k+2] = rowIdx;
        colind[k+3] = rowIdx+1;
      }
    }

    std::cout << "Graph indices created!" << std::endl;
    myGraph->setAllIndices(rowptr, colind);

    std::cout << "Graph is created and filled!" << std::endl;
    myGraph->fillComplete();
 
    // build Ac with static graph pattern ...
    Ac = MatrixFactory::Build(myGraph, paramList); 
  }
  
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetStar2D(RCP<Matrix>& Ac, RCP<Matrix> P,
                                                                              	  Teuchos::Array<LocalOrdinal> lCoarseNodesPerDim) const
  {    
    // Define some containers for the compressed row storage
    int ncoarse = lCoarseNodesPerDim[0];
    int maxNnzOnRow = 9; // nnzOnRow is here ~9 for Star2D
    const RCP<ParameterList> paramList = Teuchos::null;
  
    // get graph container  
    auto rowMap = P->getDomainMap();
    auto colMap = P->getColMap();
    RCP<CrsGraph> myGraph = CrsGraphFactory::Build(rowMap, colMap, maxNnzOnRow, paramList);

    const ArrayRCP<LO> colind(4*4+2*(ncoarse-2)*6+(ncoarse-2)*2*6+(ncoarse-2)*(ncoarse-2)*9); // TODO: is this correct?
    const ArrayRCP<size_t> rowptr(ncoarse*ncoarse+1);
    rowptr[0] = 0;
    
    // set the crs pattern into the graph
    for(int rowIdx=0; rowIdx<ncoarse; rowIdx++)
    {
      if(rowIdx==0) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+4;
        colind[k]   = rowIdx;
        colind[k+1] = rowIdx+1;
        colind[k+2] = rowIdx+ncoarse;
        colind[k+3] = rowIdx+ncoarse+1;
      }
      else if(rowIdx==ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+4;
        colind[k]   = rowIdx-1;
        colind[k+1] = rowIdx;
        colind[k+2] = rowIdx+ncoarse-1;
        colind[k+3] = rowIdx+ncoarse;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+6;
        colind[k]   = rowIdx-1;
        colind[k+1] = rowIdx;
        colind[k+2] = rowIdx+1;
        colind[k+3] = rowIdx+ncoarse-1;
        colind[k+4] = rowIdx+ncoarse;
        colind[k+5] = rowIdx+ncoarse+1;
      }
    }
    for(int i=1; i<ncoarse-1; i++) { // loop over blocks
      for(int j=0; j<ncoarse; j++) { // loop over inside
        int rowIdx = i*ncoarse+j;
        if(j==0) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+6;
          colind[k]   = rowIdx-ncoarse;
          colind[k+1] = rowIdx-ncoarse+1;
          colind[k+2] = rowIdx;
          colind[k+3] = rowIdx+1;
          colind[k+4] = rowIdx+ncoarse;
          colind[k+5] = rowIdx+ncoarse+1;
        }
        else if(j==ncoarse-1) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+6;
          colind[k]   = rowIdx-ncoarse-1;
          colind[k+1] = rowIdx-ncoarse;
          colind[k+2] = rowIdx-1;
          colind[k+3] = rowIdx;
          colind[k+4] = rowIdx+ncoarse-1;
          colind[k+5] = rowIdx+ncoarse;
        }
        else {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+9;
          colind[k]   = rowIdx-ncoarse-1;
          colind[k+1] = rowIdx-ncoarse;
          colind[k+2] = rowIdx-ncoarse+1;
          colind[k+3] = rowIdx-1;
          colind[k+4] = rowIdx;
          colind[k+5] = rowIdx+1;
          colind[k+6] = rowIdx+ncoarse-1;
          colind[k+7] = rowIdx+ncoarse;
          colind[k+8] = rowIdx+ncoarse+1;
        }
      }
    }
    for(int rowIdx=ncoarse*ncoarse-ncoarse; rowIdx<ncoarse*ncoarse; rowIdx++) {
      if(rowIdx==ncoarse*ncoarse-ncoarse) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+4;
        colind[k]   = rowIdx-ncoarse;
        colind[k+1] = rowIdx-ncoarse+1;
        colind[k+2] = rowIdx;
        colind[k+3] = rowIdx+1;
      }
      else if(rowIdx==ncoarse*ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+4;
        colind[k]   = rowIdx-ncoarse-1;
        colind[k+1] = rowIdx-ncoarse;
        colind[k+2] = rowIdx-1;
        colind[k+3] = rowIdx;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+6;
        colind[k]   = rowIdx-ncoarse-1;
        colind[k+1] = rowIdx-ncoarse;
        colind[k+2] = rowIdx-ncoarse+1;
        colind[k+3] = rowIdx-1;
        colind[k+4] = rowIdx;
        colind[k+5] = rowIdx+1;
      }
    }

    std::cout << "Graph indices created!" << std::endl;
    myGraph->setAllIndices(rowptr, colind);

    std::cout << "Graph is created and filled!" << std::endl;
    myGraph->fillComplete();
 
    // build Ac with static graph pattern ...
    Ac = MatrixFactory::Build(myGraph, paramList); 
  }
  
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetBigStar2D(RCP<Matrix>& Ac, RCP<Matrix> P,
                                                                              	     Teuchos::Array<LocalOrdinal> lCoarseNodesPerDim) const
  {
    // Something goes wrong here!

    // Define some containers for the compressed row storage
    int ncoarse = lCoarseNodesPerDim[0];
    int maxNnzOnRow = 21; // nnzOnRow is here ~21 for BigStar2D
    const RCP<ParameterList> paramList = Teuchos::null;
  
    // get graph container  
    auto rowMap = P->getDomainMap();
    auto colMap = P->getColMap();
    RCP<CrsGraph> myGraph = CrsGraphFactory::Build(rowMap, colMap, maxNnzOnRow, paramList);

    const ArrayRCP<LO> colind((4*8+4*11+2*13*(ncoarse-4))+(2*11+2*15+2*18*(ncoarse-4))
                               +(ncoarse-4)*2*13+(ncoarse-4)*2*18+(ncoarse-4)*(ncoarse-4)*21); // TODO: is this correct?
    const ArrayRCP<size_t> rowptr(ncoarse*ncoarse+1);
    rowptr[0] = 0;
    
    // set the crs pattern into the graph
    for(int rowIdx=0; rowIdx<ncoarse; rowIdx++) {
      if(rowIdx==0) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+8;
        colind[k]   = rowIdx;
        colind[k+1] = rowIdx+1;
        colind[k+2] = rowIdx+2;
        colind[k+3] = rowIdx+ncoarse;
        colind[k+4] = rowIdx+ncoarse+1;
        colind[k+5] = rowIdx+ncoarse+2;
        colind[k+6] = rowIdx+2*ncoarse;
        colind[k+7] = rowIdx+2*ncoarse+1;
      }
      else if(rowIdx==1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-1;
        colind[k+1]  = rowIdx;
        colind[k+2]  = rowIdx+1;
        colind[k+3]  = rowIdx+2;
        colind[k+4]  = rowIdx+ncoarse-1;
        colind[k+5]  = rowIdx+ncoarse;
        colind[k+6]  = rowIdx+ncoarse+1;
        colind[k+7]  = rowIdx+ncoarse+2;
        colind[k+8]  = rowIdx+2*ncoarse-1;
        colind[k+9]  = rowIdx+2*ncoarse;
        colind[k+10] = rowIdx+2*ncoarse+1;
      }  
      else if(rowIdx==ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+8;
        colind[k]   = rowIdx-2;
        colind[k+1] = rowIdx-1;
        colind[k+2] = rowIdx;
        colind[k+3] = rowIdx+ncoarse-2;
        colind[k+4] = rowIdx+ncoarse-1;
        colind[k+5] = rowIdx+ncoarse;
        colind[k+6] = rowIdx+2*ncoarse-1;
        colind[k+7] = rowIdx+2*ncoarse;
      }
      else if(rowIdx==ncoarse-2) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-2;
        colind[k+1]  = rowIdx-1;
        colind[k+2]  = rowIdx;
        colind[k+3]  = rowIdx+1;
        colind[k+4]  = rowIdx+ncoarse-2;
        colind[k+5]  = rowIdx+ncoarse-1;
        colind[k+6]  = rowIdx+ncoarse;
        colind[k+7]  = rowIdx+ncoarse+1;
        colind[k+8]  = rowIdx+2*ncoarse-1;
        colind[k+9]  = rowIdx+2*ncoarse;
        colind[k+10] = rowIdx+2*ncoarse+1;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+13;
        colind[k]    = rowIdx-2;
        colind[k+1]  = rowIdx-1;
        colind[k+2]  = rowIdx;
        colind[k+3]  = rowIdx+1;
        colind[k+4]  = rowIdx+2;
        colind[k+5]  = rowIdx+ncoarse-2;
        colind[k+6]  = rowIdx+ncoarse-1;
        colind[k+7]  = rowIdx+ncoarse;
        colind[k+8]  = rowIdx+ncoarse+1; 
        colind[k+9]  = rowIdx+ncoarse+2;
        colind[k+10] = rowIdx+2*ncoarse-1;
        colind[k+11] = rowIdx+2*ncoarse;
        colind[k+12] = rowIdx+2*ncoarse+1;
      }
    }
    for(int rowIdx=ncoarse; rowIdx<2*ncoarse; rowIdx++) {
      if(rowIdx==ncoarse) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-ncoarse;
        colind[k+1]  = rowIdx-ncoarse+1;
        colind[k+2]  = rowIdx-ncoarse+2;
        colind[k+3]  = rowIdx;
        colind[k+4]  = rowIdx+1;
        colind[k+5]  = rowIdx+2;
        colind[k+6]  = rowIdx+ncoarse;
        colind[k+7]  = rowIdx+ncoarse+1;
        colind[k+8]  = rowIdx+ncoarse+2;
        colind[k+9]  = rowIdx+2*ncoarse;
        colind[k+10] = rowIdx+2*ncoarse+1;
      }
      else if(rowIdx==ncoarse+1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+15;
        colind[k]    = rowIdx-ncoarse-1;
        colind[k+1]  = rowIdx-ncoarse;
        colind[k+2]  = rowIdx-ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse+2;
        colind[k+4]  = rowIdx-1;
        colind[k+5]  = rowIdx;
        colind[k+6]  = rowIdx+1;
        colind[k+7]  = rowIdx+2;
        colind[k+8]  = rowIdx+ncoarse-1;
        colind[k+9]  = rowIdx+ncoarse;
        colind[k+10] = rowIdx+ncoarse+1;
        colind[k+11] = rowIdx+ncoarse+2;
        colind[k+12] = rowIdx+2*ncoarse-1;
        colind[k+13] = rowIdx+2*ncoarse;
        colind[k+14] = rowIdx+2*ncoarse+1;
      }
      else if(rowIdx==2*ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-ncoarse-2;
        colind[k+1]  = rowIdx-ncoarse-1;
        colind[k+2]  = rowIdx-ncoarse;
        colind[k+3]  = rowIdx-2;
        colind[k+4]  = rowIdx-1;
        colind[k+5]  = rowIdx;
        colind[k+6]  = rowIdx+ncoarse-2;
        colind[k+7]  = rowIdx+ncoarse-1;
        colind[k+8]  = rowIdx+ncoarse;
        colind[k+9]  = rowIdx+2*ncoarse-1;
        colind[k+10] = rowIdx+2*ncoarse;
      }
      else if(rowIdx==2*ncoarse-2) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+15;        
        colind[k]    = rowIdx-ncoarse-2;
        colind[k+1]  = rowIdx-ncoarse-1;
        colind[k+2]  = rowIdx-ncoarse;
        colind[k+3]  = rowIdx-ncoarse+1;
        colind[k+4]  = rowIdx-2;
        colind[k+5]  = rowIdx-1;
        colind[k+6]  = rowIdx;
        colind[k+7]  = rowIdx+1; 
        colind[k+8]  = rowIdx+ncoarse-2;
        colind[k+9]  = rowIdx+ncoarse-1;
        colind[k+10] = rowIdx+ncoarse;
        colind[k+11] = rowIdx+ncoarse+1;
        colind[k+12] = rowIdx+2*ncoarse-1;
        colind[k+13] = rowIdx+2*ncoarse;
        colind[k+14] = rowIdx+2*ncoarse+1;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+18;
        colind[k]    = rowIdx-ncoarse-2;
        colind[k+1]  = rowIdx-ncoarse-1;
        colind[k+2]  = rowIdx-ncoarse;
        colind[k+3]  = rowIdx-ncoarse+1;
        colind[k+4]  = rowIdx-ncoarse+2;
        colind[k+5]  = rowIdx-2;
        colind[k+6]  = rowIdx-1;
        colind[k+7]  = rowIdx;
        colind[k+8]  = rowIdx+1;
        colind[k+9]  = rowIdx+2;
        colind[k+10] = rowIdx+ncoarse-2;
        colind[k+11] = rowIdx+ncoarse-1;
        colind[k+12] = rowIdx+ncoarse;
        colind[k+13] = rowIdx+ncoarse+1;
        colind[k+14] = rowIdx+ncoarse+2;
        colind[k+15] = rowIdx+2*ncoarse-1;
        colind[k+16] = rowIdx+2*ncoarse;
        colind[k+17] = rowIdx+2*ncoarse+1;
      }
    }
    for(int i=2; i<ncoarse-2; i++) { // loop over blocks
      for(int j=0; j<ncoarse; j++) { // loop over inside
        int rowIdx = i*ncoarse+j;
        if(j==0) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+13;
          colind[k]    = rowIdx-2*ncoarse;
          colind[k+1]  = rowIdx-2*ncoarse+1;
          colind[k+2]  = rowIdx-ncoarse;
          colind[k+3]  = rowIdx-ncoarse+1;
          colind[k+4]  = rowIdx-ncoarse+2;
          colind[k+5]  = rowIdx;
          colind[k+6]  = rowIdx+1;
          colind[k+7]  = rowIdx+2;
          colind[k+8]  = rowIdx+ncoarse;
          colind[k+9]  = rowIdx+ncoarse+1;
          colind[k+10] = rowIdx+ncoarse+2;
          colind[k+11] = rowIdx+2*ncoarse;
          colind[k+12] = rowIdx+2*ncoarse+1;
        }
        else if(j==1) {
           int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+18;
          colind[k]    = rowIdx-2*ncoarse-1;
          colind[k+1]  = rowIdx-2*ncoarse;
          colind[k+2]  = rowIdx-2*ncoarse+1;
          colind[k+3]  = rowIdx-ncoarse-1;
          colind[k+4]  = rowIdx-ncoarse;
          colind[k+5]  = rowIdx-ncoarse+1;
          colind[k+6]  = rowIdx-ncoarse+2;
          colind[k+7]  = rowIdx-1;
          colind[k+8]  = rowIdx;
          colind[k+9]  = rowIdx+1;
          colind[k+10] = rowIdx+2;
          colind[k+11] = rowIdx+ncoarse-1;
          colind[k+12] = rowIdx+ncoarse;
          colind[k+13] = rowIdx+ncoarse+1;
          colind[k+14] = rowIdx+ncoarse+2;
          colind[k+15] = rowIdx+2*ncoarse-1;
          colind[k+16] = rowIdx+2*ncoarse;
          colind[k+17] = rowIdx+2*ncoarse+1;
        }
        else if(j==ncoarse-1) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+13;
          colind[k]    = rowIdx-2*ncoarse-1;
          colind[k+1]  = rowIdx-2*ncoarse;
          colind[k+2]  = rowIdx-ncoarse-2;
          colind[k+3]  = rowIdx-ncoarse-1;
          colind[k+4]  = rowIdx-ncoarse;
          colind[k+5]  = rowIdx-2;
          colind[k+6]  = rowIdx-1;
          colind[k+7]  = rowIdx;
          colind[k+8]  = rowIdx+ncoarse-2;
          colind[k+9]  = rowIdx+ncoarse-1;
          colind[k+10] = rowIdx+ncoarse;
          colind[k+11] = rowIdx+2*ncoarse-1;
          colind[k+12] = rowIdx+2*ncoarse;
        }
        else if(j==ncoarse-2) {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+18;
          colind[k]    = rowIdx-2*ncoarse-1;
          colind[k+2]  = rowIdx-2*ncoarse;
          colind[k+3]  = rowIdx-2*ncoarse+1;
          colind[k+4]  = rowIdx-ncoarse-2;
          colind[k+5]  = rowIdx-ncoarse-1;
          colind[k+6]  = rowIdx-ncoarse;
          colind[k+7]  = rowIdx-ncoarse+1;
          colind[k+8]  = rowIdx-2;
          colind[k+9]  = rowIdx-1;
          colind[k+10] = rowIdx;
          colind[k+11] = rowIdx+1;
          colind[k+12] = rowIdx+ncoarse-2;
          colind[k+13] = rowIdx+ncoarse-1;
          colind[k+14] = rowIdx+ncoarse;
          colind[k+15] = rowIdx+ncoarse+1;
          colind[k+16] = rowIdx+2*ncoarse-1;
          colind[k+17] = rowIdx+2*ncoarse;
          colind[k+18] = rowIdx+2*ncoarse+1;
        }
        else {
          int k = rowptr[rowIdx];
          rowptr[rowIdx+1] = rowptr[rowIdx]+21;
          colind[k]    = rowIdx-2*ncoarse-1;
          colind[k+1]  = rowIdx-2*ncoarse;
          colind[k+2]  = rowIdx-2*ncoarse+1;
          colind[k+3]  = rowIdx-ncoarse-2;
          colind[k+4]  = rowIdx-ncoarse-1;
          colind[k+5]  = rowIdx-ncoarse;
          colind[k+6]  = rowIdx-ncoarse+1;
          colind[k+7]  = rowIdx-ncoarse+2;
          colind[k+8]  = rowIdx-2;
          colind[k+9]  = rowIdx-1;
          colind[k+10] = rowIdx;
          colind[k+11] = rowIdx+1;
          colind[k+12] = rowIdx+2;
          colind[k+13] = rowIdx+ncoarse-2;
          colind[k+14] = rowIdx+ncoarse-1;
          colind[k+15] = rowIdx+ncoarse;
          colind[k+16] = rowIdx+ncoarse+1;
          colind[k+17] = rowIdx+ncoarse+2;
          colind[k+18] = rowIdx+2*ncoarse-1;
          colind[k+19] = rowIdx+2*ncoarse;
          colind[k+20] = rowIdx+2*ncoarse+1;
        }
      }
    }
    for(int rowIdx=ncoarse*ncoarse-2*ncoarse; rowIdx<ncoarse*ncoarse-ncoarse; rowIdx++) {
      if(rowIdx==ncoarse*ncoarse-2*ncoarse) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-2*ncoarse;
        colind[k+1]  = rowIdx-2*ncoarse+1;
        colind[k+2]  = rowIdx-ncoarse;
        colind[k+3]  = rowIdx-ncoarse+1;
        colind[k+4]  = rowIdx-ncoarse+2;
        colind[k+5]  = rowIdx;
        colind[k+6]  = rowIdx+1;
        colind[k+7]  = rowIdx+2;
        colind[k+8]  = rowIdx+ncoarse;
        colind[k+9]  = rowIdx+ncoarse+1;
        colind[k+10] = rowIdx+ncoarse+2;
      }
      else if(rowIdx==ncoarse*ncoarse-2*ncoarse+1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+15;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-2*ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse-1;
        colind[k+4]  = rowIdx-ncoarse;
        colind[k+5]  = rowIdx-ncoarse+1;
        colind[k+6]  = rowIdx-ncoarse+2;
        colind[k+7]  = rowIdx-1;
        colind[k+8]  = rowIdx;
        colind[k+9]  = rowIdx+1;
        colind[k+10] = rowIdx+2;
        colind[k+11] = rowIdx+ncoarse-1;
        colind[k+12] = rowIdx+ncoarse;
        colind[k+13] = rowIdx+ncoarse+1;
        colind[k+14] = rowIdx+ncoarse+2;
      }
      else if(rowIdx==ncoarse*ncoarse-ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-ncoarse-2;
        colind[k+3]  = rowIdx-ncoarse-1;
        colind[k+4]  = rowIdx-ncoarse;
        colind[k+5]  = rowIdx-2;
        colind[k+6]  = rowIdx-1;
        colind[k+7]  = rowIdx;
        colind[k+8]  = rowIdx+ncoarse-2;
        colind[k+9]  = rowIdx+ncoarse-1;
        colind[k+10] = rowIdx+ncoarse;
      }
      else if(rowIdx==ncoarse*ncoarse-ncoarse-2) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+15;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-2*ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse-2;
        colind[k+4]  = rowIdx-ncoarse-1;
        colind[k+5]  = rowIdx-ncoarse;
        colind[k+6]  = rowIdx-ncoarse+1;
        colind[k+7]  = rowIdx-2;
        colind[k+8]  = rowIdx-1;
        colind[k+9]  = rowIdx;
        colind[k+10] = rowIdx+1;
        colind[k+11] = rowIdx+ncoarse-2;
        colind[k+12] = rowIdx+ncoarse-1;
        colind[k+13] = rowIdx+ncoarse;
        colind[k+14] = rowIdx+ncoarse+1;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+18;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-2*ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse-2;
        colind[k+4]  = rowIdx-ncoarse-1;
        colind[k+5]  = rowIdx-ncoarse;
        colind[k+6]  = rowIdx-ncoarse+1;
        colind[k+7]  = rowIdx-ncoarse+2;
        colind[k+8]  = rowIdx-2;
        colind[k+9]  = rowIdx-1;
        colind[k+10] = rowIdx;
        colind[k+11] = rowIdx+1;
        colind[k+12] = rowIdx+2;
        colind[k+13] = rowIdx+ncoarse-2;
        colind[k+14] = rowIdx+ncoarse-1;
        colind[k+15] = rowIdx+ncoarse;
        colind[k+16] = rowIdx+ncoarse+1;
        colind[k+17] = rowIdx+ncoarse+2;
      }
    }     
    for(int rowIdx=ncoarse*ncoarse-ncoarse; rowIdx<ncoarse*ncoarse; rowIdx++) {
      if(rowIdx==ncoarse*ncoarse-ncoarse) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+8;
        colind[k]   = rowIdx-2*ncoarse;
        colind[k+1] = rowIdx-2*ncoarse+1;
        colind[k+2] = rowIdx-ncoarse;
        colind[k+3] = rowIdx-ncoarse+1;
        colind[k+4] = rowIdx-ncoarse+2;
        colind[k+5] = rowIdx;
        colind[k+6] = rowIdx+1;
        colind[k+7] = rowIdx+2;
      }
      else if(rowIdx==ncoarse*ncoarse-ncoarse+1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-2*ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse-1;
        colind[k+4]  = rowIdx-ncoarse;
        colind[k+5]  = rowIdx-ncoarse+1;
        colind[k+6]  = rowIdx-ncoarse+2;
        colind[k+7]  = rowIdx-1;
        colind[k+8]  = rowIdx;
        colind[k+9]  = rowIdx+1;
        colind[k+10] = rowIdx+2;
      }
      else if(rowIdx==ncoarse*ncoarse-1) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+8;
        colind[k]   = rowIdx-2*ncoarse-1;
        colind[k+1] = rowIdx-2*ncoarse;
        colind[k+2] = rowIdx-ncoarse-2;
        colind[k+3] = rowIdx-ncoarse-1;
        colind[k+4] = rowIdx-ncoarse;
        colind[k+5] = rowIdx-2;
        colind[k+6] = rowIdx-1;
        colind[k+7] = rowIdx;
      }
      else if(rowIdx==ncoarse*ncoarse-2) {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+11;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-2*ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse-2;
        colind[k+4]  = rowIdx-ncoarse-1;
        colind[k+5]  = rowIdx-ncoarse;
        colind[k+6]  = rowIdx-ncoarse+1;
        colind[k+7]  = rowIdx-2;
        colind[k+8]  = rowIdx-1;
        colind[k+9]  = rowIdx;
        colind[k+10] = rowIdx+1;
      }
      else {
        int k = rowptr[rowIdx];
        rowptr[rowIdx+1] = rowptr[rowIdx]+13;
        colind[k]    = rowIdx-2*ncoarse-1;
        colind[k+1]  = rowIdx-2*ncoarse;
        colind[k+2]  = rowIdx-2*ncoarse+1;
        colind[k+3]  = rowIdx-ncoarse-2;
        colind[k+4]  = rowIdx-ncoarse-1;
        colind[k+5]  = rowIdx-ncoarse;
        colind[k+6]  = rowIdx-ncoarse+1;
        colind[k+7]  = rowIdx-ncoarse+2;
        colind[k+8]  = rowIdx-2;
        colind[k+9]  = rowIdx-1;
        colind[k+10] = rowIdx;
        colind[k+11] = rowIdx+1;
        colind[k+12] = rowIdx+2;
      }
    }

    std::cout << "Graph indices created!" << std::endl;
    myGraph->setAllIndices(rowptr, colind);

    std::cout << "Graph is created and filled!" << std::endl;
    myGraph->fillComplete();
 
    // build Ac with static graph pattern ...
    Ac = MatrixFactory::Build(myGraph, paramList); 
  }
  
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetElasticity2D(RCP<Matrix>& Ac, RCP<Matrix> P,
                                                                              	        Teuchos::Array<LocalOrdinal> lCoarseNodesPerDim) const
  {
    // Define some containers for the compressed row storage
    int ncoarse = lCoarseNodesPerDim[0];
    int maxNnzOnRow = 18; // nnzOnRow is here ~18 for Elasticity2D
    const RCP<ParameterList> paramList = Teuchos::null;
    const int dim = 2;
    
    // get graph container  
    auto rowMap = P->getDomainMap();
    auto colMap = P->getColMap();
    RCP<CrsGraph> myGraph = CrsGraphFactory::Build(rowMap, colMap, maxNnzOnRow, paramList);

    const ArrayRCP<LO> colind(dim*dim*(dim*8+(ncoarse-dim)*12)+dim*(ncoarse-dim)*(dim*12+(ncoarse-dim)*18));
    const ArrayRCP<size_t> rowptr(dim*ncoarse*ncoarse+1);
    rowptr[0] = 0;

    // set the crs pattern into the graph
    int end1 = dim*ncoarse;
    int if1  = dim*ncoarse-dim;
    for(int rowIdx=0; rowIdx<end1; rowIdx=rowIdx+dim) {
      if(rowIdx == 0) {
        for(int ndof=0; ndof<dim; ndof++) {
          int k = rowptr[rowIdx+ndof];
          rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+8;
          colind[k]   = rowIdx;
          colind[k+1] = rowIdx+1;
          colind[k+2] = rowIdx+2;
          colind[k+3] = rowIdx+3;
          colind[k+4] = rowIdx+dim*ncoarse;
          colind[k+5] = rowIdx+dim*ncoarse+1;
          colind[k+6] = rowIdx+dim*ncoarse+2;
          colind[k+7] = rowIdx+dim*ncoarse+3;          
        }
      }
      else if(rowIdx == if1) {
        for(int ndof=0; ndof<dim; ndof++) {
          int k = rowptr[rowIdx+ndof];
          rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+8;
          colind[k]   = rowIdx-2;
          colind[k+1] = rowIdx-1;
          colind[k+2] = rowIdx;
          colind[k+3] = rowIdx+1;
          colind[k+4] = rowIdx+dim*ncoarse-2;
          colind[k+5] = rowIdx+dim*ncoarse-1;
          colind[k+6] = rowIdx+dim*ncoarse;
          colind[k+7] = rowIdx+dim*ncoarse+1;        
        }
      }
      else {
        for(int ndof=0; ndof<dim; ndof++) {
          int k = rowptr[rowIdx+ndof];
          rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+12;
          colind[k]    = rowIdx-2;
          colind[k+1]  = rowIdx-1;
          colind[k+2]  = rowIdx;
          colind[k+3]  = rowIdx+1;
          colind[k+4]  = rowIdx+2;
          colind[k+5]  = rowIdx+3;
          colind[k+6]  = rowIdx+dim*ncoarse-2;
          colind[k+7]  = rowIdx+dim*ncoarse-1;
          colind[k+8]  = rowIdx+dim*ncoarse;
          colind[k+9]  = rowIdx+dim*ncoarse+1;
          colind[k+10] = rowIdx+dim*ncoarse+2;
          colind[k+11] = rowIdx+dim*ncoarse+3;
        }
      }
    }

    int end2 = ncoarse-1;
    int end3 = dim*ncoarse;
    for(int i=1; i<end2; i++) { // loop over blocks
      for(int j=0; j<end3; j=j+2) { // loop over inside
        int rowIdx = i*dim*ncoarse+j;
        if(j == 0) {
          for(int ndof=0; ndof<dim; ndof++) {
            int k = rowptr[rowIdx+ndof];
            rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+12;
            colind[k]    = rowIdx-dim*ncoarse;
            colind[k+1]  = rowIdx-dim*ncoarse+1;
            colind[k+2]  = rowIdx-dim*ncoarse+2;
            colind[k+3]  = rowIdx-dim*ncoarse+3;
            colind[k+4]  = rowIdx;
            colind[k+5]  = rowIdx+1;
            colind[k+6]  = rowIdx+2;
            colind[k+7]  = rowIdx+3;
            colind[k+8]  = rowIdx+dim*ncoarse;
            colind[k+9]  = rowIdx+dim*ncoarse+1;
            colind[k+10] = rowIdx+dim*ncoarse+2;
            colind[k+11] = rowIdx+dim*ncoarse+3;            
          }
        }
        else if(j == if1) {
          for(int ndof=0; ndof<dim; ndof++) {
            int k = rowptr[rowIdx+ndof];
            rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+12;
            colind[k]    = rowIdx-dim*ncoarse-2;
            colind[k+1]  = rowIdx-dim*ncoarse-1;
            colind[k+2]  = rowIdx-dim*ncoarse;
            colind[k+3]  = rowIdx-dim*ncoarse+1;
            colind[k+4]  = rowIdx-2;
            colind[k+5]  = rowIdx-1;
            colind[k+6]  = rowIdx;
            colind[k+7]  = rowIdx+1;
            colind[k+8]  = rowIdx+dim*ncoarse-2;
            colind[k+9]  = rowIdx+dim*ncoarse-1;
            colind[k+10] = rowIdx+dim*ncoarse;
            colind[k+11] = rowIdx+dim*ncoarse+1;            
          }
        } 
        else {
          for(int ndof=0; ndof<dim; ndof++) {
            int k = rowptr[rowIdx+ndof];
            rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+18;
            colind[k]    = rowIdx-dim*ncoarse-2;
            colind[k+1]  = rowIdx-dim*ncoarse-1;
            colind[k+2]  = rowIdx-dim*ncoarse;
            colind[k+3]  = rowIdx-dim*ncoarse+1;
            colind[k+4]  = rowIdx-dim*ncoarse+2;
            colind[k+5]  = rowIdx-dim*ncoarse+3;
            colind[k+6]  = rowIdx-2;
            colind[k+7]  = rowIdx-1;
            colind[k+8]  = rowIdx;
            colind[k+9]  = rowIdx+1;
            colind[k+10] = rowIdx+2;
            colind[k+11] = rowIdx+3;
            colind[k+12] = rowIdx+dim*ncoarse-2;
            colind[k+13] = rowIdx+dim*ncoarse-1;
            colind[k+14] = rowIdx+dim*ncoarse;
            colind[k+15] = rowIdx+dim*ncoarse+1;
            colind[k+16] = rowIdx+dim*ncoarse+2;
            colind[k+17] = rowIdx+dim*ncoarse+3;
          }
        }
      }
    }

    int end4 = dim*ncoarse*ncoarse;
    int if2  = dim*ncoarse*ncoarse-dim*ncoarse;
    int if3  = dim*ncoarse*ncoarse-dim;
    for(int rowIdx=dim*ncoarse*ncoarse-dim*ncoarse; rowIdx<end4; rowIdx=rowIdx+dim) {
      if(rowIdx == if2) {
        for(int ndof=0; ndof<dim; ndof++) {
          int k = rowptr[rowIdx+ndof];
          rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+8;
          colind[k]   = rowIdx-dim*ncoarse;
          colind[k+1] = rowIdx-dim*ncoarse+1;
          colind[k+2] = rowIdx-dim*ncoarse+2;
          colind[k+3] = rowIdx-dim*ncoarse+3;
          colind[k+4] = rowIdx;
          colind[k+5] = rowIdx+1;
          colind[k+6] = rowIdx+2;
          colind[k+7] = rowIdx+3;
        }
      }
      else if(rowIdx == if3) {
        for(int ndof=0; ndof<dim; ndof++) {
          int k = rowptr[rowIdx+ndof];
          rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+8;
          colind[k]   = rowIdx-dim*ncoarse-2;
          colind[k+1] = rowIdx-dim*ncoarse-1;
          colind[k+2] = rowIdx-dim*ncoarse;
          colind[k+3] = rowIdx-dim*ncoarse+1;
          colind[k+4] = rowIdx-2;
          colind[k+5] = rowIdx-1;
          colind[k+6] = rowIdx;
          colind[k+7] = rowIdx+1;         
        }
      }
      else {
        for(int ndof=0; ndof<dim; ndof++) {
          int k = rowptr[rowIdx+ndof];
          rowptr[rowIdx+ndof+1] = rowptr[rowIdx+ndof]+12;
          colind[k]    = rowIdx-dim*ncoarse-2;
          colind[k+1]  = rowIdx-dim*ncoarse-1;
          colind[k+2]  = rowIdx-dim*ncoarse;
          colind[k+3]  = rowIdx-dim*ncoarse+1;
          colind[k+4]  = rowIdx-dim*ncoarse+2;
          colind[k+5]  = rowIdx-dim*ncoarse+3;
          colind[k+6]  = rowIdx-2;
          colind[k+7]  = rowIdx-1;
          colind[k+8]  = rowIdx;
          colind[k+9]  = rowIdx+1;
          colind[k+10] = rowIdx+2;
          colind[k+11] = rowIdx+3;          
        }
      }    
    }

    std::cout << "Graph indices created!" << std::endl;
    myGraph->setAllIndices(rowptr, colind);

    std::cout << "Graph is created and filled!" << std::endl;
    myGraph->fillComplete();
 
    // build Ac with static graph pattern ...
    Ac = MatrixFactory::Build(myGraph, paramList);
  }
  
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& fineLevel, Level& coarseLevel) const {
    const bool doTranspose       = true;
    const bool doFillComplete    = true;
    const bool doOptimizeStorage = true;
    RCP<Matrix> Ac;
    {
         
      Teuchos::Array<LocalOrdinal> lCoarseNodesPerDim(3);
      lCoarseNodesPerDim = Get<Teuchos::Array<LocalOrdinal>>(fineLevel, "lCoarseNodesPerDim");
  
      FactoryMonitor m(*this, "Computing Ac", coarseLevel);
      std::ostringstream levelstr;
      levelstr << coarseLevel.GetLevelID();
      std::string labelstr = FormattingHelper::getColonLabel(coarseLevel.getObjectLabel());

      TEUCHOS_TEST_FOR_EXCEPTION(hasDeclaredInput_ == false, Exceptions::RuntimeError,
        "MueLu::RAPFactory::Build(): CallDeclareInput has not been called before Build!");

      const Teuchos::ParameterList& pL = GetParameterList();
      RCP<Matrix> A = Get< RCP<Matrix> >(fineLevel,   "A");
      RCP<Matrix> P = Get< RCP<Matrix> >(coarseLevel, "P"), AP;

      bool isEpetra = A->getRowMap()->lib() == Xpetra::UseEpetra;
#ifdef KOKKOS_ENABLE_CUDA
      bool isCuda = typeid(Node).name() == typeid(Kokkos::Compat::KokkosCudaWrapperNode).name();
#else
      bool isCuda = false;
#endif

      if (pL.get<bool>("rap: triple product") == false || isEpetra || isCuda) {
        if (pL.get<bool>("rap: triple product") && isEpetra)
          GetOStream(Warnings1) << "Switching from triple product to R x (A x P) since triple product has not been implemented for Epetra.\n";
#ifdef KOKKOS_ENABLE_CUDA
        if (pL.get<bool>("rap: triple product") && isCuda)
          GetOStream(Warnings1) << "Switching from triple product to R x (A x P) since triple product has not been implemented for Cuda.\n";
#endif

        // Reuse pattern if available (multiple solve)
        RCP<ParameterList> APparams = rcp(new ParameterList);
        if(pL.isSublist("matrixmatrix: kernel params"))
          APparams->sublist("matrixmatrix: kernel params") = pL.sublist("matrixmatrix: kernel params");

        // By default, we don't need global constants for A*P
        APparams->set("compute global constants: temporaries",APparams->get("compute global constants: temporaries",false));
        APparams->set("compute global constants",APparams->get("compute global constants",false));

        if (coarseLevel.IsAvailable("AP reuse data", this)) {
          GetOStream(static_cast<MsgType>(Runtime0 | Test)) << "Reusing previous AP data" << std::endl;

          APparams = coarseLevel.Get< RCP<ParameterList> >("AP reuse data", this);

          if (APparams->isParameter("graph"))
            AP = APparams->get< RCP<Matrix> >("graph");
        }

        {
          SubFactoryMonitor subM(*this, "MxM: A x P", coarseLevel);

          AP = MatrixMatrix::Multiply(*A, !doTranspose, *P, !doTranspose, AP, GetOStream(Statistics2),
                                      doFillComplete, doOptimizeStorage, labelstr+std::string("MueLu::A*P-")+levelstr.str(), APparams);
        }

        // Reuse coarse matrix memory if available (multiple solve)
        RCP<ParameterList> RAPparams = rcp(new ParameterList);
        if(pL.isSublist("matrixmatrix: kernel params"))
          RAPparams->sublist("matrixmatrix: kernel params") = pL.sublist("matrixmatrix: kernel params");

        if (coarseLevel.IsAvailable("RAP reuse data", this)) {
          GetOStream(static_cast<MsgType>(Runtime0 | Test)) << "Reusing previous RAP data" << std::endl;

          RAPparams = coarseLevel.Get< RCP<ParameterList> >("RAP reuse data", this);

          if (RAPparams->isParameter("graph"))
            Ac = RAPparams->get< RCP<Matrix> >("graph");

          // Some eigenvalue may have been cached with the matrix in the previous run.
          // As the matrix values will be updated, we need to reset the eigenvalue.
          Ac->SetMaxEigenvalueEstimate(-Teuchos::ScalarTraits<SC>::one());
        }

        // We *always* need global constants for the RAP, but not for the temps
        RAPparams->set("compute global constants: temporaries",RAPparams->get("compute global constants: temporaries",false));
        RAPparams->set("compute global constants",true);

        // Allow optimization of storage.
        // This is necessary for new faster Epetra MM kernels.
        // Seems to work with matrix modifications to repair diagonal entries.

        if (pL.get<bool>("transpose: use implicit") == true) {
          SubFactoryMonitor m2(*this, "MxM: P' x (AP) (implicit)", coarseLevel);

          Ac = MatrixMatrix::Multiply(*P,  doTranspose, *AP, !doTranspose, Ac, GetOStream(Statistics2),
                                      doFillComplete, doOptimizeStorage, labelstr+std::string("MueLu::R*(AP)-implicit-")+levelstr.str(), RAPparams);

        } else {
          RCP<Matrix> R = Get< RCP<Matrix> >(coarseLevel, "R");

          SubFactoryMonitor m2(*this, "MxM: R x (AP) (explicit)", coarseLevel);

          Ac = MatrixMatrix::Multiply(*R, !doTranspose, *AP, !doTranspose, Ac, GetOStream(Statistics2),
                                      doFillComplete, doOptimizeStorage, labelstr+std::string("MueLu::R*(AP)-explicit-")+levelstr.str(), RAPparams);
                                            
        }

        Teuchos::ArrayView<const double> relativeFloor = pL.get<Teuchos::Array<double> >("rap: relative diagonal floor")();
        if(relativeFloor.size() > 0) {
          Xpetra::MatrixUtils<SC,LO,GO,NO>::RelativeDiagonalBoost(Ac, relativeFloor,GetOStream(Statistics2));
        }

        bool repairZeroDiagonals = pL.get<bool>("RepairMainDiagonal") || pL.get<bool>("rap: fix zero diagonals");
        bool checkAc             = pL.get<bool>("CheckMainDiagonal")|| pL.get<bool>("rap: fix zero diagonals"); ;
        if (checkAc || repairZeroDiagonals) {
          using magnitudeType = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
          magnitudeType threshold;
          if (pL.isType<magnitudeType>("rap: fix zero diagonals threshold"))
            threshold = pL.get<magnitudeType>("rap: fix zero diagonals threshold");
          else
            threshold = Teuchos::as<magnitudeType>(pL.get<double>("rap: fix zero diagonals threshold"));
          Scalar replacement = Teuchos::as<Scalar>(pL.get<double>("rap: fix zero diagonals replacement"));
          Xpetra::MatrixUtils<SC,LO,GO,NO>::CheckRepairMainDiagonal(Ac, repairZeroDiagonals, GetOStream(Warnings1), threshold, replacement);
        }

        if (IsPrint(Statistics2)) {
          RCP<ParameterList> params = rcp(new ParameterList());;
          params->set("printLoadBalancingInfo", true);
          params->set("printCommInfo",          true);
          GetOStream(Statistics2) << PerfUtils::PrintMatrixInfo(*Ac, "Ac", params);
        }

        if(!Ac.is_null()) {std::ostringstream oss; oss << "A_" << coarseLevel.GetLevelID(); Ac->setObjectLabel(oss.str());}
        Set(coarseLevel, "A",         Ac);

        APparams->set("graph", AP);
        Set(coarseLevel, "AP reuse data",  APparams);
        RAPparams->set("graph", Ac);
        Set(coarseLevel, "RAP reuse data", RAPparams);
      
      } else {
      
      	// Here begins Ac = R x A x P as direct triple matrix product
      
        RCP<ParameterList> RAPparams = rcp(new ParameterList);
        if(pL.isSublist("matrixmatrix: kernel params"))
          RAPparams->sublist("matrixmatrix: kernel params") = pL.sublist("matrixmatrix: kernel params");

        if (coarseLevel.IsAvailable("RAP reuse data", this)) {
          GetOStream(static_cast<MsgType>(Runtime0 | Test)) << "Reusing previous RAP data" << std::endl;

          RAPparams = coarseLevel.Get< RCP<ParameterList> >("RAP reuse data", this);

          if (RAPparams->isParameter("graph"))
            Ac = RAPparams->get< RCP<Matrix> >("graph");

          // Some eigenvalue may have been cached with the matrix in the previous run.
          // As the matrix values will be updated, we need to reset the eigenvalue.
          Ac->SetMaxEigenvalueEstimate(-Teuchos::ScalarTraits<SC>::one());
        
        } else {
          // if reuse data not available, try to get sparse fill graph via the knowledge of the matrix structure
          std::string structureType = pL.get<std::string>("rap: structure type");
          if(structureType=="Laplace1D")
          {
            //std::cout << "Hello from the Laplace1D pattern determination routine!" << std::endl;
            GetLaplace1D(Ac, P, lCoarseNodesPerDim);
          }
          else if(structureType=="Laplace2D") // Here we should technically also ask for the corsening rate / interpolation order
          {
            //std::cout << "Hello from the Laplace2D pattern determination routine!" << std::endl;
            GetLaplace2D(Ac, P, lCoarseNodesPerDim);
          }                   
          else if(structureType=="Star2D")
          {
            //std::cout << "Hello from the Star2D pattern determination routine!" << std::endl;
            GetStar2D(Ac, P, lCoarseNodesPerDim);
          }
          else if(structureType=="BigStar2D")
          {
            //std::cout << "Hello from the BigStar2D pattern determination routine!" << std::endl;
            GetBigStar2D(Ac, P, lCoarseNodesPerDim);
          }
          else if(structureType=="Elasticity2D")
          {
            //std::cout << "Hello from the Elasticity2D pattern determination routine!" << std::endl;
            GetElasticity2D(Ac, P, lCoarseNodesPerDim);      
          }
        }
        
        // We *always* need global constants for the RAP, but not for the temps
        RAPparams->set("compute global constants: temporaries",RAPparams->get("compute global constants: temporaries",false));
        RAPparams->set("compute global constants",true);

        if (pL.get<bool>("transpose: use implicit") == true) {

          // why the hell is Ac build here again ?!
          //Ac = MatrixFactory::Build(P->getDomainMap(), Teuchos::as<LO>(0));

          SubFactoryMonitor m2(*this, "MxMxM: R x A x P (implicit)", coarseLevel);

          Xpetra::TripleMatrixMultiply<SC,LO,GO,NO>::
            MultiplyRAP(*P, doTranspose, *A, !doTranspose, *P, !doTranspose, *Ac, doFillComplete,
                        doOptimizeStorage, labelstr+std::string("MueLu::R*A*P-implicit-")+levelstr.str(),
                        RAPparams);
            
        } else {
          
          RCP<Matrix> R = Get< RCP<Matrix> >(coarseLevel, "R");
          
          // same here ...
          Ac = MatrixFactory::Build(R->getRowMap(), Teuchos::as<LO>(0));

          SubFactoryMonitor m2(*this, "MxMxM: R x A x P (explicit)", coarseLevel);

          Xpetra::TripleMatrixMultiply<SC,LO,GO,NO>::
            MultiplyRAP(*R, !doTranspose, *A, !doTranspose, *P, !doTranspose, *Ac, doFillComplete,
                        doOptimizeStorage, labelstr+std::string("MueLu::R*A*P-explicit-")+levelstr.str(),
                        RAPparams);
        }
      
        Teuchos::ArrayView<const double> relativeFloor = pL.get<Teuchos::Array<double> >("rap: relative diagonal floor")();
        if(relativeFloor.size() > 0) {
          Xpetra::MatrixUtils<SC,LO,GO,NO>::RelativeDiagonalBoost(Ac, relativeFloor,GetOStream(Statistics2));
        }

        bool repairZeroDiagonals = pL.get<bool>("RepairMainDiagonal") || pL.get<bool>("rap: fix zero diagonals");
        bool checkAc             = pL.get<bool>("CheckMainDiagonal")|| pL.get<bool>("rap: fix zero diagonals"); ;
        if (checkAc || repairZeroDiagonals) {
          using magnitudeType = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
          magnitudeType threshold;
          if (pL.isType<magnitudeType>("rap: fix zero diagonals threshold"))
            threshold = pL.get<magnitudeType>("rap: fix zero diagonals threshold");
          else
            threshold = Teuchos::as<magnitudeType>(pL.get<double>("rap: fix zero diagonals threshold"));
          Scalar replacement = Teuchos::as<Scalar>(pL.get<double>("rap: fix zero diagonals replacement"));
          Xpetra::MatrixUtils<SC,LO,GO,NO>::CheckRepairMainDiagonal(Ac, repairZeroDiagonals, GetOStream(Warnings1), threshold, replacement);
        }


        if (IsPrint(Statistics2)) {
          RCP<ParameterList> params = rcp(new ParameterList());;
          params->set("printLoadBalancingInfo", true);
          params->set("printCommInfo",          true);
          GetOStream(Statistics2) << PerfUtils::PrintMatrixInfo(*Ac, "Ac", params);
        }

        if(!Ac.is_null()) {std::ostringstream oss; oss << "A_" << coarseLevel.GetLevelID(); Ac->setObjectLabel(oss.str());}
        Set(coarseLevel, "A",         Ac);

        RAPparams->set("graph", Ac);
        Set(coarseLevel, "RAP reuse data", RAPparams);
      }


    }

#ifdef HAVE_MUELU_DEBUG
    MatrixUtils::checkLocalRowMapMatchesColMap(*Ac);
#endif // HAVE_MUELU_DEBUG

    if (transferFacts_.begin() != transferFacts_.end()) {
      SubFactoryMonitor m(*this, "Projections", coarseLevel);

      // call Build of all user-given transfer factories
      for (std::vector<RCP<const FactoryBase> >::const_iterator it = transferFacts_.begin(); it != transferFacts_.end(); ++it) {
        RCP<const FactoryBase> fac = *it;
        GetOStream(Runtime0) << "RAPFactory: call transfer factory: " << fac->description() << std::endl;
        fac->CallBuild(coarseLevel);
        // Coordinates transfer is marginally different from all other operations
        // because it is *optional*, and not required. For instance, we may need
        // coordinates only on level 4 if we start repartitioning from that level,
        // but we don't need them on level 1,2,3. As our current Hierarchy setup
        // assumes propagation of dependencies only through three levels, this
        // means that we need to rely on other methods to propagate optional data.
        //
        // The method currently used is through RAP transfer factories, which are
        // simply factories which are called at the end of RAP with a single goal:
        // transfer some fine data to coarser level. Because these factories are
        // kind of outside of the mainline factories, they behave different. In
        // particular, we call their Build method explicitly, rather than through
        // Get calls. This difference is significant, as the Get call is smart
        // enough to know when to release all factory dependencies, and Build is
        // dumb. This led to the following CoordinatesTransferFactory sequence:
        // 1. Request level 0
        // 2. Request level 1
        // 3. Request level 0
        // 4. Release level 0
        // 5. Release level 1
        //
        // The problem is missing "6. Release level 0". Because it was missing,
        // we had outstanding request on "Coordinates", "Aggregates" and
        // "CoarseMap" on level 0.
        //
        // This was fixed by explicitly calling Release on transfer factories in
        // RAPFactory. I am still unsure how exactly it works, but now we have
        // clear data requests for all levels.
        coarseLevel.Release(*fac);
      }
    }

  }
  
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void StructuredRAPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::AddTransferFactory(const RCP<const FactoryBase>& factory) {
    // check if it's a TwoLevelFactoryBase based transfer factory
    TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::rcp_dynamic_cast<const TwoLevelFactoryBase>(factory) == Teuchos::null, Exceptions::BadCast,
                               "MueLu::RAPFactory::AddTransferFactory: Transfer factory is not derived from TwoLevelFactoryBase. "
                               "This is very strange. (Note: you can remove this exception if there's a good reason for)");
    TEUCHOS_TEST_FOR_EXCEPTION(hasDeclaredInput_, Exceptions::RuntimeError, "MueLu::RAPFactory::AddTransferFactory: Factory is being added after we have already declared input");
    transferFacts_.push_back(factory);
  }

} //namespace MueLu

#define MUELU_STRUCTUREDRAPFACTORY_SHORT
#endif // MUELU_STRUCTUREDRAPFACTORY_DEF_HPP
