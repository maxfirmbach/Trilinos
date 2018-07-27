#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_FEMultiVector.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_FancyOStream.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "vtxlabel.hpp"
#include "graph.h"
void read_edge_mesh(char* filename, int &n, unsigned &m, int*& srcs, int *&dsts, int*& grounded_flags, int ground_sensitivity){
  std::ifstream infile;
  std::string line;
  infile.open(filename);
  
  //ignore first line
  std::getline(infile, line);
  
  std::getline(infile, line);
  int x = atoi(line.c_str());
  line = line.substr(line.find(" "));
  int y = atoi(line.c_str());
  line = line.substr(line.find(" ", line.find_first_not_of(" ")));
  int z = atoi(line.c_str());
  
  //initialize
  n = x;
  m = y*8;
  //z is the number of floating boundary edges
  
  srcs = new int[m];
  dsts = new int[m];
  //ignore the next x lines
  while(x-- > 0){
    std::getline(infile,line);
  }
  std::getline(infile,line);

  //create the final_ground_flags array, initially everything is floating
  int* final_ground_flags = new int[n];
  for(int i = 0; i < n; i++){
    final_ground_flags[i] = 0;
  }
  int edge_index = 0; 
  //for the next y lines
  //read in the first 4 ints
  //create 8 edges from those ints, subtracting one from all values for 0-indexing
  while(y-- > 0){
    int node1 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" "));
    int node2 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" ", line.find_first_not_of(" ")));
    int node3 = atoi(line.c_str()) - 1;
    line = line.substr(line.find(" ", line.find_first_not_of(" ")));
    int node4 = atoi(line.c_str()) - 1;

    // set the final grounding
    int grounding = grounded_flags[node1] + grounded_flags[node2] + grounded_flags[node3] + grounded_flags[node4];
    if(grounding >= ground_sensitivity){
      final_ground_flags[node1] += grounded_flags[node1];
      final_ground_flags[node2] += grounded_flags[node2];
      final_ground_flags[node3] += grounded_flags[node3];
      final_ground_flags[node4] += grounded_flags[node4];
    }

    srcs[edge_index] = node1;
    dsts[edge_index++] = node2;
    srcs[edge_index] = node2;
    dsts[edge_index++] = node1;
    srcs[edge_index] = node2;
    dsts[edge_index++] = node3;
    srcs[edge_index] = node3;
    dsts[edge_index++] = node2;
    srcs[edge_index] = node3;
    dsts[edge_index++] = node4;
    srcs[edge_index] = node4;
    dsts[edge_index++] = node3;
    srcs[edge_index] = node4;
    dsts[edge_index++] = node1;
    srcs[edge_index] = node1;
    dsts[edge_index++] = node4;
    
    std::getline(infile, line);
  }
  assert(edge_index == m);
  
  infile.close();

  //delete old grounding flags, and swap them for the new ones
  if(ground_sensitivity > 1){
    delete[] grounded_flags;
    grounded_flags = final_ground_flags;
  } else {
    delete [] final_ground_flags;
  }
  return;
}

void read_boundary_file(char* filename, int n, int*& boundary_flags){
  std::ifstream fin(filename);
  if(!fin){
    std::cout<<"Unable to open file "<<filename<<"\n";
    exit(0);
  }
  std::string throwaway;
  fin>>throwaway>>throwaway;
  int nodes, skip2, arrlength;
  fin>>nodes>>skip2>>arrlength;
  for(int i = 0; i <= nodes; i++){
    std::getline(fin, throwaway);
  }
  for(int i = 0; i < skip2; i++){
    std::getline(fin,throwaway);
  }
  boundary_flags = new int[n];
  for(int i = 0; i < n; i++){
    boundary_flags[i] = 0;
  }
  int a, b;
  //nodes that we see more than twice are potential articulation points.
  while(fin>>a>>b>>throwaway){
    boundary_flags[a-1] += 1;
    boundary_flags[b-1] += 1;
  }
}

void read_grounded_file(char* filename, int& n, int*& grounded_flags){
  std::ifstream fin(filename);
  if(!fin) {
    std::cout<<"Unable to open "<<filename<<"\n";
    exit(0);
  }
  //the first number is the number of vertices
  fin>>n;
  grounded_flags = new int[n];
  //the rest of the numbers are basal friction data
  for(int i = 0; i < n; i++){
    float gnd;
    fin>>gnd;
    grounded_flags[i] = (gnd > 0.0);
  }
}

void create_csr(int n, unsigned m, int* srcs, int* dsts, int*& out_array, unsigned*& out_degree_list, int& max_degree_vert, double& avg_out_degree){
  out_array = new int[m];
  out_degree_list = new unsigned[n+1];
  unsigned* temp_counts = new unsigned[n];
  
  for(unsigned i = 0; i < m; ++i)
    out_array[i] = 0;
  for(int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;
  for(int i = 0; i < n; ++i)
    temp_counts[i] = 0;

  for(unsigned i = 0; i < m; ++i)
    ++temp_counts[srcs[i]];
  for(int i = 0; i < n; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, n*sizeof(int));
  for(unsigned i = 0; i < m; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];
  delete [] temp_counts;

  unsigned max_degree = 0;
  max_degree_vert = -1;
  avg_out_degree = 0.0;
  for(int i = 0; i < n; ++i){
    unsigned degree = out_degree_list[i+1] - out_degree_list[i];
    avg_out_degree += (double) degree;
    if(degree > max_degree) {
      max_degree = degree;
      max_degree_vert = i;
    }
  }
  avg_out_degree /= (double)n;
  assert(max_degree_vert >= 0);
  assert(avg_out_degree >= 0.0);
}

typedef Tpetra::Map<> map_t;
typedef map_t::local_ordinal_type lno_t;
typedef map_t::global_ordinal_type gno_t;

int main(int narg, char **arg)
{
  Tpetra::ScopeGuard scope(&narg, &arg);
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int me = comm->getRank();
  int ierr = 0;

  // Test that the vtxLabel struct correctly overloads operators
  //declare all necessary variables
  int n;
  int* grounded_flags;
  int *srcs, *dsts;
  unsigned m;
  int* boundary_flags;
 
  //read in the input files, and construct the global problem instance
  if(me == 0){

    read_grounded_file(arg[3],n,grounded_flags);
    
    int ground_sensitivity = 4;
    read_edge_mesh(arg[1],n,m,srcs,dsts,grounded_flags, ground_sensitivity);
    
    read_boundary_file(arg[2],n,boundary_flags);
    std::cout<<"n = "<<n<<"\n";
  }
  
  //broadcast n and m, so other processors can initialize memory.

  Teuchos::broadcast<int,int>(*comm,0,1,&n); 
  Teuchos::broadcast<int,unsigned>(*comm,0,1,&m);
  
  //allocate memory to the arrays
  if(me != 0){
    grounded_flags = new int[n];
    srcs = new int[m];
    dsts = new int[m];
    boundary_flags = new int[n];
  }
  
  //broadcast global src/dst/grounding/boundary arrays
  Teuchos::broadcast<int,int>(*comm,0,n,grounded_flags);
  Teuchos::broadcast<int,int>(*comm,0,n,boundary_flags);
  Teuchos::broadcast<int,int>(*comm,0,m,srcs);
  Teuchos::broadcast<int,int>(*comm,0,m,dsts);

  
  //select locally owned vertices (build maps?)
  int nLocalOwned = 0;
  int np = comm->getSize();
  //the next two lines are garbage, but they work.
  nLocalOwned = n/np + (me < (n%np));
  int vtxOffset = std::min(me,n%np)*(n/np + 1) + std::max(0,me - (n%np))*(n/np);
  std::cout<<me<<" : vertices go from "<<vtxOffset<<" to "<<vtxOffset + nLocalOwned-1<<"\n";
  
  //cut the global arrays down to local instances, keeping neighbors of owned vertices in the csr,
  //note them as copies.
  
  
  int *copies = new int[n];
  int *localOwned = new int[n];
  int *newId = new int[n];
  int newIdCounter = 0;
  for(int i = 0; i < n; i++){
    copies[i] = 0;
    if(i>= vtxOffset && i < vtxOffset+nLocalOwned){
      localOwned[i] = 1;
      newId[i] = newIdCounter++; 
    } else {
      localOwned[i] = 0;
      newId[i] = -1;
    }
  } 
  
  int *localSrcs = new int[m];
  int *localDsts = new int[m];
  unsigned int localEdgeCounter = 0;
  int numcopies = 0;
   
  for(int i = 0; i < m; i++){
    if(localOwned[srcs[i]]){
      localSrcs[localEdgeCounter] = newId[srcs[i]];
      if(!localOwned[dsts[i]]){
        if(copies[dsts[i]] == 0){
          copies[dsts[i]] = 1;
          numcopies++;
        }
        if(newId[dsts[i]] < 0) newId[dsts[i]] = newIdCounter++;
      }
      localDsts[localEdgeCounter++] = newId[dsts[i]];
    } else if(localOwned[dsts[i]]){
      localDsts[localEdgeCounter] = newId[dsts[i]];
      if(!localOwned[srcs[i]]){
        if(copies[srcs[i]] == 0){
          copies[srcs[i]] = 1;
          numcopies++;
        }
        if(newId[srcs[i]] < 0) newId[srcs[i]] = newIdCounter++;
      }
      localSrcs[localEdgeCounter++] = newId[srcs[i]];
    }
  }
  
  //make new grounding/boundary arrays for just the local vertices (owned+copies)
  int* localGrounding = new int[nLocalOwned+numcopies];
  int* localBoundaries = new int[nLocalOwned+numcopies];
  for(int i = 0; i < nLocalOwned+numcopies; i++){
    localGrounding[i] = 0;
    localBoundaries[i] = 0;
  }
  for(int i = 0; i < n; i++){
    if(newId[i] > -1) {
      localGrounding[newId[i]] = grounded_flags[i];
      localBoundaries[newId[i]] = boundary_flags[i];
    }
  }
  
  //create the gids array for the maps
  Teuchos::Array<gno_t> gids(nLocalOwned + numcopies);
  int gidsCounter = 0;
  for(int i = 0; i < n; i++){
    if(localOwned[i]) {
      gids[gidsCounter++] = i;
    }
  }
  for(int i = 0; i < n; i++) {
    if(copies[i]) {
      gids[gidsCounter++] = i;
    }
  }
  Tpetra::global_size_t dummy = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  Teuchos::RCP<const map_t> mapWithCopies = rcp(new map_t(dummy,gids(),0,comm));
  Teuchos::RCP<const map_t> mapOwned = rcp(new map_t(dummy,gids(0,nLocalOwned),0,comm));
  
  
  //create local csr graph instances from the local src/dst arrays
  int* out_array;
  unsigned * out_degree_list;
  int max_degree_vert;
  double avg_out_degree;
  create_csr(newIdCounter,localEdgeCounter, localSrcs, localDsts, out_array, out_degree_list, max_degree_vert, avg_out_degree);
  
  graph* g = new graph({newIdCounter,localEdgeCounter,out_array,out_degree_list,max_degree_vert,avg_out_degree});
  //create an instance of iceProp::iceSheetPropagation
  iceProp::iceSheetPropagation prop(comm, mapOwned, mapWithCopies, g, localBoundaries, localGrounding,nLocalOwned, numcopies); 
  
  //run vtxLabel unit tests
  if (me == 0){ 
    //ierr += prop.vtxLabelUnitTest();
  }
  
  //run iceSheetPropagation::propagate  
  int lnum_removed = 0; 
  int gnum_removed = 0;
  int* remove = prop.propagate();
  for(int i = 0; i < nLocalOwned; i++){
    if(remove[i]>-2){
      lnum_removed++;
      //for(int j = 0; j < n; j++){
      //  if(newId[j] == i){
      //    std::cout<<me<<": removed vertex "<<j<<"\n";
      //  }
      //}
    }
  }

  std::cout<<me<<":\tremoved "<<lnum_removed<<" local owned vertices\n";

  return 0;
}