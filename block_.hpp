#ifndef __block__H__
#define __block__H__

#include "kokkosTypes.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data
// arrays for each block. Also converted into python class for modifying in the
// python wrapper
struct block_ {

  // Grid Arrays
  threeDview x, y;
};

void AEQB(threeDview &A, threeDview &B) {
  //-------------------------------------------------------------------------------------------|
  // A = B
  //-------------------------------------------------------------------------------------------|
  int indxI = A.extent(0);
  int indxJ = A.extent(1);
  int indxK = A.extent(2);
  MDRange3 range({0, 0, 0}, {indxI, indxJ, indxK});
  Kokkos::parallel_for(
      "AEQB", range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
        A(i, j, k) = B(i, j, k);
      });
}
#endif
