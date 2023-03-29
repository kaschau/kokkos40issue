# kokkos40issue

This cases is a simple example of using pykokkos-base for python/kokkos interop with pybind11 as the glue.

To run the test case, see "run_test.bash" for a bash script.

test.py recreates the issue, wherein a compute kernel written in kokkos c++ fails with kokkos 4.0, but works fine with kokkos 3.X

My setup uses CUDA 11.7, gcc 9.4, and a NVIDIA RTX1650TI card.
