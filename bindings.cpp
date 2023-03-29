#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

void AEQB(threeDview &A, threeDview &B);

void bindUtils(py::module_ &m) {
  // ./utils
  py::module utils = m.def_submodule("utils", "utility module");
  utils.def("AEQB", &AEQB, "A = B", py::arg("A view"), py::arg("B view"));
}

PYBIND11_MODULE(compute, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";
  m.attr("KokkosLocation") = &KokkosLocation;

  bindUtils(m);
  py::class_<block_>(m, "block_", py::dynamic_attr())
      .def(py::init<>())

      //----------------------------------------------------------------------------//
      //  Primary grid node coordinates
      //----------------------------------------------------------------------------//
      .def_readwrite("x", &block_::x)
      .def_readwrite("y", &block_::y);
}
