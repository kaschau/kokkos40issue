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

  // THREE D VIEW
  py::class_<threeDview>(m, "view3")
      .def(py::init<std::string, size_t, size_t, size_t>());

  py::class_<threeDview::HostMirror>(m, "mirror3", py::buffer_protocol())
      .def(py::init([](threeDview &view) {
        threeDview::HostMirror *mirror = new threeDview::HostMirror();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](threeDview::HostMirror &view) -> py::buffer_info {
        size_t shape[3] = {view.extent(0), view.extent(1), view.extent(2)};
        size_t stride[3] = {sizeof(double) * view.stride_0(),
                            sizeof(double) * view.stride_1(),
                            sizeof(double) * view.stride_2()};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            3,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  m.def("initialize", []() { Kokkos::initialize(); });
  m.def("finalize", []() { Kokkos::finalize(); });
}
