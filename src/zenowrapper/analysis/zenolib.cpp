#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) { return a + b; }

// name here must match  nanobind_add_module() in CMake
NB_MODULE(zenowrapper_ext, m) {
    nb::module_ m2 = m.def_submodule("zenolib", "A submodule of zenowrapper containing pythonic versions of zeno classes.")
    .def("add", &add);
}