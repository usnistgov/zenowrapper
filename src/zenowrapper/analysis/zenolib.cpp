#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) { return a + b; }

// name here must match  nanobind_add_module() in CMake
NB_MODULE(zenowrapper_ext, m) {
    nb::module_ m2 = m.def_submodule("zenolib", "A submodule of zenowrapper containing pythonic versions of zeno classes.")
    .def("add", &add);
}

// self.parametersWalkOnSpheres = zenolib.parametersWalkOnSpheres()
//     self.parametersWalkOnSpheres.seTotalNumWalks(n_walks)
//     self.parametersWalkOnSpheres.setMinTotalNumWalks(min_n_walks)
//     self.parametersWalkOnSpheres.setMaxErrorCapacitance(max_rsd_capacitance)
//     self.parametersWalkOnSpheres.setMaxErrorPolarizability(max_rsd_polarizability)
//     self.parametersWalkOnSpheres.setMaxRunTime(max_run_time)
//     self.parametersWalkOnSpheres.setSeed(seed)
//     self.parametersWalkOnSpheres.setSkinThickness(skin_thickness)
//     self.parametersWalkOnSpheres.setLaunchRadius(launch_radius)