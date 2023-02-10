// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
//#include "jax_pmwd_gpu.h"
#include "kernels.h"

using namespace jax_pmwd;

namespace {

pybind11::dict Registrations() {
  pybind11::dict dict;

  dict["gpu_scatter_f32"] = encapsulate_function(scatterf);
  dict["gpu_gather_f32"] = encapsulate_function(gatherf);

  dict["gpu_scatter_f64"] = encapsulate_function(scatter);
  dict["gpu_gather_f64"] = encapsulate_function(gather);

  return dict;
}

PYBIND11_MODULE(_jaxpmwd, m) {
  m.def("registrations", &Registrations);
  m.def("build_pmwd_descriptor_f32", &build_descriptor<float>);
  m.def("build_pmwd_descriptor_f64", &build_descriptor<double>);
}

}  // namespace
