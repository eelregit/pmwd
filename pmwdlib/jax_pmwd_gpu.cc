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

  dict["scatterf"] = encapsulate_function(scatterf);
  dict["gatherf"] = encapsulate_function(gatherf);

  dict["scatter"] = encapsulate_function(scatter);
  dict["gather"] = encapsulate_function(gather);

  return dict;
}

PYBIND11_MODULE(jax_pmwd_gpu, m) {
  m.def("registrations", &Registrations);
}

}  // namespace
