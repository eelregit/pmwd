// This header extends kernel_helpers.h with the pybind11 specific interface to
// serializing descriptors. It also adds a pybind11 function for wrapping our
// custom calls in a Python capsule. This is separate from kernel_helpers so that
// the CUDA code itself doesn't include pybind11.


#ifndef _JAX_PMWD_PYBIND11_KERNEL_HELPERS_H_
#define _JAX_PMWD_PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

#include "kernel_helpers.h"

namespace jax_pmwd {

template <typename T>
pybind11::bytes pack_descriptor(const T& descriptor) {
  return pybind11::bytes(pack_descriptor_as_string(descriptor));
}

template <typename T>
pybind11::capsule encapsulate_function(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
pybind11::bytes build_descriptor(int64_t n_particle, T ptcl_spacing, T cell_size,
                                 T offset_1, T offset_2, T offset_3,
                                 int64_t stride_1, int64_t stride_2, int64_t stride_3) {
  return pack_descriptor(
      PmwdDescriptor<T>{n_particle, ptcl_spacing, cell_size, {offset_1, offset_2, offset_3}, {stride_1, stride_2, stride_3}});
}

}  // namespace jax_pmwd
