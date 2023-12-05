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
                                 T offset_1, T offset_2, T offset_3, int16_t ptcl_grid_1, int16_t ptcl_grid_2, int16_t ptcl_grid_3,
                                 int16_t stride_1, int16_t stride_2, int16_t stride_3, size_t tmp_storage_size, uint32_t n_batch=1) {
  return pack_descriptor(
      PmwdDescriptor<T>{n_particle, ptcl_spacing, cell_size, {offset_1, offset_2, offset_3}, {ptcl_grid_1, ptcl_grid_2, ptcl_grid_3}, {stride_1, stride_2, stride_3}, tmp_storage_size, n_batch});
}

pybind11::bytes build_sort_keys_descriptor(int64_t n_keys, size_t tmp_storage_size) {
  return pack_descriptor(SortDescriptor{n_keys, tmp_storage_size});
}

pybind11::bytes build_argsort_descriptor(int64_t n_keys, size_t tmp_storage_size) {
  return pack_descriptor(SortDescriptor{n_keys, tmp_storage_size});
}

}  // namespace jax_pmwd

#endif
