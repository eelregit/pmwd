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

  m.def("build_pmwd_descriptor_f32",
      [](int64_t n_particle, float ptcl_spacing, float cell_size,
         float offset_1, float offset_2, float offset_3, int16_t ptcl_grid_1, int16_t ptcl_grid_2, int16_t ptcl_grid_3,
         int16_t stride_1, int16_t stride_2, int16_t stride_3)
      {
          size_t tmp_space_size = 0;
          int64_t workspace_size = get_workspace_size(n_particle, stride_1, stride_2, stride_3, tmp_space_size);
          return std::pair<int64_t, pybind11::bytes>(workspace_size, build_descriptor<float>(n_particle, ptcl_spacing, cell_size, offset_1, offset_2, offset_3, ptcl_grid_1, ptcl_grid_2, ptcl_grid_3, stride_1, stride_2, stride_3, tmp_space_size));
      });

  m.def("build_pmwd_descriptor_f64",
      [](int64_t n_particle, double ptcl_spacing, double cell_size,
         double offset_1, double offset_2, double offset_3, int16_t ptcl_grid_1, int16_t ptcl_grid_2, int16_t ptcl_grid_3,
         int16_t stride_1, int16_t stride_2, int16_t stride_3)
      {
          size_t tmp_space_size = 0;
          int64_t workspace_size = get_workspace_size(n_particle, stride_1, stride_2, stride_3, tmp_space_size);
          return std::pair<int64_t, pybind11::bytes>(workspace_size, build_descriptor<double>(n_particle, ptcl_spacing, cell_size, offset_1, offset_2, offset_3, ptcl_grid_1, ptcl_grid_2, ptcl_grid_3, stride_1, stride_2, stride_3, tmp_space_size));
      });

  m.def("build_sort_keys_descriptor_f32",
      [](int64_t n_keys)
      {
          size_t tmp_space_size = 0;
          int64_t workspace_size = get_sork_keys_workspace_size<float>(n_keys, tmp_space_size);
          return std::pair<int64_t, pybind11::bytes>(workspace_size, build_sort_keys_descriptor(n_keys, tmp_space_size));
      });

  m.def("build_sort_keys_descriptor_f64",
      [](int64_t n_keys)
      {
          size_t tmp_space_size = 0;
          int64_t workspace_size = get_sork_keys_workspace_size<double>(n_keys, tmp_space_size);
          return std::pair<int64_t, pybind11::bytes>(workspace_size, build_sort_keys_descriptor(n_keys, tmp_space_size));
      });

  m.def("build_sort_keys_descriptor_i32",
      [](int64_t n_keys)
      {
          size_t tmp_space_size = 0;
          int64_t workspace_size = get_sork_keys_workspace_size<int32_t>(n_keys, tmp_space_size);
          return std::pair<int64_t, pybind11::bytes>(workspace_size, build_sort_keys_descriptor(n_keys, tmp_space_size));
      });

  m.def("build_sort_keys_descriptor_i64",
      [](int64_t n_keys)
      {
          size_t tmp_space_size = 0;
          int64_t workspace_size = get_sork_keys_workspace_size<int64_t>(n_keys, tmp_space_size);
          return std::pair<int64_t, pybind11::bytes>(workspace_size, build_sort_keys_descriptor(n_keys, tmp_space_size));
      });
}

}  // namespace
