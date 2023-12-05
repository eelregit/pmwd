#ifndef _JAX_PMWD_COMMON_H_
#define _JAX_PMWD_COMMON_H_

// This descriptor is common to both the jax_pmwd_cpu and jax_finufft_gpu modules
// We will use the jax_pmwd namespace for both

namespace jax_pmwd {

template <typename T>
struct PmwdDescriptor {
  int64_t n_particle;
  T ptcl_spacing;
  T cell_size;
  T offset[3];
  int16_t ptcl_grid[3];
  int16_t stride[3];
  size_t tmp_storage_size;
  uint32_t n_batch;
};

struct SortDescriptor {
    int64_t n_keys;
    size_t tmp_storage_size;
};

}

#endif
