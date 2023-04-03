#ifndef _JAX_PMWD_KERNELS_H_
#define _JAX_PMWD_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>


namespace jax_pmwd {

int64_t get_workspace_size(int64_t n_ptcls, uint32_t stride_x, uint32_t stride_y, uint32_t stride_z, size_t& temp_storage_bytes);
void scatter(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void gather(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void scatterf(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void gatherf(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

template <typename T>
int64_t get_sort_keys_workspace_size(int64_t n_keys, size_t& temp_storage_bytes);
void sort_keys_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void sort_keys_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void sort_keys_i32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void sort_keys_i64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

template <typename T>
int64_t get_argsort_workspace_size(int64_t n_keys, size_t& temp_storage_bytes);
void argsort_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void argsort_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void argsort_i32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void argsort_i64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

int64_t get_enmesh_workspace_size(int64_t n_ptcls, uint32_t stride_x, uint32_t stride_y, uint32_t stride_z, size_t& temp_storage_bytes);
void enmesh_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void enmesh_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
}  // namespace jax_pmwd

#endif
