#ifndef _JAX_PMWD_KERNELS_H_
#define _JAX_PMWD_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>


namespace jax_pmwd {

void scatter(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void gather(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

void scatterf(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void gatherf(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

}  // namespace jax_pmwd

#endif
