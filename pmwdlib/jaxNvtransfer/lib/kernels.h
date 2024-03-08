#ifndef _JAX_KERNELS_H_
#define _JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace nvtransfer_jax {


struct nvtransferDescriptor {
    int flags;
    int send_buf_size;
    int recv_buf_size;
    int src_rank;
    int dst_rank;
};

/**
 * Generic signature for a custom op with CUDA
 */
void gpu_nvtransfer_i16(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

void gpu_nvtransfer_i32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

void gpu_nvtransfer_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

void gpu_nvtransfer_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

}  // namespace nvtransfer_jax

#endif