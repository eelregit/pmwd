#include <iostream>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nccl.h>
#include <mpi.h>

#include "kernel_helpers.h"
#include "kernels.h"

#define CUDA_CHECK(ans) { cuda_check((ans), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        throw std::runtime_error("CUDA error");
    }
}

#define NVSHMEM_CHECK(ans) { nvshmem_check((ans), __FILE__, __LINE__); }
inline void nvshmem_check(int code, const char *file, int line)
{
    if (code != 0) {
        fprintf(stderr,"NVSHMEM_CHECK: %d %s %d\n", code, file, line);
        throw std::runtime_error("NVSHMEM error");
    }
}

#define NCCL_CHECK(ans) { nccl_check((ans), __FILE__, __LINE__); }
inline void nccl_check(int code, const char *file, int line, bool abort=true)
{
    if (code != ncclSuccess) {
        fprintf(stderr,"NCCL_CHECK: %d %s %d\n", code, file, line);
        throw std::runtime_error("NCCL error");
    }
}

namespace nvtransfer_jax {

namespace {
#if 1   // nvshmem
void init_nvshmem()
{
    static bool bInited = false;

    if(!bInited){
        nvshmem_init();
        bInited = true;
    }
}

// Function to perform GPU-to-GPU data transfer
void gpu_to_gpu_transfer(void *send_buf, size_t send_buf_size_bytes, void *recv_buf, size_t recv_buf_size_bytes, int src_rank, int dst_rank, cudaStream_t stream)
{
    // Initialize NVSHMEM
    init_nvshmem();

    // Determine the maximum buffer size across all ranks
    size_t max_buffer_size;
    MPI_Allreduce(&send_buf_size_bytes, &max_buffer_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Allocate receive buffer on the destination rank using NVSHMEM
    if(max_buffer_size == 0)                //  workaround for output_shape is 0, call lowering function fail
        max_buffer_size = 1;

    void* max_recv_buf = nvshmem_malloc(max_buffer_size);
    if (max_recv_buf == nullptr)
    {
        std::cerr << "Error: NVSHMEM memory allocation failed. size " << max_buffer_size << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(1/*send_buf_size_bytes > 0*/) {
        // Use NVSHMEM to transfer data  to dst_rank
        nvshmemx_putmem_nbi_on_stream(max_recv_buf, send_buf, send_buf_size_bytes, dst_rank, stream);
    }
    // Synchronize CUDA stream to ensure completion of data transfer
    cudaStreamSynchronize(stream);

    if(1/*recv_buf_size_bytes > 0*/) {
        // copy the data to jax buffer 
        cudaMemcpy(recv_buf, max_recv_buf, recv_buf_size_bytes, cudaMemcpyDeviceToDevice);
    }

    // Clean up
    if (max_recv_buf != nullptr)
    {
        nvshmem_free(max_recv_buf);
        max_recv_buf = nullptr;
    }
}
#else       // nccl
static ncclComm_t ncclCommFromMPIComm(MPI_Comm mpi_comm) {
  int rank, nranks;
  MPI_Comm_rank(mpi_comm, &rank);
  MPI_Comm_size(mpi_comm, &nranks);

  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
  ncclComm_t nccl_comm;
  ncclCommInitRank(&nccl_comm, nranks, id, rank);

  return nccl_comm;
}
// Function to perform GPU-to-GPU data transfer
void gpu_to_gpu_transfer(void *send_buf, size_t send_buf_size_bytes, void *recv_buf, size_t recv_buf_size_bytes, int src_rank, int dst_rank, cudaStream_t stream)
{
    // Initialize NCCL
    static ncclComm_t ncclComm = nullptr;
    if(!ncclComm){
        ncclComm = ncclCommFromMPIComm(MPI_COMM_WORLD);
    }

    // Perform data transfer using NCCL
    ncclGroupStart();
    ncclResult_t ncclSendResult = ncclSend(send_buf, send_buf_size_bytes, ncclUint8, dst_rank, ncclComm, stream);
    ncclResult_t ncclRecvResult = ncclRecv(recv_buf, recv_buf_size_bytes, ncclUint8, src_rank, ncclComm, stream);
    ncclGroupEnd();

    // Check for errors in NCCL operations
    if (ncclSendResult != ncclSuccess || ncclRecvResult != ncclSuccess) {
        std::cerr << "NCCL error: " << ncclGetErrorString(ncclSendResult) << ", " << ncclGetErrorString(ncclRecvResult) << std::endl;
    }    

    // Synchronize CUDA stream to ensure completion of data transfer
    cudaStreamSynchronize(stream); 
}
#endif   

template <typename T>
inline void apply_nvtransfer(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {

    /**
     * Extract the parameters of the nvtransfer
     */
    const nvtransferDescriptor &d = *UnpackDescriptor<nvtransferDescriptor>(opaque, opaque_len);
    int send_buf_size_bytes = sizeof(T) * (d.send_buf_size);
    int recv_buf_size_bytes = sizeof(T) * (d.recv_buf_size);    
    recv_buf_size_bytes -= 1;               //  workaround for output_shape is 0, call lowering function fail
    int src_rank = d.src_rank;
    int dst_rank = d.dst_rank;

    void *send_buf = reinterpret_cast<void *>(buffers[0]);
    void *recv_buf = reinterpret_cast<void *>(buffers[1]);

    gpu_to_gpu_transfer(send_buf, send_buf_size_bytes, recv_buf, recv_buf_size_bytes, src_rank, dst_rank, stream);
}

}  // namespace

void gpu_nvtransfer_i16(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_nvtransfer<int16_t>(stream, buffers, opaque, opaque_len);
}

void gpu_nvtransfer_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_nvtransfer<int32_t>(stream, buffers, opaque, opaque_len);
}

void gpu_nvtransfer_f32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_nvtransfer<float>(stream, buffers, opaque, opaque_len);
}

void gpu_nvtransfer_f64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_nvtransfer<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace nvtransfer_jax    