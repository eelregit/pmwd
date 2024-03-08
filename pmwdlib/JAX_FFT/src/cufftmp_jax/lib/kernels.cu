#include <cstdio>
#include <memory>
#include <mutex>
#include <tuple>
#include <sstream>

#include <cufftMp.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "kernel_helpers.h"
#include "kernels.h"
#include "scaling.cuh"


#define CUDA_CHECK(ans) { cuda_check((ans), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        throw std::runtime_error("cufftmp CUDA error");
    }
}

#define NVSHMEM_CHECK(ans) { nvshmem_check((ans), __FILE__, __LINE__); }
inline void nvshmem_check(int code, const char *file, int line)
{
    if (code != 0) {
        fprintf(stderr,"NVSHMEM_CHECK: %d %s %d\n", code, file, line);
        throw std::runtime_error("cufftmp NVSHMEM error");
    }
}

#define CUFFT_CHECK(ans) { cufft_check((ans), __FILE__, __LINE__); }
inline void cufft_check(int code, const char *file, int line, bool abort=true)
{
    if (code != CUFFT_SUCCESS) {
        fprintf(stderr,"CUFFT_CHECK: %d %s %d\n", code, file, line);
        throw std::runtime_error("cufftmp CUFFT error");
    }
}

namespace cufftmp_jax {

namespace {

/**
 * Used to cache a plan accross executions
 * Planning can take a long time, and should only be done
 * once whenever possible.
 */

template <typename T>
struct plan_cache {

    const std::int64_t nx, ny, nz;
    const std::int64_t count;
    
    cufftHandle plan_r2c;
    cufftHandle plan_c2r;

    T* inout_d;
    T* scratch_d;

    plan_cache(std::int64_t nx, 
               std::int64_t ny, 
               std::int64_t nz, 
               std::int64_t count, 
               cufftHandle plan_r2c, 
               cufftHandle plan_c2r, 
               T* inout_d,
               T* scratch_d)
        : nx(nx), ny(ny), nz(nz), count(count), 
          plan_r2c(plan_r2c), plan_c2r(plan_c2r),
          inout_d(inout_d), scratch_d(scratch_d) {};

    static std::unique_ptr<plan_cache> create(std::int64_t nx, std::int64_t ny, std::int64_t nz, int size) {

        // Initialize NVSHMEM, do basic checks
        nvshmem_init();

        if(nx % size != 0 || ny % size != 0) {
            std::stringstream sstr;
            sstr << "cufftmp Invalid configuration; nx = " << nx << " and ny = " << ny << " need to be divisible by the number of PEs = " << size << "\n";
            throw std::runtime_error(sstr.str());
        }

        // Create plan #r2c
        size_t scratch_r2c = 0;
        cufftHandle plan_r2c = 0;
        CUFFT_CHECK(cufftCreate(&plan_r2c));
        CUFFT_CHECK(cufftMpAttachComm(plan_r2c, cufftMpCommType::CUFFT_COMM_NONE, nullptr));
        CUFFT_CHECK(cufftXtSetSubformatDefault(plan_r2c, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
        //CUFFT_CHECK(cufftSetAutoAllocation(plan_r2c, 0));
        if(std::is_same<float, T>::value) {
            if (nz == 1) {
              CUFFT_CHECK(cufftMakePlan2d(plan_r2c, nx, ny, CUFFT_R2C, &scratch_r2c));
            } else {
              CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &scratch_r2c));
            }
        } else {
            if (nz == 1) {
              CUFFT_CHECK(cufftMakePlan2d(plan_r2c, nx, ny, CUFFT_D2Z, &scratch_r2c));
            } else {
              CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_D2Z, &scratch_r2c));
            }        
        }

        // Create plan #c2r
        size_t scratch_c2r = 0;
        cufftHandle plan_c2r = 0;
        CUFFT_CHECK(cufftCreate(&plan_c2r));
        CUFFT_CHECK(cufftMpAttachComm(plan_c2r, cufftMpCommType::CUFFT_COMM_NONE, nullptr));
        CUFFT_CHECK(cufftXtSetSubformatDefault(plan_c2r, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE, cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
        //CUFFT_CHECK(cufftSetAutoAllocation(plan_c2r, 0));
        if(std::is_same<float, T>::value) {
            if (nz == 1) {
              CUFFT_CHECK(cufftMakePlan2d(plan_c2r, nx, ny, CUFFT_C2R, &scratch_c2r));
            } else {
              CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &scratch_c2r));
            }
        } else {
            if (nz == 1) {
              CUFFT_CHECK(cufftMakePlan2d(plan_c2r, nx, ny, CUFFT_Z2D, &scratch_c2r));
            } else {
              CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_Z2D, &scratch_c2r));
            }        
        }

        std::int64_t count = nx * ny * 2*(nz/2+1) / size;  // 2 means complex
        size_t scratch = std::max<size_t>(scratch_r2c, scratch_c2r);
        
        //printf("=======buffer_size = %d++++++++++++\n", count * sizeof(T));
        T* inout_d = (T*)nvshmem_malloc(count * sizeof(T));
        T* scratch_d = (T*)nvshmem_malloc(count * sizeof(T));
        CUDA_CHECK(cudaGetLastError());

        CUFFT_CHECK(cufftSetWorkArea(plan_r2c, scratch_d));
        CUFFT_CHECK(cufftSetWorkArea(plan_c2r, scratch_d));

        return std::make_unique<plan_cache>(nx, ny, nz, count, plan_r2c, plan_c2r, inout_d, scratch_d);
    }

    ~plan_cache() {
        // The context is already destroyed at this point, so releasing resources is pointless
        nvshmem_free(scratch_d);           
        nvshmem_free(inout_d);
        CUFFT_CHECK(cufftDestroy(plan_c2r));
        CUFFT_CHECK(cufftDestroy(plan_r2c));
        // nvshmem_finalize();
    }

};

// Prevents accidental access by multiple threads to the cache.
// Note that NVSHMEM does not support >1 GPU per process, and cuFFTMp has the same restriction.
static std::mutex plan_mtx;

template <typename T, typename U>
inline void apply_cufftmp(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {

    /**
     * Extract the parameters of the FFT
     */
    const cufftmpDescriptor &d = *UnpackDescriptor<cufftmpDescriptor>(opaque, opaque_len);
    std::int64_t nx = d.global_x;
    std::int64_t ny = d.global_y;
    std::int64_t nz = d.global_z;
    const int rank = d.rank;
    const int size = d.size;
    const int distribution = d.distribution;
    const int direction = d.direction;
    
    nx = nx * size;
    if(direction != 0) {
        nz = 2*(nz - 1);
    }

    // This cache holds a plan for a specific (nx, ny, nz) shape
    static std::unique_ptr<plan_cache<T>> cache(nullptr);

    void *input_d = reinterpret_cast<void *>(buffers[0]);
    void *output_d = reinterpret_cast<void *>(buffers[1]);

    /**
     * Create a cuFFTMp plan, or fetch one from the cache
     */
    //printf("=======nx = %d, ny=%d, nz=%d++++++++++++\n", nx, ny, nz);
    //printf("=======rank = %d, size=%d++++++++++++\n", rank, size);
    std::lock_guard<std::mutex> lock(plan_mtx);

    if(cache == nullptr) {
        cache = plan_cache<T>::create(nx, ny, nz, size);
    }else {
        if(cache->nx != nx || cache->ny != ny || cache->nz != nz) {
            //throw std::runtime_error("cufftmp Invalid sizes");
            cache.reset();      // destroy the cache
            cache = plan_cache<T>::create(nx, ny, nz, size);        // create new cache
        }
    }

    cufftHandle plan = 0;
    // CUFFT_FORWARD + CUFFT_XT_FORMAT_INPLACE
    // or CUFFT_INVERSE + CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    if(direction == 0) { 
            plan = cache->plan_r2c; 
    // Otherwise...
    } else {
            plan = cache->plan_c2r;
    }

    /**
     * Set streams
     */
    CUFFT_CHECK(cufftSetStream(plan, stream));
    //printf("------------stream = %d, rank = %d-----------\n", stream, rank);

    /**
     * Local copy: input_d --> inout_d
     * Execute the FFT in place, from and to NVSHMEM allocate memory: inout_d --> inout_d
     * Local copy: inout_d --> output_d
     */

    // Run the cuFFTMp plan in inout_d
    if(direction == 0) {
        // R2C
        // Copy input buffer in inout_d, which is NVSHMEM-allocated 设置源和目标的cudaMemcpy3DParms结构体
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr.ptr = input_d;
        copyParams.srcPtr.pitch = (nz) * sizeof(T);
        copyParams.srcPtr.xsize = nx/size;
        copyParams.srcPtr.ysize = ny;
        
        copyParams.dstPtr.ptr = cache->inout_d;
        copyParams.dstPtr.pitch = (nz+2) * sizeof(T);
        copyParams.dstPtr.xsize = nx/size;
        copyParams.dstPtr.ysize = ny;
        
        copyParams.extent.width = nz * sizeof(T);
        copyParams.extent.height = ny;
        copyParams.extent.depth = nx/size;  // 拷贝 添加 padding 部分
        copyParams.kind = cudaMemcpyDefault;
        // 执行数据拷贝
        cudaMemcpy3D(&copyParams);          
        
        if(std::is_same<float, T>::value) {
            CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)(cache->inout_d), (cufftComplex*)(cache->inout_d))); 
        }
        else {
            CUFFT_CHECK(cufftExecD2Z(plan, (cufftDoubleReal*)(cache->inout_d), (cufftDoubleComplex*)(cache->inout_d)));   // cufftDoubleReal, cufftDoubleComplex
        }
        // At this point, data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED
        auto [begin_d, end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_R2C, 
                                             rank, size, nx, ny, nz, (U*)(cache->inout_d));
        const size_t num_elements = std::distance(begin_d, end_d);
        const size_t num_threads  = 128;
        const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
        scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);        
        
        // Copy from inout_d to output buffer
        size_t buffer_size_O = cache->count * sizeof(T);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(std::is_same<float, T>::value) {
            CUDA_CHECK(cudaMemcpy( (cufftComplex*)(output_d), (cufftComplex*)(cache->inout_d), buffer_size_O, cudaMemcpyDefault));     
        }
        else {
            CUDA_CHECK(cudaMemcpy( (cufftDoubleComplex*)(output_d), (cufftDoubleComplex*)(cache->inout_d), buffer_size_O, cudaMemcpyDefault));  
        }
    }
    else {
        // C2R
        // Copy input buffer in inout_d, which is NVSHMEM-allocated 
        size_t buffer_size_B = cache->count * sizeof(T);
        //printf("=======buffer_size_B = %d++++++++++++\n", buffer_size_B);
        CUDA_CHECK(cudaMemcpy(cache->inout_d, input_d, buffer_size_B, cudaMemcpyDefault));
        
        if(std::is_same<float, T>::value) {
            CUFFT_CHECK(cufftExecC2R(plan, (cufftComplex*)(cache->inout_d), (cufftReal*)(cache->inout_d)));
        }
        else {
            CUFFT_CHECK(cufftExecZ2D(plan, (cufftDoubleComplex*)(cache->inout_d), (cufftDoubleReal*)(cache->inout_d)));   // cufftDoubleComplex, cufftDoubleReal
        }
        
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Copy from inout_d to output buffer 设置源和目标的cudaMemcpy3DParms结构体
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr.ptr = cache->inout_d;
        copyParams.srcPtr.pitch = (nz+2) * sizeof(T);
        copyParams.srcPtr.xsize = nx/size;
        copyParams.srcPtr.ysize = ny;
        
        copyParams.dstPtr.ptr = output_d;
        copyParams.dstPtr.pitch = nz * sizeof(T);
        copyParams.dstPtr.xsize = nx/size;
        copyParams.dstPtr.ysize = ny;
        
        copyParams.extent.width = nz * sizeof(T);
        copyParams.extent.height = ny;
        copyParams.extent.depth = nx/size;  // 拷贝 去除padding 部分
        copyParams.kind = cudaMemcpyDefault;
        // 执行数据拷贝
        cudaMemcpy3D(&copyParams);        
    }

}

}  // namespace

void gpu_cufftmp_f32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_cufftmp<float, cufftComplex>(stream, buffers, opaque, opaque_len);
}

void gpu_cufftmp_f64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    apply_cufftmp<double, cufftDoubleComplex>(stream, buffers, opaque, opaque_len);
}

}  // namespace cufftmp_jax
