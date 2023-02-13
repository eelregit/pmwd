#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/zip_function.h>

#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>

//#include "cnpy.h"
#include "kernels.h"
#include "kernel_helpers.h"

namespace jax_pmwd {

#define DIM 3
#define blk 1

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                                   \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (cudaSuccess != err)                                                \
        {                                                                      \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

template <typename T, size_t N>
struct data_elm {
    T data[N];
};

typedef data_elm<int32_t,3> i4_3;
typedef data_elm<int64_t,3> i8_3;
typedef data_elm<char,8> char_8;


template <typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_bin_size(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_int1* sortidx, T_int1* bin_count){

    for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        T_int1 g_stride[3] = {stride[0], stride[1], stride[2]};

        // cell index for each dimension
        T_int1  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int1 bin_id;
        bin_id = c_index[2] * nbinx*nbiny + c_index[1] * nbiny + c_index[0];
        T_int1 oldidx = atomicAdd(&bin_count[bin_id], 1);
        sortidx[tid] = oldidx;
    }
}

template <typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_sortidx(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_int1* sortidx, T_int1* bin_start, T_int1* index){

    for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        T_int1 g_stride[3] = {stride[0], stride[1], stride[2]};

        // cell index for each dimension
        T_int1  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int1 bin_id;
        bin_id = c_index[2] * nbinx*nbiny + c_index[1] * nbiny + c_index[0];
        index[bin_start[bin_id]+sortidx[tid]] = tid;
    }
}

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
scatter_kernel_gm(T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals){
    // each thread <-> one particle 
    T_int2 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n_particle){
        // read particle data from global memory
        T_int2 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};
        T_float p_val = values[tid];

        // strides
        T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};
        T_int2 hstride[2] = {g_stride[0] * g_stride[1], g_stride[0]};

        // displacement with in a cell for cell (i,j,k)==(0,0,0)
        T_float t_disp[DIM];
        for(int idim=0; idim<3; idim++){
            t_disp[idim] = p_disp[idim]/cell_size;
            t_disp[idim] -= std::floor(t_disp[idim]);
        }

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }

        // grid value to calculate
        T_float t_val;
        T_int2 cell_id;

        // loop over all 8 vertice(cells) 
        for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        for(int k=0; k<2; k++){
            // grid value
            t_val = p_val*(1-std::abs(t_disp[0]-i))*(1-std::abs(t_disp[1]-j))*(1-std::abs(t_disp[2]-k));
            // cell_id
            cell_id = (c_index[2]+k)%g_stride[2] * hstride[0] +
                      (c_index[1]+j)%g_stride[1] * hstride[1] + (c_index[0]+i)%g_stride[0];
            // atomic write to grid values global memory
            atomicAdd(&grid_vals[cell_id], t_val);
        }
    }
}

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
scatter_kernel_sm(T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals,
                  T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int2* bin_start, T_int2* bin_count, T_int2* index){
    extern __shared__ char shared_char[];
    T_value* gval_shared = (T_value*)&shared_char[0];
    T_int1 N = (bin_size_x+1)*(bin_size_y+1)*(bin_size_z+1);
    for(int i=threadIdx.x; i<N; i+=blockDim.x){
        gval_shared[i] = 0.0;
    }
    __syncthreads();

    // each block represents a bin
    T_int2 bid = blockIdx.x;
    // strides
    T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};
    T_int2 hstride[2] = {(bin_size_x+1)*(bin_size_y+1), bin_size_x+1};
    int idx;

    int pstart = bin_start[bid];
    int npts = bin_count[bid];
    for(int i=threadIdx.x; i<npts; i+=blockDim.x){
        idx = index[pstart + i];
        T_int2 p_pmid[DIM] = {pmid[idx*DIM + 0], pmid[idx*DIM + 1], pmid[idx*DIM + 2]};
        T_float p_disp[DIM] = {disp[idx*DIM + 0], disp[idx*DIM + 1], disp[idx*DIM + 2]};
        T_float p_val = values[idx];


        // displacement with in a cell for cell (i,j,k)==(0,0,0)
        T_float t_disp[DIM];
        for(int idim=0; idim<3; idim++){
            t_disp[idim] = p_disp[idim]/cell_size;
            t_disp[idim] -= std::floor(t_disp[idim]);
        }

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }

        // grid value to calculate
        T_float t_val;
        T_int2 cell_id;

        // loop over all 8 vertice(cells) 
        for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        for(int k=0; k<2; k++){
            // grid value
            t_val = p_val*(1-std::abs(t_disp[0]-i))*(1-std::abs(t_disp[1]-j))*(1-std::abs(t_disp[2]-k));
            // cell_id
            cell_id = (c_index[2]%bin_size_z+k) * hstride[0] +
                      (c_index[1]%bin_size_y+j) * hstride[1] + (c_index[0]%bin_size_x+i);
            // atomic write to grid values shared memory
            atomicAdd(&gval_shared[cell_id], t_val);
        }
    }
    __syncthreads();
    for(int i=threadIdx.x; i<N; i+=blockDim.x){
        int ix = i%(bin_size_x+1);
        int iy = (i/(bin_size_x+1)) % (bin_size_y+1);
        int iz = i/((bin_size_x+1)*(bin_size_y+1));
        int icx = bid*bin_size_x + ix;
        int icy = bid*bin_size_y + iy;
        int icz = bid*bin_size_z + iz;

        if(icx<(g_stride[0]+1) && icy<(g_stride[1]+1) && icz<(g_stride[2]+1)){
            int outidx = icx%g_stride[0] + (icy%g_stride[1])*g_stride[0] + (icz%g_stride[2])*g_stride[0]*g_stride[1];
            atomicAdd(&grid_vals[outidx], gval_shared[i]);
        }
    }

}

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
gather_kernel_sm(T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals,
                  T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int2* bin_start, T_int2* bin_count, T_int2* index){
    // shared mem to read in grid vals
    extern __shared__ char shared_char[];
    T_value* gval_shared = (T_value*)&shared_char[0];
    // number of cells in shared mem
    T_int1 N = (bin_size_x+1)*(bin_size_y+1)*(bin_size_z+1);
    // each block represents a bin
    T_int2 bid = blockIdx.x;

    // strides
    T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};
    T_int2 hstride[2] = {(bin_size_x+1)*(bin_size_y+1), bin_size_x+1};

    for(int i=threadIdx.x; i<N; i+=blockDim.x){
        int ix = i%(bin_size_x+1);
        int iy = (i/(bin_size_x+1)) % (bin_size_y+1);
        int iz = i/((bin_size_x+1)*(bin_size_y+1));
        int icx = bid*bin_size_x + ix;
        int icy = bid*bin_size_y + iy;
        int icz = bid*bin_size_z + iz;

        if(icx<(g_stride[0]+1) && icy<(g_stride[1]+1) && icz<(g_stride[2]+1)){
            int outidx = icx%g_stride[0] + (icy%g_stride[1])*g_stride[0] + (icz%g_stride[2])*g_stride[0]*g_stride[1];
            gval_shared[i] = grid_vals[outidx];
        }
    }
    __syncthreads();

    int idx;
    int pstart = bin_start[bid];
    int npts = bin_count[bid];
    for(int i=threadIdx.x; i<npts; i+=blockDim.x){
        idx = index[pstart + i];
        T_int2 p_pmid[DIM] = {pmid[idx*DIM + 0], pmid[idx*DIM + 1], pmid[idx*DIM + 2]};
        T_float p_disp[DIM] = {disp[idx*DIM + 0], disp[idx*DIM + 1], disp[idx*DIM + 2]};

        // displacement with in a cell for cell (i,j,k)==(0,0,0)
        T_float t_disp[DIM];
        for(int idim=0; idim<3; idim++){
            t_disp[idim] = p_disp[idim]/cell_size;
            t_disp[idim] -= std::floor(t_disp[idim]);
        }

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }

        // grid value to calculate
        T_float t_val;
        T_int2 cell_id;

        // loop over all 8 vertice(cells)
        T_float pt_val=0;
        for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        for(int k=0; k<2; k++){
            // cell_id
            cell_id = (c_index[2]%bin_size_z+k) * hstride[0] +
                      (c_index[1]%bin_size_y+j) * hstride[1] + (c_index[0]%bin_size_x+i);
            t_val = gval_shared[cell_id]*(1-std::abs(t_disp[0]-i))*(1-std::abs(t_disp[1]-j))*(1-std::abs(t_disp[2]-k));

            pt_val += t_val;
        }
        // accumulate point val to its global mem
        values[idx] += pt_val;
    }
}

template <typename T>
void scatter_sm(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    // inputs/outputs
    const PmwdDescriptor<T> *descriptor = unpack_descriptor<PmwdDescriptor<T>>(opaque, opaque_len);
    T cell_size = descriptor->cell_size;
    int64_t n_particle = descriptor->n_particle;
    uint32_t stride[3]  = {descriptor->stride[0], descriptor->stride[1], descriptor->stride[2]};
    uint32_t *pmid = reinterpret_cast<uint32_t *>(buffers[0]);
    T *disp = reinterpret_cast<T *>(buffers[1]);
    T *particle_values = reinterpret_cast<T *>(buffers[2]);
    T *grid_values = reinterpret_cast<T *>(buffers[4]);
    printf("n_particle %d\n",n_particle);
    printf("cell_size %f\n",cell_size);
    printf("mesh shape %d, %d, %d\n",stride[0], stride[1], stride[2]);

    int n_elm = 10;
    thrust::device_ptr<uint32_t> dev_ptr_key = thrust::device_pointer_cast(pmid);
    for(int i=0; i<n_elm; i++){
      printf("pmid Element number %d is equal to %d\n",i,(uint32_t)*(dev_ptr_key+i));
    }

    thrust::device_ptr<T> dev_ptr_keyt1 = thrust::device_pointer_cast(disp);
    for(int i=0; i<n_elm; i++){
      printf("disp Element number %d is equal to %f\n",i,(T)*(dev_ptr_keyt1+i));
    }

    thrust::device_ptr<T> dev_ptr_keyt2 = thrust::device_pointer_cast(particle_values);
    for(int i=0; i<n_elm; i++){
      printf("val Element number %d is equal to %f\n",i,(T)*(dev_ptr_keyt2+i));
    }

    thrust::device_ptr<T> dev_ptr_keyt3 = thrust::device_pointer_cast(grid_values);
    for(int i=0; i<n_elm; i++){
      printf("mesh Element number %d is equal to %f\n",i,(T)*(dev_ptr_keyt3+i));
    }

    //printf("disp %f, %f, %f, %f, %f, %f\n",disp[0], disp[1], disp[2], disp[3], disp[4], disp[5]);
    //printf("vals %f, %f, %f, %f, %f, %f\n",particle_values[0], particle_values[1], particle_values[2], particle_values[3], particle_values[4], particle_values[5]);
    //printf("mesh %f, %f, %f, %f, %f, %f\n",grid_values[0], grid_values[1], grid_values[2], grid_values[3], grid_values[4], grid_values[5]);

    // parameters for shared mem using bins to group cells
    uint32_t bin_size = 16;
    uint32_t nbinx = stride[0]/bin_size;
    uint32_t nbiny = stride[1]/bin_size;
    uint32_t nbinz = stride[2]/bin_size;

    uint32_t* d_bin_count;
    uint32_t* d_sortidx;
    uint32_t* d_bin_start;
    uint32_t* d_index;
    uint32_t* d_stride;

    // Allocate cuda mem
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_bin_count, sizeof(uint32_t) * nbinx*nbiny*nbinz));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_sortidx, sizeof(uint32_t) * n_particle));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_bin_start, sizeof(uint32_t) * nbinx*nbiny*nbinz));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_index, sizeof(uint32_t) * n_particle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stride, sizeof(uint32_t) * DIM));

    // set device array values
    cudaMemset(d_bin_count, 0, nbinx*nbiny*nbinz*sizeof(uint32_t));
    cudaMemcpy(d_stride, stride, sizeof(uint32_t) * DIM,
               cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid_size = ((n_particle + block_size) / block_size);
    // count number of points in each bin
    cal_bin_size<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbinx, nbiny, nbinz, n_particle, pmid, disp, cell_size, d_stride, d_sortidx, d_bin_count);
    // start points of each bin
    thrust::device_ptr<uint32_t> d_ptr(d_bin_count);
    thrust::device_ptr<uint32_t> d_result(d_bin_start);
    thrust::exclusive_scan(d_ptr, d_ptr+nbinx*nbiny*nbinz, d_result);
    // calculate the index of sorted points
    cal_sortidx<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbinx, nbiny, nbinz, n_particle, pmid, disp, cell_size, d_stride, d_sortidx, d_bin_start, d_index);
    // scatter using shared memory
    cudaFuncSetAttribute(scatter_kernel_sm<uint32_t,uint32_t,T,T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    scatter_kernel_sm<<<nbinx*nbiny*nbinz, 512, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(T)>>>(pmid, disp, cell_size, d_stride, particle_values, grid_values, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_index);
    thrust::device_ptr<T> dev_ptr_keyt4 = thrust::device_pointer_cast(grid_values);
    for(int i=0; i<n_elm; i++){
      printf("mesh Element number %d is equal to %f\n",i,(T)*(dev_ptr_keyt4+i));
    }

}

template <typename T>
void gather_sm(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    // inputs/outputs
    const PmwdDescriptor<T> *descriptor = unpack_descriptor<PmwdDescriptor<T>>(opaque, opaque_len);
    T cell_size = descriptor->cell_size;
    int64_t n_particle = descriptor->n_particle;
    uint32_t stride[3]  = {descriptor->stride[0], descriptor->stride[1], descriptor->stride[2]};
    uint32_t *pmid = reinterpret_cast<uint32_t *>(buffers[0]);
    T *disp = reinterpret_cast<T *>(buffers[1]);
    T *particle_values = reinterpret_cast<T *>(buffers[4]);
    T *grid_values = reinterpret_cast<T *>(buffers[3]);

    // parameters for shared mem using bins to group cells
    uint32_t bin_size = 16;
    uint32_t nbinx = stride[0]/bin_size;
    uint32_t nbiny = stride[1]/bin_size;
    uint32_t nbinz = stride[2]/bin_size;

    uint32_t* d_bin_count;
    uint32_t* d_sortidx;
    uint32_t* d_bin_start;
    uint32_t* d_index;
    uint32_t* d_stride;

    // Allocate cuda mem
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_bin_count, sizeof(uint32_t) * nbinx*nbiny*nbinz));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_sortidx, sizeof(uint32_t) * n_particle));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_bin_start, sizeof(uint32_t) * nbinx*nbiny*nbinz));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_index, sizeof(uint32_t) * n_particle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stride, sizeof(uint32_t) * DIM));

    // set device array values
    cudaMemset(d_bin_count, 0, nbinx*nbiny*nbinz*sizeof(uint32_t));
    cudaMemcpy(d_stride, stride, sizeof(uint32_t) * DIM,
               cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid_size = ((n_particle + block_size) / block_size);
    // count number of points in each bin
    cal_bin_size<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbinx, nbiny, nbinz, n_particle, pmid, disp, cell_size, d_stride, d_sortidx, d_bin_count);
    // start points of each bin
    thrust::device_ptr<uint32_t> d_ptr(d_bin_count);
    thrust::device_ptr<uint32_t> d_result(d_bin_start);
    thrust::exclusive_scan(d_ptr, d_ptr+nbinx*nbiny*nbinz, d_result);
    // calculate the index of sorted points
    cal_sortidx<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbinx, nbiny, nbinz, n_particle, pmid, disp, cell_size, d_stride, d_sortidx, d_bin_start, d_index);
    // gather using shared memory
    cudaFuncSetAttribute(gather_kernel_sm<uint32_t,uint32_t,T,T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    gather_kernel_sm<<<nbinx*nbiny*nbinz, 512, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(T)>>>(pmid, disp, cell_size, d_stride, particle_values, grid_values, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_index);
}

void scatter(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    scatter_sm<double>(stream, buffers, opaque, opaque_len);
}

void scatterf(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    scatter_sm<float>(stream, buffers, opaque, opaque_len);
}

void gather(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    gather_sm<double>(stream, buffers, opaque, opaque_len);
}

void gatherf(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    gather_sm<float>(stream, buffers, opaque, opaque_len);
}

__global__ void
pp_gm(uint32_t* cell_ids, uint32_t n_cell, uint32_t* pos,
                     float* disp, uint32_t* particle_ids, uint32_t n_particle,
                     int32_t* neighcell_offset, uint32_t n_neigh,
                     uint32_t* stride, float cell_size, float box_size,
                     float* force)
{
    // target particle id
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // source particle id
    uint32_t sid = 0;
    if (tid < n_particle)
    {
        // source - target
        float dx[3];
        // target original postion in multiple of cell_size
        uint32_t tpos[3] = {pos[tid * DIM + 0], pos[tid * DIM + 1],
                            pos[tid * DIM + 2]};
        // displacement
        float tdisp[3] = {disp[tid * DIM + 0], disp[tid * DIM + 1],
                          disp[tid * DIM + 2]};
        // force on target due to all near source particles
        float t_force[3] = {0.0f, 0.0f, 0.0f};
        // target particle cell id
        uint32_t tcell_id = particle_ids[tid];
        // given cell index (ix, iy, iz), cell id is ix + iy*hstride[1] +
        // iz*hstride[0] stride[i]: dimention i number of cells
        uint32_t hstride[2] = {stride[0] * stride[1], stride[0]};
        // compute target cell index
        uint32_t tc_index[3] = {tcell_id % hstride[1], 0,
                                tcell_id / hstride[0]};
        tc_index[1] = (tcell_id - tc_index[2] * hstride[0]) / hstride[1];
        // loop over neighbor cells
        // uint32_t s_cnt = 0;
        for (uint32_t i_sc = 0; i_sc < n_neigh; i_sc++)
        {
            // compute neighbor cell index
            uint32_t sc_index[3];
            for (uint32_t idim = 0; idim < DIM; idim++)
            {
                // periodic mod by stride[idim]
                sc_index[idim] =
                    (tc_index[idim] + neighcell_offset[i_sc * DIM + idim] +
                     stride[idim]) %
                    stride[idim];
            }
            // compute neighbor cell id
            uint32_t scell_id = sc_index[2] * hstride[0] +
                                sc_index[1] * hstride[1] + sc_index[0];
            // loop over all source particle in cell scell_id
            for (sid = cell_ids[2 * scell_id]; sid < cell_ids[2 * scell_id + 1];
                 sid++)
            {
                if (sid != tid)
                {
                    // printf("sid: %u, tid: %u\n",sid,tid);
                    // s_cnt += 1;
                    // compute dx, abs(source - target)
                    for (uint32_t idim = 0; idim < DIM; idim++)
                    {
                        dx[idim] =
                            fabsf((int32_t(pos[sid * DIM + idim]) -
                                   int32_t(tpos[idim])) *
                                      cell_size +
                                  (disp[sid * DIM + idim] - tdisp[idim]));
                        // printf("dx: %f\n",dx[idim]);
                        // periodic test
                        while (dx[idim] > box_size * 0.5)
                        {
                            // printf("dx: %f\n",dx[idim]);
                            dx[idim] -= box_size;
                        }
                    }
                    // distance square
                    float r2 = dx[0] * dx[0] + dx[1] * dx[1] * dx[2] * dx[2];
                    // accumulate force from source particle sid
                    for (uint32_t idim = 0; idim < DIM; idim++)
                    {
                        t_force[idim] += r2 + sqrtf(r2) + rsqrtf(r2 * 1000.0);
                    }
                }
            }
        }
        // set force on target due to all neighbor source particles
        for (uint32_t idim = 0; idim < DIM; idim++)
        {
            // todo which faster
            // 1.
            force[tid * DIM + idim] += t_force[idim];
            // 2.
            // t_force read from force at the begin
            // then
            // force[tid * DIM + idim] = t_force[idim];
        }
        // printf("tid %u, total source particle %u\n",tid,s_cnt);
    }
}

__global__ void
pp_sm(uint32_t* cell_ids, uint32_t n_cell, uint32_t* pos,
                     float* disp, uint32_t* particle_ids, uint32_t n_particle,
                     int32_t* neighcell_offset, uint32_t n_neigh,
                     uint32_t* stride, float cell_size, float box_size,
                     float* force)
{
}

/*
int test()
{
    float* force;
    float* disp;
    uint32_t* pos;
    uint32_t* cell_ids;
    uint32_t* particle_ids;
    uint32_t* stride;
    int32_t* neighcell_offset;

    float* d_force;
    float* d_value;
    float* d_grid_val;
    float* d_disp;
    uint32_t* d_pos;
    uint32_t* d_cell_ids;
    uint32_t* d_particle_ids;
    uint32_t* d_stride;
    int32_t* d_neighcell_offset;
    uint32_t* d_bin_count;
    uint32_t* d_sortidx;
    uint32_t* d_bin_start;
    uint32_t* d_index;

    uint32_t n_particle;
    uint32_t n_cell;
    float box_size, cell_size;
    // uint32_t np_per_cell = 64;
    uint32_t n_neigh = 27;

    uint32_t cell_stride = 768;
    uint32_t particle_stride = 512;
    uint32_t bin_size = 16;
    uint32_t nbin = cell_stride/bin_size;

    // load test data
    const cnpy::NpyArray pmid_arr = cnpy::npy_load("short/1_768_512_pmid.npy");
    const int16_t* pmid_data = pmid_arr.data<int16_t>();
    const cnpy::NpyArray counts_arr =
        cnpy::npy_load("short/1_768_512_counts.npy");
    const int16_t* counts_data = counts_arr.data<int16_t>();
    const cnpy::NpyArray disp_arr = cnpy::npy_load("short/1_768_512_disp.npy");
    const float* disp_data = disp_arr.data<float>();

    // Yin's example
    stride = (uint32_t*)malloc(sizeof(uint32_t) * DIM);
    stride[0] = cell_stride, stride[1] = cell_stride, stride[2] = cell_stride;
    n_cell = stride[0] * stride[1] * stride[2];
    n_particle = particle_stride * particle_stride * particle_stride;
    cell_size = 1.0f;
    box_size = cell_size * stride[0];

    // Allocate host memory
    force = (float*)malloc(sizeof(float) * n_particle * DIM);
    disp = (float*)malloc(sizeof(float) * n_particle * DIM);
    pos = (uint32_t*)malloc(sizeof(uint32_t) * n_particle * DIM);
    cell_ids = (uint32_t*)malloc(sizeof(uint32_t) * n_cell * 2);
    particle_ids = (uint32_t*)malloc(sizeof(uint32_t) * n_particle);
    neighcell_offset = (int32_t*)malloc(sizeof(int32_t) * DIM * n_neigh);

    // load data
    uint32_t ip = 0;
    uint32_t cstart = 0;
    uint32_t cend = 0;
    for (uint32_t ic = 0; ic < n_cell; ic++)
    {
        uint32_t ncount = counts_data[ic];
        cend = cstart + ncount;
        cell_ids[ic * 2 + 0] = cstart;
        cell_ids[ic * 2 + 1] = cend;
        for (uint32_t icount = 0; icount < ncount; icount++)
        {
            particle_ids[ip] = ic;
            //particle_ids[ip] = 1;
            for (uint32_t idim = 0; idim < DIM; idim++)
            {
                disp[DIM * ip + idim] = disp_data[DIM * ip + idim];
                pos[DIM * ip + idim] = pmid_data[DIM * ip + idim];
                force[DIM * ip + idim] = 0.0f;
            }
            ip++;
        }
        cstart += ncount;
    }
    int32_t list[27][3];
    uint32_t icnt = 0;
    for (int ii = -1; ii < 2; ii++)
        for (int jj = -1; jj < 2; jj++)
            for (int kk = -1; kk < 2; kk++)
            {
                list[icnt][0] = ii;
                list[icnt][1] = jj;
                list[icnt][2] = kk;
                icnt++;
            }

    for (uint32_t ineigh = 0; ineigh < n_neigh; ineigh++)
    {
        for (uint32_t idim = 0; idim < DIM; idim++)
        {
            neighcell_offset[ineigh * DIM + idim] = list[ineigh][idim];
        }
    }

    // Allocate device memory
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_force, sizeof(float) * n_particle * DIM));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_value, sizeof(float) * n_particle));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_grid_val, sizeof(float) * n_cell));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_disp, sizeof(float) * n_particle * DIM));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_pos, sizeof(uint32_t) * n_particle * DIM));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_cell_ids, sizeof(uint32_t) * n_cell * 2));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_particle_ids, sizeof(uint32_t) * n_particle));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stride, sizeof(uint32_t) * DIM));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighcell_offset,
                              sizeof(int32_t) * DIM * n_neigh));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_bin_count, sizeof(uint32_t) * nbin*nbin*nbin));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_sortidx, sizeof(uint32_t) * n_particle));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_bin_start, sizeof(uint32_t) * nbin*nbin*nbin));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_index, sizeof(uint32_t) * n_particle));

    // Transfer data from host to device memory
    cudaMemcpy(d_force, force, sizeof(float) * n_particle * DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_disp, disp, sizeof(float) * n_particle * DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos, sizeof(uint32_t) * n_particle * DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_ids, cell_ids, sizeof(uint32_t) * n_cell * 2,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_ids, particle_ids, sizeof(uint32_t) * n_particle,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, stride, sizeof(uint32_t) * DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighcell_offset, neighcell_offset,
               sizeof(int32_t) * DIM * n_neigh, cudaMemcpyHostToDevice);

    // Executing kernel
    int block_size = 1024;
    int grid_size = ((n_particle + block_size) / block_size);
    thrust::device_vector<uint32_t> histogram;
    thrust::device_vector<uint32_t> histogram_values;
    thrust::device_vector<uint32_t> histogram_counts;
    thrust::device_vector<uint32_t> f1(n_particle);
    thrust::device_vector<uint32_t> f2(n_particle);

    printf("shared mem size %d\n",(bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(float));
    cudaFuncSetAttribute(scatter_kernel_sm<uint32_t,uint32_t,float,float>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    cudaFuncSetAttribute(gather_kernel_sm<uint32_t,uint32_t,float,float>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    for (int i = 0; i < 30; i++)
    {
        printf("starting cuda kernel\n");
        //printf("block size: %i, grid size: %i.\n", block_size, grid_size);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // count number of points in each bin
        cudaMemset(d_bin_count,0,nbin*nbin*nbin*sizeof(uint32_t));
        cal_bin_size<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbin, nbin, nbin, n_particle, d_pos, d_disp, cell_size, d_stride, d_sortidx, d_bin_count);
        // start points of each bin
        thrust::device_ptr<uint32_t> d_ptr(d_bin_count);
        thrust::device_ptr<uint32_t> d_result(d_bin_start);
        thrust::exclusive_scan(d_ptr, d_ptr+nbin*nbin*nbin, d_result);
        // calculate the index of sorted points
        cal_sortidx<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbin, nbin, nbin, n_particle, d_pos, d_disp, cell_size, d_stride, d_sortidx, d_bin_start, d_index);
        // scatter using shared memory
        scatter_kernel_sm<<<nbin*nbin*nbin, 512, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(float)>>>(d_pos, d_disp, cell_size, d_stride, d_value, d_grid_val, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_index);
        // gather using shared memory
        gather_kernel_sm<<<nbin*nbin*nbin, 512, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(float)>>>(d_pos, d_disp, cell_size, d_stride, d_value, d_grid_val, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_index);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaDeviceSynchronize();
        printf("cuda kernel takes: %f milliseconds\n", milliseconds);

    }

    // Transfer data back to host memory
    cudaMemcpy(force, d_force, sizeof(float) * n_particle * DIM,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(disp, d_disp, sizeof(float) * n_particle * DIM,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(pos, d_pos, sizeof(uint32_t) * n_particle * DIM,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_ids, d_cell_ids, sizeof(uint32_t) * n_cell * 2,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ids, d_particle_ids, sizeof(uint32_t) * n_particle,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(stride, d_stride, sizeof(uint32_t) * DIM,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(neighcell_offset, d_neighcell_offset,
               sizeof(int32_t) * DIM * n_neigh, cudaMemcpyDeviceToHost);


    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_force);
    cudaFree(d_value);
    cudaFree(d_grid_val);
    cudaFree(d_disp);
    cudaFree(d_pos);
    cudaFree(d_cell_ids);
    cudaFree(d_particle_ids);
    cudaFree(d_stride);
    cudaFree(d_neighcell_offset);
    cudaFree(d_bin_count);
    cudaFree(d_sortidx);
    cudaFree(d_bin_start);
    cudaFree(d_index);

    // Deallocate host memory
    free(force);
    free(disp);
    free(pos);
    free(cell_ids);
    free(particle_ids);
    free(stride);
    free(neighcell_offset);
}
*/

}
