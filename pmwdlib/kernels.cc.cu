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
#define BINSIZE 16
//#define SCATTER_DEV_TIME

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

template <typename T_int0, typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_cellid(T_int2 n_particle, T_int0* pmid, T_float* disp, T_float cell_size, T_float ptcl_spacing, T_int1 ptcl_gridx, T_int1 ptcl_gridy, T_int1 ptcl_gridz, T_int1 stridex, T_int1 stridey, T_int1 stridez, T_float offsetx, T_float offsety, T_float offsetz, T_int1* cellid, T_int1* sortidx){

    for(uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        //T_int1 p_stride[3] = {ptcl_gridx, ptcl_gridy, ptcl_gridz};
        T_int1 g_stride[3] = {stridex, stridey, stridez};
        T_float g_offset[3] = {offsetx, offsety, offsetz};

        // cell index for each dimension
        T_int1 c_index[DIM];
        T_int1 c_index2;
        T_float g_disp;
        T_float g_disp2;
        T_float ptcl_spacing_d = static_cast<T_float>(ptcl_spacing);
        T_float cell_size_d = static_cast<T_float>(cell_size);
        T_float L1[DIM] = {ptcl_spacing_d*ptcl_gridx, ptcl_spacing_d*ptcl_gridy, ptcl_spacing_d*ptcl_gridz};
        for(int idim=0; idim<3; idim++){
            g_disp = ptcl_spacing_d*p_pmid[idim]+p_disp[idim]-g_offset[idim];
            g_disp = g_disp - floor(g_disp/L1[idim])*L1[idim];
            g_disp2 = g_disp + cell_size_d;
            g_disp2 = g_disp2 - floor(g_disp2/L1[idim])*L1[idim];
            c_index[idim] = static_cast<T_int1>(floor(g_disp/cell_size_d));
            c_index2 = static_cast<T_int1>(floor(g_disp2/cell_size_d));
            // for ghost particles not in grid2 but will contribute to some vertices in grid2
            if(c_index[idim] >= g_stride[idim] && c_index2 < g_stride[idim])
                c_index[idim] = c_index2;
            // for out of grid2 particles assign random cell in grid2
            c_index[idim] = c_index[idim] % g_stride[idim];
        }

        T_int1 cell_id = c_index[0]*g_stride[2]*g_stride[1] + c_index[1]*g_stride[1] + c_index[2];
        cellid[tid] = cell_id;
        sortidx[tid] = tid;
    }

}

template <typename T_int0, typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_binid(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int0* pmid, T_float* disp, T_float cell_size, T_float ptcl_spacing, T_int1 ptcl_gridx, T_int1 ptcl_gridy, T_int1 ptcl_gridz, T_int1 stridex, T_int1 stridey, T_int1 stridez, T_float offsetx, T_float offsety, T_float offsetz, T_int1* binid, T_int1* sortidx){

    for(uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        //T_int1 p_stride[3] = {ptcl_gridx, ptcl_gridy, ptcl_gridz};
        T_int1 g_stride[3] = {stridex, stridey, stridez};
        T_float g_offset[3] = {offsetx, offsety, offsetz};

        // cell index for each dimension
        T_int1 c_index[DIM];
        T_int1 c_index2;
        double g_disp;
        double g_disp2;
        double ptcl_spacing_d = static_cast<double>(ptcl_spacing);
        double cell_size_d = static_cast<double>(cell_size);
        double L1[DIM] = {ptcl_spacing_d*ptcl_gridx, ptcl_spacing_d*ptcl_gridy, ptcl_spacing_d*ptcl_gridz};
        for(int idim=0; idim<3; idim++){
            g_disp = ptcl_spacing_d*p_pmid[idim]+p_disp[idim]-g_offset[idim];
            g_disp = g_disp - floor(g_disp/L1[idim])*L1[idim];
            g_disp2 = g_disp + cell_size_d;
            g_disp2 = g_disp2 - floor(g_disp2/L1[idim])*L1[idim];
            c_index[idim] = static_cast<T_int1>(floor(g_disp/cell_size_d));
            c_index2 = static_cast<T_int1>(floor(g_disp2/cell_size_d));
            // for ghost particles not in grid2 but will contribute to some vertices in grid2
            if(c_index[idim] >= g_stride[idim] && c_index2 < g_stride[idim])
                c_index[idim] = c_index2;
            // for out of grid2 particles assign random cell in grid2
            c_index[idim] = c_index[idim] % g_stride[idim];
        }

        // calculate binid for this particle, tricky boundary condition: particle may not belong to any of grid2's cells but will contribute to the grid vertics.
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int1 bin_id;
        bin_id = c_index[0]*nbiny*nbinz + c_index[1]*nbinz + c_index[2];
        binid[tid] = bin_id;
        sortidx[tid] = tid;
    }
}

template <typename T_int0, typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_bin_size(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int0* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_int1* sortidx, T_int1* bin_count){

    for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        T_int1 g_stride[3] = {stride[0], stride[1], stride[2]};

        // cell index for each dimension
        T_int1  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int1 bin_id;
        bin_id = c_index[0]*nbiny*nbinz + c_index[1]*nbinz + c_index[2];
        T_int1 oldidx = atomicAdd(&bin_count[bin_id], 1);
        sortidx[tid] = oldidx;
    }
}

template <typename T_int0, typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_sortidx(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int0* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_int1* sortidx, T_int1* bin_start, T_int1* index){

    for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        T_int1 g_stride[3] = {stride[0], stride[1], stride[2]};

        // cell index for each dimension
        T_int1  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int1 bin_id;
        bin_id = c_index[0]*nbiny*nbinz + c_index[1]*nbinz + c_index[2];
        index[bin_start[bin_id]+sortidx[tid]] = tid;
    }
}

template <typename T_int0, typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
scatter_kernel_gm(T_int2 n_particle, T_int0* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals){
    // each thread <-> one particle 
    T_int2 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n_particle){
        // read particle data from global memory
        T_int1 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};
        T_float p_val = values[tid];

        // strides
        T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};
        T_int2 hstride[2] = {g_stride[1] * g_stride[2], g_stride[2]};

        // displacement with in a cell for cell (i,j,k)==(0,0,0)
        T_float t_disp[DIM];
        for(int idim=0; idim<3; idim++){
            t_disp[idim] = p_disp[idim]/cell_size;
            t_disp[idim] -= floor(t_disp[idim]);
        }

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }

        // grid value to calculate
        T_float t_val;
        T_int2 cell_id;

        // loop over all 8 vertice(cells)
        for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        for(int k=0; k<2; k++){
            // grid value
            t_val = p_val*(1-abs(t_disp[0]-i))*(1-abs(t_disp[1]-j))*(1-abs(t_disp[2]-k));
            // cell_id
            cell_id = ((c_index[0]+i)%g_stride[0]) * hstride[0] +
                      ((c_index[1]+j)%g_stride[1]) * hstride[1] + (c_index[2]+k)%g_stride[2];
            // atomic write to grid values global memory
            atomicAdd(&grid_vals[cell_id], t_val);
        }
    }
}

template <typename T_int0, typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
scatter_kernel_sm(T_int0* pmid, T_float* disp, T_float cell_size, T_float ptcl_spacing, T_int1 ptcl_gridx, T_int1 ptcl_gridy, T_int1 ptcl_gridz,
                  T_int1 stridex, T_int1 stridey, T_int1 stridez, T_float offsetx, T_float offsety, T_float offsetz, T_value* values, T_value* grid_vals,
                  T_int1 nbinx, T_int1 nbiny, T_int1 nbinz,
                  T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int2* bin_start, T_int2* bin_count, T_int2* index, int64_t n_particle, T_int2 n_batch=1){
    extern __shared__ char shared_char[];
    T_value* gval_shared = (T_value*)&shared_char[0];
    T_int1 N = (bin_size_x+1)*(bin_size_y+1)*(bin_size_z+1);
    int64_t n_grid = stridex * stridey * stridez;
    for(int ibatch=0; ibatch<n_batch; ibatch++){
        for(int i=threadIdx.x; i<N; i+=blockDim.x){
            gval_shared[i] = 0.0;
        }
        __syncthreads();

        // each block represents a bin
        T_int2 bid = blockIdx.x;
        T_int1 bidx = bid/(nbiny*nbinz);
        T_int1 bidy = (bid/nbinz)%nbiny;
        T_int1 bidz = bid%nbinz;

        // strides
        double ptcl_spacing_d = static_cast<double>(ptcl_spacing);
        double cell_size_d = static_cast<double>(cell_size);
        double L1[DIM] = {ptcl_spacing_d*ptcl_gridx, ptcl_spacing_d*ptcl_gridy, ptcl_spacing_d*ptcl_gridz};
        T_int1 g_stride[3] = {stridex, stridey, stridez};
        T_float g_offset[3] = {offsetx, offsety, offsetz};
        T_int2 hstride[2] = {(bin_size_z+1)*(bin_size_y+1), bin_size_z+1};
        int idx;
        int pstart = bin_start[bid];
        int npts = bin_count[bid];
        double g_disp[DIM];
        double t_disp[DIM];
        T_int1 v_index[DIM];
        T_int1  c_index[DIM];
        T_int1  p_index[DIM];
        T_float t_val;
        T_float w_val;
        T_int2 cell_id;
        for(int i=threadIdx.x; i<npts; i+=blockDim.x){
            idx = index[pstart + i];
            T_int1 p_pmid[DIM] = {pmid[idx*DIM + 0], pmid[idx*DIM + 1], pmid[idx*DIM + 2]};
            T_float p_disp[DIM] = {disp[idx*DIM + 0], disp[idx*DIM + 1], disp[idx*DIM + 2]};
            T_float p_val = values[idx+ibatch*n_particle];

            // displacement with in a cell for cell (i,j,k)==(0,0,0)
            for(int idim=0; idim<3; idim++){
                g_disp[idim] = ptcl_spacing_d*p_pmid[idim]+p_disp[idim]-g_offset[idim];
                g_disp[idim] = g_disp[idim] - floor(g_disp[idim]/L1[idim])*L1[idim];
                p_index[idim] = static_cast<T_int1>(floor(g_disp[idim]/cell_size_d));
            }

            // loop over all 8 vertice(cells)
            for(int ii=0; ii<2; ii++)
            for(int jj=0; jj<2; jj++)
            for(int kk=0; kk<2; kk++){
                int neighbor[3] = {ii,jj,kk};
                for(int idim=0; idim<3; idim++){
                    t_disp[idim] = g_disp[idim] + neighbor[idim]*cell_size_d;
                    t_disp[idim] = t_disp[idim] - floor(t_disp[idim]/L1[idim])*L1[idim];
                    v_index[idim] = static_cast<T_int1>(floor(t_disp[idim]/cell_size_d));
                    c_index[idim] = p_index[idim];
                    if(c_index[idim] >= g_stride[idim] && v_index[idim] < g_stride[idim])
                    {
                      c_index[idim] = v_index[idim];
                      neighbor[idim] = 0;
                    }
                    c_index[idim] = c_index[idim] % g_stride[idim];
                    t_disp[idim]  = g_disp[idim] - cell_size_d*v_index[idim];
                    t_disp[idim] -= floor(t_disp[idim]/L1[idim]+0.5)*L1[idim];
                    t_disp[idim] /= cell_size_d;
                }

                w_val = 1.0;
                if(v_index[0]>=g_stride[0] ||  v_index[1]>=g_stride[1] || v_index[2]>=g_stride[2])
                    w_val = 0.0;

                // grid value
                t_val = w_val*p_val*(1-abs(t_disp[0]))*(1-abs(t_disp[1]))*(1-abs(t_disp[2]));
                // vertex_id
                cell_id = (c_index[0]%bin_size_x+neighbor[0]) * hstride[0] +
                          (c_index[1]%bin_size_y+neighbor[1]) * hstride[1] + (c_index[2]%bin_size_z+neighbor[2]);
                // atomic write to grid values shared memory
                atomicAdd(&gval_shared[cell_id], t_val);
            }
        }
        __syncthreads();
        for(int i=threadIdx.x; i<N; i+=blockDim.x){
            int ix = i/((bin_size_y+1)*(bin_size_z+1));
            int iy = (i/(bin_size_z+1)) % (bin_size_y+1);
            int iz = i%(bin_size_z+1);
            int icx = bidx*bin_size_x + ix;
            int icy = bidy*bin_size_y + iy;
            int icz = bidz*bin_size_z + iz;

            if(icx<(g_stride[0]+1) && icy<(g_stride[1]+1) && icz<(g_stride[2]+1)){ // CHECK condition
                int outidx = icz%g_stride[2] + (icy%g_stride[1])*g_stride[2] + (icx%g_stride[0])*g_stride[2]*g_stride[1];
                atomicAdd(&grid_vals[outidx+ibatch*n_grid], gval_shared[i]);
            }
        }
    }
}

template <typename T_int0, typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
gather_kernel_sm(T_int0* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals,
                  T_int1 nbinx, T_int1 nbiny, T_int1 nbinz,
                  T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int2* bin_start, T_int2* bin_count, T_int2* index){
    // shared mem to read in grid vals
    extern __shared__ char shared_char[];
    T_value* gval_shared = (T_value*)&shared_char[0];
    // number of cells in shared mem
    T_int1 N = (bin_size_x+1)*(bin_size_y+1)*(bin_size_z+1);
    // each block represents a bin
    T_int2 bid = blockIdx.x;
    T_int2 bidx = bid/(nbiny*nbinz);
    T_int2 bidy = (bid/nbinz)%nbiny;
    T_int2 bidz = bid%nbinz;

    // strides
    T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};
    T_int2 hstride[2] = {(bin_size_z+1)*(bin_size_y+1), bin_size_z+1};

    for(int i=threadIdx.x; i<N; i+=blockDim.x){
        int ix = i/((bin_size_y+1)*(bin_size_z+1));
        int iy = (i/(bin_size_z+1)) % (bin_size_y+1);
        int iz = i%(bin_size_z+1);
        int icx = bidx*bin_size_x + ix;
        int icy = bidy*bin_size_y + iy;
        int icz = bidz*bin_size_z + iz;

        if(icx<(g_stride[0]+1) && icy<(g_stride[1]+1) && icz<(g_stride[2]+1)){ // CHECK condition
            int outidx = icz%g_stride[2] + (icy%g_stride[1])*g_stride[2] + (icx%g_stride[0])*g_stride[2]*g_stride[1];
            gval_shared[i] = grid_vals[outidx];
        }
    }
    __syncthreads();

    int idx;
    int pstart = bin_start[bid];
    int npts = bin_count[bid];
    for(int i=threadIdx.x; i<npts; i+=blockDim.x){
        idx = index[pstart + i];
        T_int1 p_pmid[DIM] = {pmid[idx*DIM + 0], pmid[idx*DIM + 1], pmid[idx*DIM + 2]};
        T_float p_disp[DIM] = {disp[idx*DIM + 0], disp[idx*DIM + 1], disp[idx*DIM + 2]};

        // displacement with in a cell for cell (i,j,k)==(0,0,0)
        T_float t_disp[DIM];
        for(int idim=0; idim<3; idim++){
            t_disp[idim] = p_disp[idim]/cell_size;
            t_disp[idim] -= floor(t_disp[idim]);
        }

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
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
            cell_id = (c_index[0]%bin_size_x+i) * hstride[0] +
                      (c_index[1]%bin_size_y+j) * hstride[1] + (c_index[2]%bin_size_z+k);
            t_val = gval_shared[cell_id]*(1-abs(t_disp[0]-i))*(1-abs(t_disp[1]-j))*(1-abs(t_disp[2]-k));

            pt_val += t_val;
        }
        // accumulate point val to its global mem
        values[idx] += pt_val;
    }
}

template <typename T>
void sort_keys_kernel(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
#ifdef SCATTER_TIME
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // inputs/outputs
    const SortDescriptor *descriptor = unpack_descriptor<SortDescriptor>(opaque, opaque_len);
    int64_t n_keys = descriptor->n_keys;
    size_t temp_storage_bytes = descriptor->tmp_storage_size;
    T *d_keys = reinterpret_cast<T*>(buffers[0]);
    void *work_d = buffers[1];
    char *work_i_d = static_cast<char*>(work_d);

    int64_t keys_mem_size = sizeof(T) * n_keys;
    T* d_keys_buff = (T*)&work_i_d[0];
    void *d_temp_storage = (void*)&work_i_d[keys_mem_size];

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    cub::DoubleBuffer<T> d_keys_dbuff(d_keys, d_keys_buff);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_dbuff, n_keys);
    d_keys = d_keys_dbuff.Current();
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel Sort: %f milliseconds\n", milliseconds);
#endif
}

template <typename T>
void argsort_kernel(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
#ifdef SCATTER_TIME
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // inputs/outputs
    const SortDescriptor *descriptor = unpack_descriptor<SortDescriptor>(opaque, opaque_len);
    int64_t n_keys = descriptor->n_keys;
    size_t temp_storage_bytes = descriptor->tmp_storage_size;
    T *d_keys = reinterpret_cast<T*>(buffers[0]);
    void *work_d = buffers[1];
    char *work_i_d = static_cast<char*>(work_d);
    uint32_t *d_indices = reinterpret_cast<uint32_t*>(buffers[2]);

    int64_t keys_mem_size = sizeof(T) * n_keys;
    int64_t indices_mem_size = sizeof(uint32_t)  * n_keys;
    T* d_keys_buff = (T*)&work_i_d[0];
    uint32_t* d_indices_buff = (uint32_t*)&work_i_d[keys_mem_size];
    void *d_temp_storage = (void*)&work_i_d[keys_mem_size+indices_mem_size];

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    thrust::device_ptr<uint32_t> dev_ptr = thrust::device_pointer_cast(d_indices);
    thrust::sequence(dev_ptr, dev_ptr+n_keys);
    cub::DoubleBuffer<T> d_keys_dbuff(d_keys, d_keys_buff);
    cub::DoubleBuffer<uint32_t> d_indices_dbuff(d_indices, d_indices_buff);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_dbuff, d_indices_dbuff, n_keys);
    d_indices = d_indices_dbuff.Current();
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel Sort: %f milliseconds\n", milliseconds);
#endif
}

template <typename T>
void enmesh_kernel(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
#ifdef SCATTER_TIME
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // inputs/outputs
    const PmwdDescriptor<T> *descriptor = unpack_descriptor<PmwdDescriptor<T>>(opaque, opaque_len);
    T cell_size = descriptor->cell_size;
    T ptcl_spacing = descriptor->ptcl_spacing;
    int64_t n_particle = descriptor->n_particle;
    uint32_t ptcl_grid[3]  = {descriptor->ptcl_grid[0], descriptor->ptcl_grid[1], descriptor->ptcl_grid[2]};
    uint32_t stride[3]  = {descriptor->stride[0], descriptor->stride[1], descriptor->stride[2]};
    int64_t n_cells = stride[0] * stride[1] * stride[2];
    T offset[3]  = {descriptor->offset[0], descriptor->offset[1], descriptor->offset[2]};
    size_t   temp_storage_bytes = descriptor->tmp_storage_size;
    int16_t *pmid = reinterpret_cast<int16_t *>(buffers[0]);
    T *disp = reinterpret_cast<T *>(buffers[1]);
    T *particle_values = reinterpret_cast<T *>(buffers[2]);
    void *work_d = buffers[4];
    T *grid_values = reinterpret_cast<T *>(buffers[5]);
    char *work_i_d = static_cast<char *>(work_d);

    uint32_t npts_mem_size = sizeof(uint32_t) * n_particle;
    uint32_t ncells_mem_size = sizeof(uint32_t) * n_cells;
    uint32_t* d_sortidx = (uint32_t*)work_i_d;
    uint32_t* d_sortidx_buff = (uint32_t*)&work_i_d[npts_mem_size];
    uint32_t* d_index = (uint32_t*)&work_i_d[2*npts_mem_size];
    uint32_t* d_index_buff = (uint32_t*)&work_i_d[3*npts_mem_size];
    uint32_t* d_cell_count = (uint32_t*)&work_i_d[4*npts_mem_size];
    uint32_t* d_cell_start = (uint32_t*)&work_i_d[4*npts_mem_size + ncells_mem_size];
    void     *d_temp_storage = (void*)&work_i_d[4*npts_mem_size + 2*ncells_mem_size + sizeof(uint32_t)];
    int block_size = 1024;
    int grid_size = ((n_particle + block_size) / block_size);

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    cal_cellid<<<grid_size, block_size>>>(n_particle, pmid, disp, cell_size, ptcl_spacing, ptcl_grid[0], ptcl_grid[1], ptcl_grid[2], stride[0], stride[1], stride[2], offset[0], offset[1], offset[2], d_index, d_sortidx);
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel cal_cellid: %f milliseconds\n", milliseconds);
#endif

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    cub::DoubleBuffer<uint32_t> d_keys(d_index, d_index_buff);
    cub::DoubleBuffer<uint32_t> d_values(d_sortidx, d_sortidx_buff);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_particle);
    d_index = d_keys.Current();
    d_sortidx = d_values.Current();
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel SortPairs: %f milliseconds\n", milliseconds);
#endif

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    /*
       dense histogram ad edge using,
       histogram:
       unique_cellid, sparse_counts = cub::Encode(sorted_cellid)
       # alternatively: unique_cellid, sparse_counts =
       #   thrust::reduce_by_key(sorted_cellid)

       dense_counts = thrust::set_union_by_key(
         unique_cellid,
         thrust::counting_iterator,
         sparse_counts,
         thrust::constant_iterator<0>,
         thrust::discard_iterator,
         dense_counts,
       )
       # alternatively: dense_counts = thrust::scatter(
       #   sparse_counts, unique_cellid, dense_counts=zeros)

       dense_edges = thrust::exclusive_scan(dense_counts)

       find edges:
       unique_cellid, sparse_edges = thrust::unique_by_key_copy(
         sorted_cellid,
         thrust::counting_iterator,
         unique_cellid,
         sparse_edges,
       )
       sparse_edges[-1] = number_of_particles  # dense_edges is longer by 1

       # unnecessary here: dense_counts = thrust:adjacent_difference(dense_edges)

       # Though I don't know how to turn sparse_edges into dense_edges
       # use expand?

       code:
        int A_keys[7] = {12, 10, 8, 6, 5, 2, 0};
        int A_vals[7] = { 0,  0, 0, 0, 100, 0, 0};
        int B_keys[5] = {9, 7, 5, 3, 1};
        int B_vals[5] = {1, 1, 1, 1, 1};
        int keys_result[11];
        int vals_result[11];
        thrust::pair<int*,int*> end = thrust::set_union_by_key(thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals,B_vals, keys_result, vals_result, thrust::greater<int>());
        thrust::for_each(thrust::host, keys_result, keys_result+11,
        printf_functor());
        printf("\n");
        thrust::for_each(thrust::host, vals_result, vals_result+11,
        printf_functor());
        printf("\n");
        // sparse histogram
        uint32_t num_bins = thrust::inner_product(dev_ptr, dev_ptr + n_particle -1,
                            dev_ptr+1,
                            uint32_t(1),
                            thrust::plus<uint32_t>(),
                            thrust::not_equal_to<uint32_t>());
        histogram_values.resize(num_bins);
        histogram_counts.resize(num_bins);
        thrust::reduce_by_key(dev_ptr, dev_ptr+n_particle,
                thrust::constant_iterator<uint32_t>(1),
                histogram_values.begin(),
                histogram_counts.begin());
    */
    // dense histogram and edge using upper bound
    thrust::counting_iterator<uint32_t> search_begin(0);
    thrust::upper_bound(thrust::device_ptr<uint32_t>(d_index), thrust::device_ptr<uint32_t>(d_index)+uint32_t(n_particle),
                        search_begin, search_begin+n_cells,
                        thrust::device_ptr<uint32_t>(d_cell_start)+1);
    thrust::adjacent_difference(thrust::device_ptr<uint32_t>(d_cell_start)+1, thrust::device_ptr<uint32_t>(d_cell_start)+1+n_cells, thrust::device_ptr<uint32_t>(d_cell_count));
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel cell count: %f milliseconds\n", milliseconds);
#endif
}

template <typename T>
void scatter_sm(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
#ifdef SCATTER_TIME
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // inputs/outputs
    const PmwdDescriptor<T> *descriptor = unpack_descriptor<PmwdDescriptor<T>>(opaque, opaque_len);
    T cell_size = descriptor->cell_size;
    T ptcl_spacing = descriptor->ptcl_spacing;
    int64_t n_particle = descriptor->n_particle;
    uint32_t ptcl_grid[3]  = {descriptor->ptcl_grid[0], descriptor->ptcl_grid[1], descriptor->ptcl_grid[2]};
    uint32_t stride[3]  = {descriptor->stride[0], descriptor->stride[1], descriptor->stride[2]};
    T offset[3]  = {descriptor->offset[0], descriptor->offset[1], descriptor->offset[2]};
    size_t   temp_storage_bytes = descriptor->tmp_storage_size;
    uint32_t n_batch = descriptor->n_batch;
    int16_t *pmid = reinterpret_cast<int16_t *>(buffers[0]);
    T *disp = reinterpret_cast<T *>(buffers[1]);
    T *particle_values = reinterpret_cast<T *>(buffers[2]);
    T *grid_values = reinterpret_cast<T *>(buffers[4]);
    void *work_d = buffers[5];
    char *work_i_d = static_cast<char *>(work_d);


    // parameters for shared mem using bins to group cells
    uint32_t bin_size = BINSIZE;
    uint32_t nbinx = static_cast<uint32_t>(std::ceil(1.0*stride[0]/bin_size));
    uint32_t nbiny = static_cast<uint32_t>(std::ceil(1.0*stride[1]/bin_size));
    uint32_t nbinz = static_cast<uint32_t>(std::ceil(1.0*stride[2]/bin_size));

    int64_t npts_mem_size = sizeof(uint32_t) * (int64_t)n_particle;
    int64_t nbins_mem_size = sizeof(uint32_t) * (int64_t)nbinx*nbiny*nbinz;
    cudaMemset(work_i_d, 0, 4*npts_mem_size + 2*nbins_mem_size + sizeof(uint32_t) + temp_storage_bytes);
    uint32_t* d_sortidx = (uint32_t*)work_i_d;
    uint32_t* d_sortidx_buff = (uint32_t*)&work_i_d[npts_mem_size];
    uint32_t* d_index = (uint32_t*)&work_i_d[2*npts_mem_size];
    uint32_t* d_index_buff = (uint32_t*)&work_i_d[3*npts_mem_size];
    uint32_t* d_bin_count = (uint32_t*)&work_i_d[4*npts_mem_size];
    uint32_t* d_bin_start = (uint32_t*)&work_i_d[4*npts_mem_size + nbins_mem_size];
    void     *d_temp_storage = (void*)&work_i_d[4*npts_mem_size + 2*nbins_mem_size + sizeof(uint32_t)];
    int block_size = 1024;
    int grid_size = ((n_particle + block_size) / block_size);

#ifdef SCATTER_DEV_TIME
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int ii=0; ii<1024; ii++){
    cudaEventRecord(start);
    cal_cellid<<<grid_size, block_size>>>(n_particle, pmid, disp, cell_size, ptcl_spacing, ptcl_grid[0], ptcl_grid[1], ptcl_grid[2], stride[0], stride[1], stride[2], offset[0], offset[1], offset[2], d_index, d_sortidx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel cal_cellid: %f milliseconds\n", milliseconds);

    cudaEventRecord(start);
    cub::DoubleBuffer<uint32_t> d_keys(d_index, d_index_buff);
    cub::DoubleBuffer<uint32_t> d_values(d_sortidx, d_sortidx_buff);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_particle);
    d_index = d_keys.Current();
    d_sortidx = d_values.Current();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel SortPairs cellids: %f milliseconds\n", milliseconds);

    // TODO: change to sparse histogram, upper_bound is slow
    cudaEventRecord(start);
    thrust::counting_iterator<uint32_t> search_begin(0);
    thrust::upper_bound(thrust::device_ptr<uint32_t>(d_index), thrust::device_ptr<uint32_t>(d_index)+uint32_t(n_particle),
                        search_begin, search_begin+n_particle,
                        thrust::device_ptr<uint32_t>(d_bin_start)+1);
    thrust::adjacent_difference(thrust::device_ptr<uint32_t>(d_bin_start)+1, thrust::device_ptr<uint32_t>(d_bin_start)+1+n_particle, thrust::device_ptr<uint32_t>(d_bin_count));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel cell count: %f milliseconds\n", milliseconds);
    }
#endif

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    cal_binid<<<grid_size, block_size>>>(bin_size, bin_size, bin_size, nbinx, nbiny, nbinz, n_particle, pmid, disp, cell_size, ptcl_spacing, ptcl_grid[0], ptcl_grid[1], ptcl_grid[2], stride[0], stride[1], stride[2], offset[0], offset[1], offset[2], d_index, d_sortidx);
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel cal_binid: %f milliseconds\n", milliseconds);
#endif

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    // slower than cub
    //thrust::stable_sort_by_key(thrust::device, thrust::device_ptr<uint32_t>(d_index), thrust::device_ptr<uint32_t>(d_index)+uint32_t(n_particle), thrust::device_ptr<uint32_t>(d_sortidx));
    cub::DoubleBuffer<uint32_t> d_keys(d_index, d_index_buff);
    cub::DoubleBuffer<uint32_t> d_values(d_sortidx, d_sortidx_buff);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_particle);
    d_index = d_keys.Current();
    d_sortidx = d_values.Current();
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel SortPairs: %f milliseconds\n", milliseconds);
#endif

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    thrust::counting_iterator<uint32_t> search_begin(0);
    thrust::upper_bound(thrust::device_ptr<uint32_t>(d_index), thrust::device_ptr<uint32_t>(d_index)+uint32_t(n_particle),
                        search_begin, search_begin+nbinx*nbiny*nbinz,
                        thrust::device_ptr<uint32_t>(d_bin_start)+1);
    thrust::adjacent_difference(thrust::device_ptr<uint32_t>(d_bin_start)+1, thrust::device_ptr<uint32_t>(d_bin_start)+1+nbinx*nbiny*nbinz, thrust::device_ptr<uint32_t>(d_bin_count));
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel bin count: %f milliseconds\n", milliseconds);
#endif

#ifdef SCATTER_TIME
    cudaEventRecord(start);
#endif
    // scatter using shared memory
    cudaFuncSetAttribute(scatter_kernel_sm<int16_t, uint32_t,uint32_t,T,T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    scatter_kernel_sm<<<nbinx*nbiny*nbinz, 128, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(T)>>>(pmid, disp, cell_size, ptcl_spacing, ptcl_grid[0], ptcl_grid[1], ptcl_grid[2], stride[0], stride[1], stride[2], offset[0], offset[1], offset[2], particle_values, grid_values, nbinx, nbiny, nbinz, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_sortidx, n_particle, n_batch);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
#ifdef SCATTER_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda kernel scatter: %f milliseconds\n", milliseconds);
#endif

}

template <typename T>
void gather_sm(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    // inputs/outputs
    const PmwdDescriptor<T> *descriptor = unpack_descriptor<PmwdDescriptor<T>>(opaque, opaque_len);
    T cell_size = descriptor->cell_size;
    int64_t n_particle = descriptor->n_particle;
    uint32_t stride[3]  = {descriptor->stride[0], descriptor->stride[1], descriptor->stride[2]};
    int16_t *pmid = reinterpret_cast<int16_t *>(buffers[0]);
    T *disp = reinterpret_cast<T *>(buffers[1]);
    T *particle_values = reinterpret_cast<T *>(buffers[4]);
    T *grid_values = reinterpret_cast<T *>(buffers[3]);

    // parameters for shared mem using bins to group cells
    uint32_t bin_size = BINSIZE;
    uint32_t nbinx = static_cast<uint32_t>(std::ceil(1.0*stride[0]/bin_size));
    uint32_t nbiny = static_cast<uint32_t>(std::ceil(1.0*stride[1]/bin_size));
    uint32_t nbinz = static_cast<uint32_t>(std::ceil(1.0*stride[2]/bin_size));

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
    cudaFuncSetAttribute(gather_kernel_sm<int16_t, uint32_t,uint32_t,T,T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    gather_kernel_sm<<<nbinx*nbiny*nbinz, 512, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(T)>>>(pmid, disp, cell_size, d_stride, particle_values, grid_values, nbinx, nbiny, nbinz, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_index);
    cudaFree(d_bin_count);
    cudaFree(d_sortidx);
    cudaFree(d_bin_start);
    cudaFree(d_index);
    cudaFree(d_stride);
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

void sort_keys_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    sort_keys_kernel<float>(stream, buffers, opaque, opaque_len);
}

void sort_keys_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    sort_keys_kernel<double>(stream, buffers, opaque, opaque_len);
}

void sort_keys_i32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    sort_keys_kernel<int32_t>(stream, buffers, opaque, opaque_len);
}

void sort_keys_i64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    sort_keys_kernel<int64_t>(stream, buffers, opaque, opaque_len);
}

void argsort_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    argsort_kernel<float>(stream, buffers, opaque, opaque_len);
}

void argsort_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    argsort_kernel<double>(stream, buffers, opaque, opaque_len);
}

void argsort_i32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    argsort_kernel<int32_t>(stream, buffers, opaque, opaque_len);
}

void argsort_i64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    argsort_kernel<int64_t>(stream, buffers, opaque, opaque_len);
}

void enmesh_f32(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    enmesh_kernel<float>(stream, buffers, opaque, opaque_len);
}

void enmesh_f64(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len){
    enmesh_kernel<double>(stream, buffers, opaque, opaque_len);
}

int64_t get_workspace_size(int64_t n_ptcls, uint32_t stride_x, uint32_t stride_y, uint32_t stride_z, size_t& temp_storage_bytes){
    // get arrays storages
    // 4 arrays of n_ptcls of uint32_t
    // 1 array of nbins of uint32_t
    // 1 array of (nbins+1) of uint32_t
    int64_t bin_size = BINSIZE;
    uint32_t nbinx = static_cast<uint32_t>(std::ceil(1.0*stride_x/bin_size));
    uint32_t nbiny = static_cast<uint32_t>(std::ceil(1.0*stride_y/bin_size));
    uint32_t nbinz = static_cast<uint32_t>(std::ceil(1.0*stride_z/bin_size));
    int64_t nbins = nbinx*nbiny*nbinz;
    int64_t npts_mem_size = sizeof(uint32_t) * n_ptcls * 4;
    int64_t nbins_mem_size = sizeof(uint32_t) * (2*nbins+1);

    // get sort workspace size
    void *d_temp_storage = NULL;
    temp_storage_bytes=0;
    cub::DoubleBuffer<uint32_t> d_keys(NULL, NULL);
    cub::DoubleBuffer<uint32_t> d_values(NULL, NULL);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_ptcls);
    return temp_storage_bytes + npts_mem_size + nbins_mem_size;
}

template <typename T>
int64_t get_sort_keys_workspace_size(int64_t n_keys, size_t& temp_storage_bytes){
    int64_t nkeys_mem_size = sizeof(T) * n_keys;
    void *d_temp_storage = NULL;
    temp_storage_bytes=0;
    cub::DoubleBuffer<T> d_keys(NULL, NULL);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n_keys);
    return temp_storage_bytes + nkeys_mem_size;
}

template int64_t get_sort_keys_workspace_size<float>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_sort_keys_workspace_size<double>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_sort_keys_workspace_size<int32_t>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_sort_keys_workspace_size<int64_t>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_sort_keys_workspace_size<uint32_t>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_sort_keys_workspace_size<uint64_t>(int64_t n_keys, size_t& temp_storage_bytes);

template <typename T>
int64_t get_argsort_workspace_size(int64_t n_keys, size_t& temp_storage_bytes){
    int64_t nkeys_mem_size = sizeof(T) * n_keys + sizeof(uint32_t) * n_keys;
    void *d_temp_storage = NULL;
    temp_storage_bytes=0;
    cub::DoubleBuffer<T> d_keys(NULL, NULL);
    cub::DoubleBuffer<uint32_t> d_indices(NULL, NULL);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_indices, n_keys);
    return temp_storage_bytes + nkeys_mem_size;
}

template int64_t get_argsort_workspace_size<float>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_argsort_workspace_size<double>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_argsort_workspace_size<int32_t>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_argsort_workspace_size<int64_t>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_argsort_workspace_size<uint32_t>(int64_t n_keys, size_t& temp_storage_bytes);
template int64_t get_argsort_workspace_size<uint64_t>(int64_t n_keys, size_t& temp_storage_bytes);

int64_t get_enmesh_workspace_size(int64_t n_ptcls, uint32_t stride_x, uint32_t stride_y, uint32_t stride_z, size_t& temp_storage_bytes){
    // get arrays storages
    // 4 arrays of n_ptcls of uint32_t
    // 1 array of nbins of uint32_t
    // 1 array of (nbins+1) of uint32_t
    int64_t n_cells = stride_x*stride_y*stride_z;
    int64_t npts_mem_size = sizeof(uint32_t) * n_ptcls * 4;
    int64_t ncells_mem_size = sizeof(uint32_t) * (2*n_cells+1);

    // get sort workspace size
    void *d_temp_storage = NULL;
    temp_storage_bytes=0;
    cub::DoubleBuffer<uint32_t> d_keys(NULL, NULL);
    cub::DoubleBuffer<uint32_t> d_values(NULL, NULL);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_ptcls);
    return temp_storage_bytes + npts_mem_size + ncells_mem_size;
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
    cudaFuncSetAttribute(scatter_kernel_sm<int16_t, uint32_t,uint32_t,float,float>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    cudaFuncSetAttribute(gather_kernel_sm<int16_t, uint32_t,uint32_t,float,float>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
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
