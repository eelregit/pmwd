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

#include "cnpy.h"

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

struct printf_functor
{
    __host__ __device__ void operator()(int x) { printf("%d,\t", x); }
};

template <typename T_k, typename T_v>
__global__ void
sort_values_by_key(T_k* keys, T_v* values, uint64_t N){
    thrust::sort_by_key(thrust::device, keys, keys + N, values);
}

template <typename T_k>
__global__ void
sort_keys(T_k* keys, uint64_t N){
    thrust::sort(thrust::device, keys, keys+N);
}

template <typename T_k>
//__global__ void
thrust::host_vector<uint64_t> sort_keys_arg(T_k* keys, uint64_t N){
    thrust::host_vector<uint64_t> index(N);
    thrust::sequence(index.begin(), index.end());
    thrust::sort_by_key(thrust::host, keys, keys+N, index.begin());
    return index;
}

template <typename T_k>
//__global__ void
thrust::device_vector<uint64_t> sort_keys_arg_device(T_k* keys, uint64_t N){
    thrust::device_vector<uint64_t> index(N);
    thrust::sequence(index.begin(), index.end());
    thrust::sort_by_key(thrust::device, keys, keys+N, index.begin());
    return index;
}

template <typename T_k, typename T_v>
__global__ void
gather_values(T_k* index, uint64_t N, T_v* values, T_v* output){
    thrust::gather(thrust::device, index, index+N, values, output);
}

template <typename T_k, typename T_v>
__global__ void
sort_values_by_keys(T_k* keys, uint64_t N, T_v* values){
    thrust::sort_by_key(thrust::device, keys, keys+N, values);
}

template <>
__global__ void
sort_values_by_keys<int32_t, i4_3>(int32_t* keys, uint64_t N, i4_3* values);

template <>
__global__ void
sort_values_by_keys<int32_t, char_8>(int32_t* keys, uint64_t N, char_8* values){
    thrust::sort_by_key(thrust::device, keys, keys+N, values);
}

template <typename T_k, typename T_v>
__global__ void
scatter_forward(){
    /*
    thrust::scatter
    thrust::sort_by_key(thrust::device, keys, keys +
                        n_particle, values);
    */
}

template <typename T_k, typename T_v>
__global__ void
scatter_reverse(){
    /*
    thrust::gather
    thrust::sort_by_key(thrust::device, keys, keys +
                        n_particle, values);
    */
}

template <typename T_k, typename T_v>
__global__ void
histogram(){
//      uint32_i num_bins = thrust::inner_product(dev_ptr, dev_ptr + n_particle -1,
//                         dev_ptr+1,
//                         uint32_t(1),
//                         thrust::plus<uint32_t>(),
//                         thrust::not_equal_to<uint32_t>());
//     histogram_values.resize(num_bins);
//     histogram_counts.resize(num_bins);
//     thrust::reduce_by_key(dev_ptr, dev_ptr+n_particle,
//             thrust::constant_iterator<uint32_t>(1),
//             histogram_values.begin(),
//             histogram_counts.begin());
//    uint32_t num_bins = thrust::inner_product(dev_ptr, dev_ptr + n_particle -1,
//                         dev_ptr+1,
//                         uint32_t(1),
//                         thrust::plus<uint32_t>(),
//                         thrust::not_equal_to<uint32_t>());
//     histogram_values.resize(num_bins);
//     histogram_counts.resize(num_bins);
//     thrust::reduce_by_key(dev_ptr, dev_ptr+n_particle,
//             thrust::constant_iterator<uint32_t>(1),
//             histogram_values.begin(),
//             histogram_counts.begin());

    /*
    thrust::sort_by_key(thrust::device, keys, keys +
                        n_particle, values);
    */
}

template <typename T_k, typename T_v>
__global__ void
bincount(){
    /*
    thrust::sort_by_key(thrust::device, keys, keys +
                        n_particle, values);
    */
}

template <typename T_int1, typename T_int2, typename T_float>
struct cell_id_functor{
    __host__ __device__
    void operator()(const data_elm<T_int1, 3>& pmid, const data_elm<T_float, 3>& disp,
                    const T_float& cell_size, const data_elm<T_int2, 3>& stride, T_int2& cell_id)
    {
        T_int1 c_index[3];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(disp.data[idim]/cell_size))%stride.data[idim]+pmid.data[idim]+stride.data[idim]) % stride.data[idim];
        }
        T_int1 hstride[2] = {stride.data[0] * stride.data[1], stride.data[0]};
        cell_id = c_index[2] * hstride[0] +
                  c_index[1] * hstride[1] + c_index[0];

    }
};

template <typename T_out>
struct scatter_functor{
    template <typename Tuple>
    __host__ __device__
    T_out operator()(Tuple t)
    {
        // tuple is value, i, j, k, disp, cell_size
        T_out val = thrust::get<0>(t);
        T_out i = thrust::get<1>(t);
        T_out j = thrust::get<2>(t);
        T_out k = thrust::get<3>(t);
        T_out disp[3];
        disp[0] =  thrust::get<4>(t).data[0];
        disp[1] =  thrust::get<4>(t).data[1];
        disp[2] =  thrust::get<4>(t).data[2];
        T_out cell_size = thrust::get<5>(t);

        T_out t_disp[3];
        for(int idim=0; idim<3; idim++){
            t_disp[idim] = disp[idim]/cell_size;
            t_disp[idim] -= std::floor(t_disp[idim]);
        }
        t_disp[0] -= i; t_disp[1] -= j; t_disp[2] -= k;

        return  val*(1-std::abs(t_disp[0]))*(1-std::abs(t_disp[1]))*(1-std::abs(t_disp[2]));
    }
};

template <typename T_int>
struct permutation_functor{
    template <typename Tuple>
    __host__ __device__
    T_int operator()(Tuple t)
    {
        // tuple is i, j, k, unique_cell_id, stride
        T_int i = thrust::get<0>(t);
        T_int j = thrust::get<1>(t);
        T_int k = thrust::get<2>(t);
        T_int unique_cell_id = thrust::get<3>(t);
        T_int stride[3];
        stride[0] = thrust::get<4>(t).data[0];
        stride[1] = thrust::get<4>(t).data[0];
        stride[2] = thrust::get<4>(t).data[0];

        // uint32_t ix, iy, iz;
        T_int hstride[2] = {stride[0] * stride[1], stride[0]};

        // compute target cell index
        T_int tc_index[3] = {unique_cell_id % hstride[1], 0,
                                unique_cell_id / hstride[0]};

        tc_index[1] = (unique_cell_id - tc_index[2] * hstride[0]) / hstride[1];

        tc_index[0] = (tc_index[0]+i)%stride[0];
        tc_index[1] = (tc_index[1]+j)%stride[1];
        tc_index[2] = (tc_index[2]+k)%stride[2];

        T_int scell_id = tc_index[2] * hstride[0] +
                            tc_index[1] * hstride[1] + tc_index[0];


        return scell_id;
    }
};

template <typename T_int1, typename T_int2, typename T_float>
struct gather_functor{
    T_float* grid_vals;

    gather_functor(T_float* grid_vals)
        : grid_vals(grid_vals) {}

    __host__ __device__
    void operator()(const data_elm<T_int1, 3*blk>& pmid, const data_elm<T_float, 3*blk>& disp,
                    const T_float& cell_size, const data_elm<T_int2, 3>& stride, data_elm<T_float, blk>& val)
    {
      /*
        for(size_t ip = 0; ip < blk; ++ip){
            val.data[ip] += 1.0;
        }
        */
    T_int1 c_index[3];
    T_int1 hstride[2] = {stride.data[0] * stride.data[1], stride.data[0]};
    T_int2 cell_id;
    T_float t_disp[3];
    T_float val_tmp  = 0.0;
    for(int ip = 0; ip<blk; ip++){
        //std::cout<<blk<<"\n";
        val_tmp = 0.0;
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<T_int2>(std::floor(disp.data[idim+ip*DIM]/cell_size)+pmid.data[idim+ip*DIM])%stride.data[idim]+stride.data[idim]) % stride.data[idim];
        }

        for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        for(int k=0; k<2; k++){
            cell_id = (c_index[2]+k)%stride.data[2] * hstride[0] +
                      (c_index[1]+j)%stride.data[1] * hstride[1] + (c_index[0]+i)%stride.data[0];

            for(int idim=0; idim<3; idim++){
                t_disp[idim] = disp.data[idim+ip*DIM]/cell_size;
                t_disp[idim] -= std::floor(t_disp[idim]);
            }
            t_disp[0] -= i; t_disp[1] -= j; t_disp[2] -= k;

            val_tmp += (*(grid_vals+cell_id)) * (1-std::abs(t_disp[0])) * (1-std::abs(t_disp[1])) * (1-std::abs(t_disp[2]));
            //val_tmp = i*j*0.3*0.1;
        }

        val.data[ip] += val_tmp;
    }
    }
};

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
void scatter_thrust(T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals){
    // todo: return cell_ids to caller, cell_ids needed for pp
    thrust::device_vector<T_int2> cell_ids(n_particle);

    data_elm<T_int1,3> domain_stride;
    domain_stride.data[0] = stride[0], domain_stride.data[1] = stride[1], domain_stride.data[2] = stride[2];

    // calculate cell_ids
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast((data_elm<T_int1, 3>*)pmid),
                                                                  thrust::device_pointer_cast((data_elm<T_float, 3>*)disp),
                                                                  thrust::constant_iterator<T_float>(cell_size),
                                                                  thrust::constant_iterator<data_elm<T_int1,3>>(domain_stride),
                                                                  cell_ids.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast((data_elm<T_int1, 3>*)pmid) + n_particle,
                                                                  thrust::device_pointer_cast((data_elm<T_float, 3>*)disp) + n_particle,
                                                                  thrust::constant_iterator<T_float>(cell_size) + n_particle,
                                                                  thrust::constant_iterator<data_elm<T_int1,3>>(domain_stride) + n_particle,
                                                                  cell_ids.end())),
                     thrust::make_zip_function(cell_id_functor<T_int1, T_int2, T_float>()));


    thrust::sort_by_key(thrust::device, cell_ids.begin(), cell_ids.end(), values);

    T_int2 num_bins = thrust::inner_product(cell_ids.begin(), cell_ids.end() - 1,
                        cell_ids.begin() + 1,
                        T_int2(1),
                        thrust::plus<T_int2>(),
                        thrust::not_equal_to<T_int2>());

    std::cout<<"num_bins: "<<num_bins<<"\n";
    thrust::device_vector<T_int2> unique_cell_ids(num_bins);

    thrust::unique_copy(thrust::device, cell_ids.begin(), cell_ids.end(), unique_cell_ids.begin());

    for(T_int1 i=0;i<2;i++)
    for(T_int1 j=0;j<2;j++)
    for(T_int1 k=0;k<2;k++)
        reduce_by_key(thrust::device, cell_ids.begin(), cell_ids.end(),
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(values),thrust::make_constant_iterator(i),thrust::make_constant_iterator(j),thrust::make_constant_iterator(k),thrust::device_pointer_cast((data_elm<T_float, 3>*)disp),thrust::make_constant_iterator(cell_size))), scatter_functor<T_float>()),
                      thrust::make_discard_iterator(),
                      thrust::make_permutation_iterator(thrust::device_pointer_cast(grid_vals), thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::make_constant_iterator(i),thrust::make_constant_iterator(j),thrust::make_constant_iterator(k),unique_cell_ids.begin(),thrust::make_constant_iterator(domain_stride))), permutation_functor<T_int2>())));
}

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
void gather_thrust(T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals){
    data_elm<T_int1,3> domain_stride;
    domain_stride.data[0] = stride[0], domain_stride.data[1] = stride[1], domain_stride.data[2] = stride[2];

    // calculate cell_ids
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast((data_elm<T_int1, 3*blk>*)pmid),
                                                                  thrust::device_pointer_cast((data_elm<T_float, 3*blk>*)disp),
                                                                  thrust::constant_iterator<T_float>(cell_size),
                                                                  thrust::constant_iterator<data_elm<T_int1,3>>(domain_stride),
                                                                  thrust::device_pointer_cast((data_elm<T_float, blk>*)values))),
                     thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast((data_elm<T_int1, 3*blk>*)pmid) + n_particle/blk,
                                                                  thrust::device_pointer_cast((data_elm<T_float, 3*blk>*)disp) + n_particle/blk,
                                                                  thrust::constant_iterator<T_float>(cell_size) + n_particle/blk,
                                                                  thrust::constant_iterator<data_elm<T_int1,3>>(domain_stride) + n_particle/blk,
                                                                  thrust::device_pointer_cast((data_elm<T_float, blk>*)values) + n_particle/blk)),
                     thrust::make_zip_function(gather_functor<T_int1, T_int2, T_float>(grid_vals)));
}

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
void scatter_cuda(T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals){
}

template <typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_bin_size(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_int2* sortidx, T_int2* bin_count){

    for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int2 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int2 bin_id;
        bin_id = c_index[2] * nbinx*nbiny + c_index[1] * nbiny + c_index[0];
        T_int2 oldidx = atomicAdd(&bin_count[bin_id], 1);
        sortidx[tid] = oldidx;
    }
}

template <typename T_int1, typename T_int2, typename T_float>
__global__ void
cal_sortidx(T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int1 nbinx, T_int1 nbiny, T_int1 nbinz, T_int2 n_particle, T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_int2* sortidx, T_int2* bin_start, T_int2* index){

    for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_particle; tid+=gridDim.x*blockDim.x){
        // read particle data from global memory
        T_int2 p_pmid[DIM] = {pmid[tid*DIM + 0], pmid[tid*DIM + 1], pmid[tid*DIM + 2]};
        T_float p_disp[DIM] = {disp[tid*DIM + 0], disp[tid*DIM + 1], disp[tid*DIM + 2]};

        // strides
        T_int2 g_stride[3] = {stride[0], stride[1], stride[2]};

        // cell index for each dimension
        T_int2  c_index[DIM];
        for(int idim=0; idim<3; idim++){
            c_index[idim] = (static_cast<int>(std::floor(p_disp[idim]/cell_size)+p_pmid[idim])%g_stride[idim]+g_stride[idim]) % g_stride[idim];
        }
        c_index[0] = c_index[0]/bin_size_x;
        c_index[1] = c_index[1]/bin_size_y;
        c_index[2] = c_index[2]/bin_size_z;

        T_int2 bin_id;
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
scatter_kernel_gmsort(){
}

template <typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
scatter_kernel_sm(T_int1* pmid, T_float* disp, T_float cell_size, T_int1* stride, T_value* values, T_value* grid_vals,
                  T_int1 bin_size_x, T_int1 bin_size_y, T_int1 bin_size_z, T_int2* bin_start, T_int2* bin_count, T_int2* index){
    extern __shared__ T_value gval_shared[];
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
    extern __shared__ T_value gval_shared[];

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

/*
template <typename T_int1, typename T_int2, typename T_float, typename T_value>
__global__ void
scatter_kernel_sm_lb(){
    extern __shared__ T_float gval_shared[];
    T_int1 N = (bin_size_x+1)*(bin_size_y+1)*(bin_size_z+1);
    for(int i=threadIdx.x; i<N; i+=blockDim.x){
        gval_shared[i] = 0.0;
    }
    __syncthreads();
    for(int i=threadIdx.x; i<npts; i+=blockDim.x){
    }
    __syncthreads();
    for(int i=threadIdx.x; i<N; i+=blockDim.x){
    }
}
*/

/*
 * @cell_ids: start and end index of sorted particle for each cell
 * @n_cell: number of cells
 * @pos: particle initial postion in multiple of cell_size
 * @disp: particle displacement in physical space
 * @particle_ids: current cell index for each particle
 * @n_particle: number of particles
 * @neighcell_offset: relative neighbor cell offset in x,y,z dim,
 * [0,0,0,0,1,0,....], size=DIM*n_neigh
 * @n_neigh: number of neighbor cells
 * @stride: number of cells in each dimension
 * @cell_size: cell size
 * @box_size: box size
 * @force: output for force on target particle due to all source particles in
 * neighbor cells note if cell index is [ix,iy,iz], cell id is
 * ix+iy*stride[0]+iz*stride[0]*stride[1]
 */
__global__ void
local_nbody_pair_sum(uint32_t* cell_ids, uint32_t n_cell, uint32_t* pos,
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

/*
 * @cell_ids: start and end index of sorted particle for each cell
 * @n_cell: number of cells
 * @pos: particle initial postion in multiple of cell_size
 * @disp: particle displacement in physical space
 * @particle_ids: current cell index for each particle
 * @n_particle: number of particles
 * @neighcell_offset: relative neighbor cell offset in x,y,z dim,
 * [0,0,0,0,1,0,....], size=DIM*n_neigh
 * @n_neigh: number of neighbor cells
 * @stride: number of cells in each dimension
 * @cell_size: cell size
 * @box_size: box size
 * @grad: output for grad on target particle due to all source particles in
 * neighbor cells note if cell index is [ix,iy,iz], cell id is
 * ix+iy*stride[0]+iz*stride[0]*stride[1]
 */
__global__ void
local_nbody_pair_sum_grad(uint32_t* cell_ids, uint32_t n_cell, uint32_t* pos,
                          float* disp, uint32_t* particle_ids,
                          uint32_t n_particle, int32_t* neighcell_offset,
                          uint32_t n_neigh, uint32_t* stride, float cell_size,
                          float box_size, float* grad)
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
        // grad on target due to all near source particles
        float t_grad[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f};
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
                    // accumulate grad from source particle sid
                    for (uint32_t idim = 0; idim < DIM; idim++)
                    {
                        for (uint32_t jdim = 0; jdim < DIM; jdim++)
                        {
                            t_grad[idim * DIM + jdim] +=
                                r2 + sqrtf(r2) + rsqrtf(r2 * 1000.0);
                        }
                    }
                }
            }
        }
        // set grad on target due to all neighbor source particles
        for (uint32_t idim = 0; idim < DIM; idim++)
        {

            for (uint32_t jdim = 0; jdim < DIM; jdim++)
            {
                grad[tid * DIM * DIM + idim * DIM + jdim] =
                    t_grad[idim * DIM + jdim];
            }
        }
        // printf("tid %u, total source particle %u\n",tid,s_cnt);
    }
}

// random number in [0, 1)
float get_random() { return ((float)rand() / (float)RAND_MAX); }

int main()
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

    // yin's example
    stride = (uint32_t*)malloc(sizeof(uint32_t) * DIM);
    stride[0] = cell_stride, stride[1] = cell_stride, stride[2] = cell_stride;
    n_cell = stride[0] * stride[1] * stride[2];
    n_particle = particle_stride * particle_stride * particle_stride;
    cell_size = 1.0f;
    box_size = cell_size * stride[0];

    /*
    // test example
    stride = (uint32_t*)malloc(sizeof(uint32_t) * DIM);
    stride[0] = 256, stride[1] = 256, stride[2] = 256;
    n_cell = stride[0]*stride[1]*stride[2];
    n_particle = np_per_cell*n_cell;
    cell_size = 1.0f;
    box_size = cell_size*stride[0];
    */

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

    /*
    neighcell_offset[0*DIM + 0] = 0;neighcell_offset[0*DIM + 1] =
    0;neighcell_offset[0*DIM + 2] = 0; neighcell_offset[1*DIM + 0] =
    -1;neighcell_offset[1*DIM + 1] = 0;neighcell_offset[1*DIM + 2] = 0;
    neighcell_offset[2*DIM + 0] = 1;neighcell_offset[2*DIM + 1] =
    0;neighcell_offset[2*DIM + 2] = 0; neighcell_offset[3*DIM + 0] =
    0;neighcell_offset[3*DIM + 1] = -1;neighcell_offset[3*DIM + 2] = 0;
    neighcell_offset[4*DIM + 0] = 0;neighcell_offset[4*DIM + 1] =
    1;neighcell_offset[4*DIM + 2] = 0; neighcell_offset[5*DIM + 0] =
    0;neighcell_offset[5*DIM + 1] = 0;neighcell_offset[5*DIM + 2] = -1;
    neighcell_offset[6*DIM + 0] = 0;neighcell_offset[6*DIM + 1] =
    0;neighcell_offset[6*DIM + 2] = 1;
    */
    /*
    // Initialize host arrays
    for(uint32_t ix=0; ix<stride[0]; ix++)
      for(uint32_t iy=0; iy<stride[1]; iy++)
        for(uint32_t iz=0; iz<stride[2]; iz++){
          uint32_t cell_id = ix + iy*stride[0] + iz*stride[0]*stride[1];
          cell_ids[cell_id*2+0] = cell_id*np_per_cell;
          cell_ids[cell_id*2+1] = (cell_id+1)*np_per_cell;
          for(uint32_t ip=0; ip<np_per_cell; ip++){
            uint32_t pid = cell_id*np_per_cell+ip;
            pos[DIM*pid+0] = ix;
            pos[DIM*pid+1] = iy;
            pos[DIM*pid+2] = iz;
            //printf("pos: %u, %u, %u\n", pos[DIM*pid+0], pos[DIM*pid+1],
    pos[DIM*pid+2]); particle_ids[pid] = cell_id; for(uint32_t idim=0; idim<DIM;
    idim++){ disp[DIM*pid+idim] = 0.2f*get_random();
              //printf("disp: %f\n", disp[DIM*pid+idim]);
              force[DIM*pid+idim] = 0.0f;
            }
          }
        }
    for(uint32_t ineigh=0; ineigh<n_neigh; ineigh++){
      for(uint32_t idim=0; idim<DIM; idim++){
        neighcell_offset[ineigh*DIM + idim] = 1;
      }
    }
    for(uint32_t i=0; i<n_neigh*DIM; i++){
      //printf("neigh %u\n", neighcell_offset[i]);
    }
    */

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

        /*
        if(i%2==0)
          thrust::copy(thrust::device, f1.begin(), f1.end(), f2.begin());
        else
          thrust::copy(thrust::device, f2.begin(), f2.end(), f1.begin());
        */

        //scatter_thrust(n_particle, d_pos, d_disp, cell_size, stride, d_value, d_grid_val);
        //gather_thrust(n_particle, d_pos, d_disp, cell_size, stride, d_value, d_grid_val);


        /*
        thrust::device_ptr<uint32_t> dev_ptr = thrust::device_pointer_cast((uint32_t*)d_particle_ids);
        thrust::stable_sort(thrust::device, dev_ptr, dev_ptr + n_particle, thrust::greater<uint32_t>());
        thrust::stable_sort_by_key(thrust::device, dev_ptr, dev_ptr + n_particle, f1.begin());
        thrust::sort_by_key(thrust::device, dev_ptr, dev_ptr + n_particle, f1.begin());
        */

        /*
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
        */


        /*
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

        /*
        // dense histogram
        uint32_t num_bins = n_cell + 1;
        histogram.resize(num_bins);
        thrust::counting_iterator<uint32_t> search_begin(0);
        thrust::upper_bound(dev_ptr,dev_ptr+n_particle,
                            search_begin, search_begin + num_bins,
                            histogram.begin());
        thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());
        */

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
        //scatter_kernel_gm<<<grid_size, block_size>>>(n_particle, d_pos, d_disp, cell_size, d_stride, d_value, d_grid_val);
        // gather using shared memory
        gather_kernel_sm<<<nbin*nbin*nbin, 512, (bin_size+1)*(bin_size+1)*(bin_size+1)*sizeof(float)>>>(d_pos, d_disp, cell_size, d_stride, d_value, d_grid_val, bin_size, bin_size, bin_size, d_bin_start, d_bin_count, d_index);

        //local_nbody_pair_sum<<<grid_size, block_size>>>(
        //    d_cell_ids, n_cell, d_pos, d_disp, d_particle_ids, n_particle,
        //    d_neighcell_offset, n_neigh, d_stride, cell_size, box_size,
        //    d_force);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaDeviceSynchronize();
        printf("cuda kernel takes: %f milliseconds\n", milliseconds);

        // thrust::for_each(thrust::device, d_particle_ids, d_particle_ids+100,
        // printf_functor());
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

    // Verification
    // for(int i = 0; i < N; i++){
    // assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    //}

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
