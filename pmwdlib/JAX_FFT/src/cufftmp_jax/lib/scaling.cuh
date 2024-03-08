#include "box_iterator.hpp"

template<typename T>
__global__
void scaling_kernel(BoxIterator<T> begin, BoxIterator<T> end, int rank, int size, size_t nx, size_t ny, size_t nz) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    begin += tid;
    if(begin < end) {
        // begin.x(), begin.y() and begin.z() are the global 3D coordinate of the data pointed by the iterator
        // begin->x and begin->y are the real and imaginary part of the corresponding cufftComplex element
        /*if(tid < 10) {
            printf("GPU data (after first transform): global 3D index [%d %d %d], local index %d, rank %d is (%f,%f)\n", 
                (int)begin.x(), (int)begin.y(), (int)begin.z(), (int)begin.i(), rank, begin->x, begin->y);
        }*/
        *begin = {begin->x / (float)(nx * ny * nz), begin->y / (float)(nx * ny * nz)};
    }
};

__global__
void scaling_kernel1(BoxIterator<cufftComplex> begin, BoxIterator<cufftComplex> end, int rank, int size, size_t nx, size_t ny, size_t nz) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    begin += tid;
    if(begin < end) {
        // begin.x(), begin.y() and begin.z() are the global 3D coordinate of the data pointed by the iterator
        // begin->x and begin->y are the real and imaginary part of the corresponding cufftComplex element
        if(tid < 10) {
            printf("GPU data (after first transform): global 3D index [%d %d %d], local index %d, rank %d is (%f,%f)\n", 
                (int)begin.x(), (int)begin.y(), (int)begin.z(), (int)begin.i(), rank, begin->x, begin->y);
        }
        //*begin = {begin->x / (float)(nx * ny * nz), begin->y / (float)(nx * ny * nz)};
    }
};

__global__
void print_kernel(float* data) {
    printf("data ====== %f  %f  %f  %f  %f  %f \n", data[0], data[1], data[2], data[3], data[4], data[5] );
};