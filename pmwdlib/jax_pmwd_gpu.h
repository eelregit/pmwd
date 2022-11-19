#ifndef _JAX_PMWD_GPU_H_
#define _JAX_PMWD_GPU_H_

#include <complex>

namespace jax_pmwd {

template <typename T>
struct data_type;

template <>
struct data_type<double> {
  typedef double dtype;
};

template <>
struct data_type<float> {
  typedef float dtype;
};

template <typename T>
void scatter_cuda();

template <>
void scatter_cuda<float>() {
}

template <>
void scatter_cuda<double>() {
}

template <typename T>
void gather_cuda();

template <>
void gather_cuda<float>() {
}

template <>
void gather_cuda<double>() {
}

}  // namespace jax_pmwd

#endif
