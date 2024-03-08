#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace nvtransfer_jax;

/**
 * Boilerplate used to
 * (1) Expose the gpu_nvtransfer function to Python (to launch our custom op)
 * (2) Expose the nvtransferDescriptor (to pass parameters from Python to C++)
 */

namespace {

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["gpu_nvtransfer_i16"] = EncapsulateFunction(gpu_nvtransfer_i16);
    dict["gpu_nvtransfer_i32"] = EncapsulateFunction(gpu_nvtransfer_i32);
    dict["gpu_nvtransfer_f32"] = EncapsulateFunction(gpu_nvtransfer_f32);
    dict["gpu_nvtransfer_f64"] = EncapsulateFunction(gpu_nvtransfer_f64);
    return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
    m.def("registrations", &Registrations);
    m.def("build_nvtransfer_descriptor",
        [](int flag, int send_buf_size, int recv_buf_size, int src_rank, int dst_rank) {
            return PackDescriptor(nvtransferDescriptor{flag, send_buf_size, recv_buf_size, src_rank, dst_rank});
        }
    );
}

}  // namespace
