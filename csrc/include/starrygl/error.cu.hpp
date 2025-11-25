#pragma once
#include <cuda_runtime.h>

struct KernelError {
    cudaError_t code;
    int line;
    const char* file;
    const char* msg;
    //__device__ __host__ KernelError() : code(cudaSuccess), line(0), file(nullptr), msg(nullptr) {}
};

extern __device__ KernelError g_kernel_error;
__device__ void set_kernel_error(cudaError_t code, int line, const char* file, const char* msg);
void initialize_kernel_error(cudaStream_t stream);
#define KERNEL_CHECK(cond, msg) \
    do { if (!(cond)) { set_kernel_error(cudaErrorIllegalAddress, __LINE__, __FILE__, msg); return; } } while(0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)