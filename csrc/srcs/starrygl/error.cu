#include "error.cu.hpp"


// 定义符号
__device__ KernelError g_kernel_error;

__device__ void set_kernel_error(cudaError_t code, int line, const char* file, const char* msg) {
    g_kernel_error.code = code;
    g_kernel_error.line = line;
    g_kernel_error.file = file;
    g_kernel_error.msg = msg;
}

// 初始化 kernel
__global__ void init_kernel_error_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_kernel_error.code = cudaSuccess;
        g_kernel_error.line = 0;
        g_kernel_error.file = nullptr;
        g_kernel_error.msg = nullptr;
    }
}

// Host 函数（供 C++ 调用）
void initialize_kernel_error(cudaStream_t stream ) {
    init_kernel_error_kernel<<<1, 1, 0, stream>>>();
    // 注意：这里不调用 cudaStreamSynchronize，由调用方决定
}