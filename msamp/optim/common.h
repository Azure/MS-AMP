#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <string>

using namespace std;

using byte = uint8_t;
using int32 = int32_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

#define TORCH_DTYPE_SWITCH(dtype, type, ...)                                                                           \
    switch (dtype) {                                                                                                   \
    case torch::kUInt8: {                                                                                              \
        using type = fp8e4m3;                                                                                          \
        { __VA_ARGS__ }                                                                                                \
    } break;                                                                                                           \
    case torch::kInt8: {                                                                                               \
        using type = fp8e5m2;                                                                                          \
        { __VA_ARGS__ }                                                                                                \
    } break;                                                                                                           \
    case torch::kFloat32: {                                                                                            \
        using type = float;                                                                                            \
        { __VA_ARGS__ }                                                                                                \
    } break;                                                                                                           \
    case torch::kFloat16: {                                                                                            \
        using type = fp16;                                                                                             \
        { __VA_ARGS__ }                                                                                                \
    } break;                                                                                                           \
    case torch::kBFloat16: {                                                                                           \
        using type = bf16;                                                                                             \
        { __VA_ARGS__ }                                                                                                \
    } break;                                                                                                           \
    default:                                                                                                           \
        throw "Unexcepted data type";                                                                                  \
    }

const int HIP_MAX_GRID_NUM = 65535;
const int HIP_MAX_NUM_THREADS = 512;

inline int HIP_GET_NUM_THREADS(const int n) {
    return HIP_MAX_NUM_THREADS;
    // return std::min(HIP_MAX_NUM_THREADS, ((n + 31) / 32) * 32);
}

inline int HIP_GET_BLOCKS(const int n, const int num_threads) {
    return std::min(HIP_MAX_GRID_NUM, n + num_threads - 1) / num_threads;
}

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T, typename S> __host__ __device__ T cast_dtype(const S value) { return T(value); }

template <> __host__ __device__ fp16 cast_dtype(const float value) { return __float2half(value); }

template <> __host__ __device__ bf16 cast_dtype(const float value) { return __float2bfloat16(value); }

template <> __host__ __device__ float cast_dtype(const fp16 value) { return __half2float(value); }

template <> __host__ __device__ float cast_dtype(const bf16 value) { return __bfloat162float(value); }