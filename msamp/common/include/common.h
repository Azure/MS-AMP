// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSAMP_COMMON_H_
#define MSAMP_COMMON_H_

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

template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
    return (((x) + ((y)-1)) / (y));
}

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

#define SELECT_FP8_TYPE(is_e4m3, type, ...)                                                                            \
    if (is_e4m3){                                                                                                      \
        using type = fp8e4m3;                                                                                          \
        { __VA_ARGS__ }                                                                                                \
    }                                                                                                                  \
    else {                                                                                                             \
        using type = fp8e5m2;                                                                                          \
        { __VA_ARGS__ }                                                                                                \
    }


#define TORCH_DTYPE_SWITCH_INPUT(dtype, type, ...)                                                                     \
    switch (dtype) {                                                                                                   \
        case torch::kFloat32: {                                                                                        \
            using type = float;                                                                                        \
            { __VA_ARGS__ }                                                                                            \
        } break;                                                                                                       \
        case torch::kBFloat16: {                                                                                       \
            using type = bf16;                                                                                         \
            { __VA_ARGS__ }                                                                                            \
        } break;                                                                                                       \
        case torch::kFloat16: {                                                                                        \
            using type = fp16;                                                                                         \
            { __VA_ARGS__ }                                                                                            \
        } break;                                                                                                       \
        default:                                                                                                       \
            throw "Unexcepted data type";                                                                              \
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

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

template <typename T>
struct is_half : std::false_type {};

template <>
struct is_half<fp16> : std::true_type {};

template <>
struct is_half<bf16> : std::true_type {};

#endif  // MSAMP_COMMON_H_