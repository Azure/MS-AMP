// Reference: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/utils.cuh

/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef MSAMP_UTILS_CUH_
#define MSAMP_UTILS_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>


////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t THREADS_PER_WARP = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace msamp {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint16 {
    uint4 u;
    uint4 v;
    uint4 s;
    uint4 t;
};

struct uint8 {
    uint4 u;
    uint4 v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES>
struct BytesToType {};

template<>
struct BytesToType<64> {
    using Type = uint16;
    static_assert(sizeof(Type) == 64);
};

template<>
struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
};

template<>
struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<>
struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<>
struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<>
struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<>
struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int num_elems>
__device__ __forceinline__ float warp_reduce_max(const float m) {
    float tmp = m;
#pragma unroll
    for (int delta = num_elems/2; delta > 0; delta /= 2) {
        const float other_m = __shfl_down_sync(0xFFFFFFFF, tmp, delta);
        __builtin_assume(tmp >= 0);
        __builtin_assume(other_m >= 0);
        tmp = fmaxf(tmp, other_m);
    }
    return tmp;
}

template <int num_warps, typename compute_t>
__device__ __forceinline__ compute_t reduce_max(const compute_t m, const int warpid) {
    __shared__ float staging[num_warps];
    constexpr int warp_size = 32;
    const float my_max = m;
    const float my_warp_max = warp_reduce_max<warp_size>(my_max);
    if (threadIdx.x % 32 == 0) {
        staging[warpid] = my_warp_max;
    }
    __syncthreads();
    compute_t result = 0;
    if (warpid == 0) {
        const float my_max = threadIdx.x < num_warps ? staging[threadIdx.x] : 0;
        result = warp_reduce_max<num_warps>(my_max);
    }
    return result;
}

// Works only on positive values
__device__ __forceinline__ void atomicMaxFloat(float * addr, const float value) {
    atomicMax(reinterpret_cast<int *>(addr), __float_as_int(value));
}

}  // namespace msamp

#endif  // MSAMP_UTILS_CUH_
