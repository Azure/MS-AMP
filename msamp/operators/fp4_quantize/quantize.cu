// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../common/include/common.h"


__global__ void quantize_bf16_kernel(const __nv_bfloat16* x, __nv_bfloat16* output, int x_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) {
        __nv_bfloat16 value = x[idx];
        __nv_bfloat16 closest;

        if (__hlt(value, __float2bfloat16(-5.0f))) {
            closest = __float2bfloat16(-6.0f);
        } else if (__hlt(value, __float2bfloat16(-3.5f))) {
            closest = __float2bfloat16(-4.0f);
        } else if (__hlt(value, __float2bfloat16(-2.5f))) {
            closest = __float2bfloat16(-3.0f);
        } else if (__hlt(value, __float2bfloat16(-1.75f))) {
            closest = __float2bfloat16(-2.0f);
        } else if (__hlt(value, __float2bfloat16(-1.25f))) {
            closest = __float2bfloat16(-1.5f);
        } else if (__hlt(value, __float2bfloat16(-0.75f))) {
            closest = __float2bfloat16(-1.0f);
        } else if (__hlt(value, __float2bfloat16(-0.25f))) {
            closest = __float2bfloat16(-0.5f);
        } else if (__hlt(value, __float2bfloat16(0.25f))) {
            closest = __float2bfloat16(0.0f);
        } else if (__hlt(value, __float2bfloat16(0.75f))) {
            closest = __float2bfloat16(0.5f);
        } else if (__hlt(value, __float2bfloat16(1.25f))) {
            closest = __float2bfloat16(1.0f);
        } else if (__hlt(value, __float2bfloat16(1.75f))) {
            closest = __float2bfloat16(1.5f);
        } else if (__hlt(value, __float2bfloat16(2.5f))) {
            closest = __float2bfloat16(2.0f);
        } else if (__hlt(value, __float2bfloat16(3.5f))) {
            closest = __float2bfloat16(3.0f);
        } else if (__hlt(value, __float2bfloat16(5.0f))) {
            closest = __float2bfloat16(4.0f);
        } else {
            closest = __float2bfloat16(6.0f);
        }

        output[idx] = closest;
    }
}

void quantize_bf16(at::Tensor input, at::Tensor output, int size) {

    const __nv_bfloat16* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());  
    __nv_bfloat16* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()); 

    const int threadsPerBlock = HIP_GET_NUM_THREADS(size);
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    quantize_bf16_kernel<<<blocks, threadsPerBlock, 0, stream>>>(input_data, output_data, size);
}


__device__ float power_derivative(float x, float delta, float k, float power_clamp_max) {
    float abs_term = fabsf(2.0f * x / delta - 1.0f);
    return fminf(powf(abs_term, 1.0f / k - 1.0f) / k, power_clamp_max);
}


// for fixed E2M1_no_NaN section: [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
__global__ void differentiable_quantize_derivative(
    const __nv_bfloat16* input, __nv_bfloat16* output, 
    float k, float power_clamp_max, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = __bfloat162float(input[idx]);
    float dy = 0.0f;

    if (x < -4.0f) {
        dy = power_derivative(x + 6.0f, 2.0f, k, power_clamp_max);
    } else if (x >= -4.0f && x < -3.0f) {
        dy = power_derivative(x + 4.0f, 1.0f, k, power_clamp_max);
    } else if (x >= -3.0f && x < -2.0f) {
        dy = power_derivative(x + 3.0f, 1.0f, k, power_clamp_max);
    } else if (x >= -2.0f && x < -1.5f) {
        dy = power_derivative(x + 2.0f, 0.5f, k, power_clamp_max);
    } else if (x >= -1.5f && x < -1.0f) {
        dy = power_derivative(x + 1.5f, 0.5f, k, power_clamp_max);
    } else if (x >= -1.0f && x < -0.5f) {
        dy = power_derivative(x + 1.0f, 0.5f, k, power_clamp_max);
    } else if (x >= -0.5f && x < 0.0f) {
        dy = power_derivative(x + 0.5f, 0.5f, k, power_clamp_max);
    } else if (x >= 0.0f && x < 0.5f) {
        dy = power_derivative(x, 0.5f, k, power_clamp_max);
    } else if (x >= 0.5f && x < 1.0f) {
        dy = power_derivative(x - 0.5f, 0.5f, k, power_clamp_max);
    } else if (x >= 1.0f && x < 1.5f) {
        dy = power_derivative(x - 1.0f, 0.5f, k, power_clamp_max);
    } else if (x >= 1.5f && x < 2.0f) {
        dy = power_derivative(x - 1.5f, 0.5f, k, power_clamp_max);
    } else if (x >= 2.0f && x < 3.0f) {
        dy = power_derivative(x - 2.0f, 1.0f, k, power_clamp_max);
    } else if (x >= 3.0f && x < 4.0f) {
        dy = power_derivative(x - 3.0f, 1.0f, k, power_clamp_max);
    } else if (x >= 4.0f && x <= 6.0f) {
        dy = power_derivative(x - 4.0f, 2.0f, k, power_clamp_max);
    }

    output[idx] = __float2bfloat16(dy);
}


void launch_differentiable_quantize_derivative(
    at::Tensor input, at::Tensor output,
    float k, float power_clamp_max, int size
) {
    const __nv_bfloat16* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());  
    __nv_bfloat16* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()); 

    const int threadsPerBlock = HIP_GET_NUM_THREADS(size);
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    differentiable_quantize_derivative<<<blocks, threadsPerBlock, 0, stream>>>(input_data, output_data, k, power_clamp_max, size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_bf16", &quantize_bf16, "Simulated Quantize FP4 Function in BF16 Format");
    m.def("launch_differentiable_quantize_derivative", &launch_differentiable_quantize_derivative, "Differentiable Quantize Derivative Function");
}
