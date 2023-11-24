// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <torch/extension.h>

#include <cmath>
#include <type_traits>

#include "../common/include/common.h"
#include "../common/include/utils.cuh"

using namespace std;
using namespace torch;
using namespace transformer_engine;
using index_t = int;

__global__ void adamw_kernel(index_t n, float *param, float *grad, float *exp_avg_value, float beta1,
                             float *exp_avg_sq_value, float beta2, float eps, float step_size, float sqrt_bias_corr2) {
    CUDA_KERNEL_LOOP(i, n) {
        exp_avg_value[i] = beta1 * exp_avg_value[i] + (1.f - beta1) * grad[i];
        exp_avg_sq_value[i] = beta2 * exp_avg_sq_value[i] + (1.f - beta2) * (grad[i] * grad[i]);
        float denom = sqrt(exp_avg_sq_value[i]) / sqrt_bias_corr2 + eps;
        param[i] -= step_size * (exp_avg_value[i] / denom);
    }
}

template <typename T, typename S>
__global__ void adamw_fp8_stage1_kernel(index_t n, float *param, float *grad, T *exp_avg_value, float exp_avg_factor,
                                        float *exp_avg_amax, float beta1, S *exp_avg_sq_value, float exp_avg_sq_factor,
                                        float *exp_avg_sq_amax, float beta2, float eps, float step_size,
                                        float sqrt_bias_corr2) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // record the maximum value of exp_avg_value_i and exp_avg_sq_value_i
    float max_exp_avg_value = 0, max_exp_avg_sq_value = 0;
    // pre-compute for maximum state
    CUDA_KERNEL_LOOP(i, n) {
        float exp_avg_value_i = beta1 * cast_dtype<float>(exp_avg_value[i]) * exp_avg_factor + (1.f - beta1) * grad[i];
        float exp_avg_sq_value_i =
            beta2 * cast_dtype<float>(exp_avg_sq_value[i]) * exp_avg_sq_factor + (1.f - beta2) * (grad[i] * grad[i]);

        max_exp_avg_value = fmaxf(fabsf(exp_avg_value_i), max_exp_avg_value);
        max_exp_avg_sq_value = fmaxf(fabsf(exp_avg_sq_value_i), max_exp_avg_sq_value);

        float denom = sqrt(exp_avg_sq_value_i) / sqrt_bias_corr2 + eps;
        // update param
        param[i] -= step_size * (exp_avg_value_i / denom);
    }

    max_exp_avg_value = reduce_max<HIP_MAX_NUM_THREADS / THREADS_PER_WARP>(max_exp_avg_value, warp_id);
    max_exp_avg_sq_value = reduce_max<HIP_MAX_NUM_THREADS / THREADS_PER_WARP>(max_exp_avg_sq_value, warp_id);

    if (threadIdx.x == 0) {
        atomicMaxFloat(exp_avg_amax, max_exp_avg_value);
        atomicMaxFloat(exp_avg_sq_amax, max_exp_avg_sq_value);
    }
}

template <typename T, typename S>
__global__ void adamw_fp8_stage2_kernel(index_t n, float *grad, T *exp_avg_value, float exp_avg_factor,
                                        float new_exp_avg_factor, float beta1, S *exp_avg_sq_value,
                                        float exp_avg_sq_factor, float new_exp_avg_sq_factor, float beta2) {
    // write to state
    CUDA_KERNEL_LOOP(i, n) {
        float exp_avg_value_i = beta1 * cast_dtype<float>(exp_avg_value[i]) * exp_avg_factor + (1.f - beta1) * grad[i];
        float exp_avg_sq_value_i =
            beta2 * cast_dtype<float>(exp_avg_sq_value[i]) * exp_avg_sq_factor + (1.f - beta2) * (grad[i] * grad[i]);
        // quantize states to T
        exp_avg_value[i] = cast_dtype<T>(exp_avg_value_i * new_exp_avg_factor);
        exp_avg_sq_value[i] = cast_dtype<S>(exp_avg_sq_value_i * new_exp_avg_sq_factor);
    }
}

// for FP32
void adamw_compute(Tensor param, Tensor grad, Tensor exp_avg_value, Tensor exp_avg_factor, Tensor exp_avg_amax,
                   float beta1, Tensor exp_avg_sq_value, Tensor exp_avg_sq_factor, Tensor exp_avg_sq_amax, float beta2,
                   float eps, int step, float lr) {
    at::cuda::CUDAGuard device_guard(param.device());
    float bias_correction1 = 1 - pow(beta1, step);
    float bias_correction2 = 1 - pow(beta2, step);
    float step_size = lr / bias_correction1;
    float sqrt_bias_corr2 = sqrt(bias_correction2);
    float *p_param = param.data_ptr<float>();
    float *p_grad = grad.data_ptr<float>();
    index_t numel = param.numel();

    const int threadsPerBlock = HIP_GET_NUM_THREADS(numel);
    const int blocks = HIP_GET_BLOCKS(numel, threadsPerBlock);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float *p_exp_avg_value = exp_avg_value.data_ptr<float>();
    float *p_exp_avg_sq_value = exp_avg_sq_value.data_ptr<float>();
    adamw_kernel<<<blocks, threadsPerBlock, 0, stream>>>(numel, p_param, p_grad, p_exp_avg_value, beta1,
                                                         p_exp_avg_sq_value, beta2, eps, step_size, sqrt_bias_corr2);
}

void adamw_fp8_stage1_compute(Tensor param, Tensor grad, Tensor exp_avg_value, float exp_avg_factor,
                              Tensor exp_avg_amax, float beta1, Tensor exp_avg_sq_value, float exp_avg_sq_factor,
                              Tensor exp_avg_sq_amax, float beta2, float eps, int step, float lr,
                              bool bias_correction) {
    at::cuda::CUDAGuard device_guard(param.device());
    float bias_correction1 = bias_correction ? 1 - pow(beta1, step) : 1.0;
    float bias_correction2 = bias_correction ? 1 - pow(beta2, step) : 1.0;
    float step_size = lr / bias_correction1;
    float sqrt_bias_corr2 = sqrt(bias_correction2);
    float *p_param = param.data_ptr<float>();
    float *p_grad = grad.data_ptr<float>();
    index_t numel = param.numel();

    const int threadsPerBlock = HIP_GET_NUM_THREADS(numel);
    const int blocks = HIP_GET_BLOCKS(numel, threadsPerBlock);
    auto state_dtype = exp_avg_value.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float *p_exp_avg_amax = exp_avg_amax.data_ptr<float>();
    float *p_exp_avg_sq_amax = exp_avg_sq_amax.data_ptr<float>();
    // pre-compute and update params
    TORCH_DTYPE_SWITCH(exp_avg_value.scalar_type(), T, {
        TORCH_DTYPE_SWITCH(exp_avg_sq_value.scalar_type(), S, {
            T *p_exp_avg_value = (T *)exp_avg_value.data_ptr();
            S *p_exp_avg_sq_value = (S *)exp_avg_sq_value.data_ptr();
            adamw_fp8_stage1_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
                numel, p_param, p_grad, p_exp_avg_value, exp_avg_factor, p_exp_avg_amax, beta1, p_exp_avg_sq_value,
                exp_avg_sq_factor, p_exp_avg_sq_amax, beta2, eps, step_size, sqrt_bias_corr2);
        });
    });
}

void adamw_fp8_stage2_compute(Tensor grad, Tensor exp_avg_value, float exp_avg_factor, float new_exp_avg_factor,
                              float beta1, Tensor exp_avg_sq_value, float exp_avg_sq_factor,
                              float new_exp_avg_sq_factor, float beta2, int step, bool bias_correction) {
    at::cuda::CUDAGuard device_guard(grad.device());
    float bias_correction1 = bias_correction ? 1 - pow(beta1, step) : 1.0;
    float bias_correction2 = bias_correction ? 1 - pow(beta2, step) : 1.0;
    float *p_grad = grad.data_ptr<float>();
    index_t numel = grad.numel();

    const int threadsPerBlock = HIP_GET_NUM_THREADS(numel);
    const int blocks = HIP_GET_BLOCKS(numel, threadsPerBlock);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TORCH_DTYPE_SWITCH(exp_avg_value.scalar_type(), T, {
        TORCH_DTYPE_SWITCH(exp_avg_sq_value.scalar_type(), S, {
            T *p_exp_avg_value = (T *)exp_avg_value.data_ptr();
            S *p_exp_avg_sq_value = (S *)exp_avg_sq_value.data_ptr();
            adamw_fp8_stage2_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
                numel, p_grad, p_exp_avg_value, exp_avg_factor, new_exp_avg_factor, beta1, p_exp_avg_sq_value,
                exp_avg_sq_factor, new_exp_avg_sq_factor, beta2);
        });
    });
}

string version() { return "1.0.0"; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamw_compute", &adamw_compute, "Adamw Compute");
    m.def("adamw_fp8_stage1_compute", &adamw_fp8_stage1_compute, "Adamw FP8 Stage1 Compute");
    m.def("adamw_fp8_stage2_compute", &adamw_fp8_stage2_compute, "Adamw FP8 Stage2 Compute");
    m.def("version", &version);
}
