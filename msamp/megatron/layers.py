# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The layers module msamp.megatron."""

import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_global_memory_buffer
)
from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.operators.gemm import Gemm

import os

MSAMP_USE_WEIGHT_SIMULATE_FP4 = bool(int(os.getenv('MSAMP_USE_WEIGHT_SIMULATE_FP4', 0)))
MSAMP_USE_WEIGHT_DIFFERENTIABLE_GRADIENT_ESTIMATOR = bool(int(os.getenv('MSAMP_USE_WEIGHT_DIFFERENTIABLE_GRADIENT_ESTIMATOR', 0)))
MSAMP_USE_ACTIVATION_SIMULATE_FP4 = bool(int(os.getenv('MSAMP_USE_ACTIVATION_SIMULATE_FP4', 0)))

from msamp.operators.fp4_quantize import FP4_QUANTIZER


class FP8LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """A linear function with FP8 support, grad accumulation and async communication."""
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion, async_grad_allreduce, sequence_parallel):
        """Forward pass.

        Args:
            ctx: Context to store arbitrary data which can be retrieved during the backward pass.
            input (torch.Tensor): Input tensor.
            weight (ScalingTensor): Weight tensor.
            bias (torch.Tensor): Bias tensor.
            gradient_accumulation_fusion (bool): Whether to fuse gradient accumulation.
            async_grad_allreduce (bool): Whether to use asynchronous all-reduce.
            sequence_parallel (bool): Whether to use sequence parallel.

        Returns:
            torch.Tensor: Output tensor.
        """
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        input_shape = input.shape
        ctx.input_shape = input_shape
        metas = weight._scaling_metas
        ctx.metas = metas
        input_meta = metas['input']
        tp_group = get_tensor_model_parallel_group()

        output_dtype = input.dtype
        input = input.contiguous()

        old_meta_group = input_meta.group
        input_meta.group = tp_group
        if MSAMP_USE_ACTIVATION_SIMULATE_FP4:
            fp4_input_in_float = FP4_QUANTIZER.quantize_simu_fp4_in_bf16(input.bfloat16(), format='e2m1', nan_existed=False, token_wise=True, outlier_clip=True, clip_threshold=0.99)
            input_fp8 = fp4_input_in_float.cast(Dtypes.kfloat8_e4m3, meta=input_meta, sync=sequence_parallel)
        else:
            input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta, sync=sequence_parallel)
        input_meta.group = old_meta_group

        input_fp8.requires_grad = input.requires_grad
        input = input_fp8.value

        if MSAMP_USE_WEIGHT_SIMULATE_FP4:
            if MSAMP_USE_WEIGHT_DIFFERENTIABLE_GRADIENT_ESTIMATOR:
                fp4_weight_in_float, scaled_w = FP4_QUANTIZER.quantize_simu_fp4_in_bf16(weight.bfloat16(), format='e2m1', nan_existed=False, channel_wise=True, return_scaled_input_for_bwd=True)
            else:
                fp4_weight_in_float = FP4_QUANTIZER.quantize_simu_fp4_in_bf16(weight.bfloat16(), format='e2m1', nan_existed=False, channel_wise=True)
            weight_fp8 = fp4_weight_in_float.cast(Dtypes.kfloat8_e4m3)
        else:
            weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)
        weight_fp8.requires_grad = weight.requires_grad

        # save tensors
        ctx.input_fp8 = input_fp8
        ctx.weight_fp8 = weight_fp8
        ctx.weight = weight
        if MSAMP_USE_WEIGHT_DIFFERENTIABLE_GRADIENT_ESTIMATOR:
            ctx.save_for_backward(scaled_w)

        dim_size = list(input.size())
        if sequence_parallel:
            assert input.dtype == torch.uint8
            world_size = get_tensor_model_parallel_world_size()
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, 'mpu')
            torch.distributed._all_gather_base(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input
        ctx.dim_size = dim_size

        assert total_input.dtype == torch.uint8
        total_input_fp8 = ScalingTensor(total_input.view(-1, input_shape[-1]), input_fp8.meta)

        output_qtype = Dtypes.dtype_to_qtype[output_dtype]
        ctx.output_qtype = output_qtype
        output = Gemm.fp8_gemm(weight_fp8, total_input_fp8, output_qtype, use_split_accumulator=False)
        output = output.view(dim_size[:-1] + [-1])
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward pass.

        Args:
            grad_output (torch.Tensor): Output gradient tensor.

        Returns:
            A tuple of gradients of the arguments.
        """
        input_fp8 = ctx.input_fp8
        weight_fp8 = ctx.weight_fp8
        input = input_fp8.value
        output_qtype = ctx.output_qtype
        metas = ctx.metas
        ograd_meta = metas['ograd']

        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            assert input.dtype == torch.uint8
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, 'mpu')
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input

        grad_output = grad_output.contiguous()
        input_shape = ctx.input_shape
        output_shape = grad_output.shape
        if len(output_shape) != 2:
            grad_output = grad_output.view(-1, output_shape[-1])
        grad_output_fp8, grad_output_fp8_t = grad_output.fused_cast_transpose(Dtypes.kfloat8_e5m2, meta=ograd_meta)

        # grad_input
        weight_fp8_t = weight_fp8.fp8_transpose()
        grad_input = Gemm.fp8_gemm(weight_fp8_t, grad_output_fp8, output_qtype, use_split_accumulator=True)
        grad_input = grad_input.view(ctx.dim_size)

        if ctx.sequence_parallel:
            handle.wait()

        total_input_fp8 = ScalingTensor(total_input.view(-1, input_shape[-1]), input_fp8.meta)

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=grad_input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        assert not ctx.gradient_accumulation_fusion, \
            'gradient_accumulation_fusion not supported for FP8LinearWithGradAccumulationAndAsyncCommunication'
        # MS-AMP: compute wgrad
        total_input_fp8_t = total_input_fp8.fp8_transpose()
        wgrad_qtype = output_qtype

        grad_weight = Gemm.fp8_gemm(
            total_input_fp8_t,
            grad_output_fp8_t,
            wgrad_qtype,
            use_split_accumulator=True,
        )
        if MSAMP_USE_WEIGHT_DIFFERENTIABLE_GRADIENT_ESTIMATOR:
            scaled_w = ctx.saved_tensors[0]
            grad_weight.mul_(FP4_QUANTIZER.apply_DGE_item(scaled_w))

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # FP8 Weight Gradient
        ctx.weight.backward_grad_update(grad_weight)
        grad_weight = None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None
