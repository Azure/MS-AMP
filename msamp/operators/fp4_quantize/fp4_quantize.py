# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""fp4_quantize module."""

import torch
from typing import Literal

from msamp.common.tensor import ScalingMeta

import msamp_quantize


class FP4_QUANTIZER:
    """FP4 Quantization operator."""
    @staticmethod
    def apply_DGE_item(
        input_tensor: torch.Tensor,
        k: float = 5.0,
        power_clamp_max: float = 3.0
    ) -> torch.Tensor:
        """
        Apply DGE item to input tensor. Note that this function is fixed to E2M1 format with no NaN.
        DGE: Abbreviation of the method 'Differentiable Gradient Estimator' for more accurate gradient update in FP4 training.

        Args:
            input (torch.Tensor): input tensor.
            k (float): parameter k to determine the sharpness of the differentiable quantization estimator.
            power_clamp_max (float): parameter power_clamp_max to restrict the amplitude of the estimated gradient.
        Returns:
            torch.Tensor: output tensor.
        """
        if not (input_tensor.is_cuda and input_tensor.is_contiguous):
            raise ValueError('The input tensor is not in cuda memory or contiguous.')
        if not (input_tensor.dtype == torch.bfloat16):
            raise ValueError('The input tensor is not in bfloat16.')

        output_tensor = torch.zeros_like(input_tensor)
        msamp_quantize.launch_differentiable_quantize_derivative(input_tensor, output_tensor, k, power_clamp_max, torch.numel(input_tensor))
        return output_tensor


    @staticmethod
    def _apply_quantile_clipping(
        input: torch.Tensor, 
        clip_threshold: float = 0.99,
        channel_wise: bool = False,
        token_wise: bool = False,
        return_residual: bool = False,
    ) -> tuple:
        '''
        Apply quantile clipping to the input tensor.
        
        Args:
            input (torch.Tensor): input tensor.
            clip_threshold (float): threshold for quantile clipping. Default is 0.99.
            channel_wise (bool): whether to apply clipping through channel dimension. Default is False.
            token_wise (bool): whether to apply clipping through token dimension. Default is False.
            return_residual (bool): whether to return the residual. Default is False.
        Returns:
            tuple: output tensor and residual tensor (if return_residual is True).
        '''
        float_input = input.float() if input.dtype != torch.float32 else input

        if channel_wise:
            sorted_tensor = torch.sort(input, dim=0).values
            lower_index = int((1 - clip_threshold) * sorted_tensor.size(0))
            upper_index = int(clip_threshold * sorted_tensor.size(0))

            lower_bound = sorted_tensor[lower_index:lower_index+1, :]
            upper_bound = sorted_tensor[upper_index:upper_index+1, :]

            output = torch.clamp(input, min=lower_bound, max=upper_bound)

        elif token_wise:
            sorted_tensor = torch.sort(input, dim=1).values
            lower_index = int((1 - clip_threshold) * sorted_tensor.size(1))
            upper_index = int(clip_threshold * sorted_tensor.size(1))

            lower_bound = sorted_tensor[:, lower_index:lower_index+1]
            upper_bound = sorted_tensor[:, upper_index:upper_index+1]

            output = torch.clamp(input, min=lower_bound, max=upper_bound)

        else:
            sorted_tensor = torch.sort(float_input.view(-1))[0]
            lower_index = int((1 - clip_threshold) * sorted_tensor.size(0))
            upper_index = int(clip_threshold * sorted_tensor.size(0))
            
            lower_bound = sorted_tensor[lower_index:lower_index+1]
            upper_bound = sorted_tensor[upper_index:upper_index+1]
            
            output = torch.clamp(input, min=lower_bound, max=upper_bound)

        output = output.to(input.dtype)
        if return_residual:
            return output, input - output
        else:
            return output, None


    @staticmethod
    def quantize_simulate_fp4_in_bf16(
        input_tensor: torch.Tensor,
        format: Literal['e2m1', 'e1m2'] = 'e1m2',
        nan_existed: bool = False,
        channel_wise: bool = False,
        token_wise: bool = False,
        outlier_clip: bool = False,
        clip_threshold: float = 0.99,
        residual_compensation: bool = False,
        return_scaled_input_for_bwd: bool = False,
    ) -> torch.Tensor:
        """
        Quantize high precision tensor to FP4 tensor.
        
        Args:
            input_tensor (torch.Tensor): high precision tensor to quantize. Note that the input tensor should be in cuda memory and bfloat16 dtype.
            format (Literal['e2m1', 'e1m2']): format of the quantized tensor. Default is 'e1m2'.
            nan_existed (bool): whether NaN value exists in the input tensor. Default is False.
            channel_wise (bool): whether to quantize the input tensor through channel dimension. Default is False.
            token_wise (bool): whether to quantize the input tensor through token dimension. Default is False.
            outlier_clip (bool): whether to apply outlier clipping to the input tensor. Default is False.
            clip_threshold (float): threshold for outlier clipping. Default is 0.99.
            residual_compensation (bool): whether to add residual back to the quantized tensor. Default is False.
            return_scaled_input_for_bwd (bool): whether to return scaled input tensor for backward computation. Default is False.
            
            Note: param 'nan_existed' claimed but needn't to be used (to keep API consistent with other functions).
        Returns:
            torch.Tensor: simulted FP4-quantied tensor, but still in bfloat16 dtype.
        """
        if not (input_tensor.is_cuda and input_tensor.is_contiguous):
            raise ValueError('The input tensor is not in cuda memory or contiguous.')
        if not (input_tensor.dtype == torch.bfloat16):
            raise ValueError('The input tensor is not in bfloat16.')

        # handle tensor shape for channel_wise or token_wise quantization
        shape = input_tensor.shape
        assert not (channel_wise and token_wise), f"channel_wise and token_wise cannot be True at the same time."
        if (channel_wise or token_wise) and len(shape) != 2:
            dim = shape[-1]
            input_tensor = input_tensor.reshape(-1, dim)

        # handle outlier clipping
        if outlier_clip:
            input_tensor, residual = FP4_QUANTIZER._apply_quantile_clipping(input_tensor, clip_threshold, channel_wise, token_wise, return_residual=residual_compensation)

        # get amax
        if channel_wise:
            amax = input_tensor.abs().max(dim=0, keepdim=True)[0]      # channel-wise max value
            scale = torch.ones((1, 1), dtype=input_tensor.dtype, device='cuda')     # 2-D tensor shape
        elif token_wise:
            amax = input_tensor.abs().max(dim=1, keepdim=True)[0]      # token-wise max value
            scale = torch.ones((1, 1), dtype=input_tensor.dtype, device='cuda')     # 2-D tensor shape
        else:
            amax = input_tensor.abs().max()
            scale = torch.ones((), dtype=input_tensor.dtype, device='cuda')
        # compute scaling factor
        fp_max = 6.0 if format == 'e2m1' else 7.0       # Fixed. For e1m2, actually it is 3.5, but we *2 for directly round()
        margin = 0
        sf = ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin)
        
        # quantize
        scaled_input = input_tensor * sf        # this * operation can handle matrix-tensor broadcasting. For example, (3, 4) * (4,) -> (3, 4)
        if format == 'e2m1':
            output_tensor = torch.zeros_like(scaled_input)
            msamp_quantize.quantize_bf16(scaled_input, output_tensor, torch.numel(scaled_input))
        else:
            output_tensor = torch.round(scaled_input)
        output_tensor.div_(sf)      # this .div_() method can also handle matrix-tensor broadcasting
        if residual_compensation:
            output_tensor = output_tensor + residual
        output_tensor.requires_grad = input_tensor.requires_grad

        # reshape output tensor to original shape
        if (channel_wise or token_wise) and len(shape) != 2:
            output_tensor = output_tensor.view(shape[:-1] + (-1, ))
            if return_scaled_input_for_bwd:
                scaled_input = scaled_input.view(shape[:-1] + (-1, ))

        if return_scaled_input_for_bwd:
            return output_tensor, scaled_input
        return output_tensor
