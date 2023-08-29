# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP te.extension module."""

import torch

import transformer_engine.pytorch as te
import transformer_engine_extensions as tex

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor


class TeExtentionOverrider:
    """An Overrider to override some extension functions in transformer engine."""
    dtype_map = {
        tex.DType.kFloat8E4M3: Dtypes.kfloat8_e4m3,
        tex.DType.kFloat8E5M2: Dtypes.kfloat8_e5m2,
        tex.DType.kBFloat16: Dtypes.kbfloat16,
        tex.DType.kFloat16: Dtypes.kfloat16,
        tex.DType.kFloat32: Dtypes.kfloat32,
    }

    original_fused_cast_transpose = tex.fused_cast_transpose
    original_cast_to_fp8 = te.cpp_extensions.cast_to_fp8

    @classmethod
    @torch.no_grad()
    def fused_cast_transpose(cls, input, scale, amax, scale_inv, input_cast, input_transpose, otype):
        """Fused cast and transpose, support ScalingTensor.

        Args:
            input (torch.Tensor or ScalingTensor): Input tensor.
            scale (torch.Tensor): Scale tensor.
            amax (torch.Tensor): Amax tensor.
            scale_inv (torch.Tensor): Scale inverse tensor.
            input_cast (torch.Tensor): Casted input tensor.
            input_transpose (torch.Tensor): Transposed input tensor.
            otype (tex.DType): Output type.
        """
        if isinstance(input, ScalingTensor):
            qtype = TeExtentionOverrider.dtype_map[otype]
            if input_transpose is not None:
                sv = input.cast(qtype)
                # data should be contiguous, and TE does not check it.
                st = sv.t().contiguous()
                v, t = sv.value, st.value
                input_transpose.data = t
            else:
                sv = input.cast(qtype)
                v = sv.value

            if input_cast is not None:
                input_cast.data = v
            scale_inv.copy_(sv.meta.scale_inv)
        else:
            cls.original_fused_cast_transpose(input, scale, amax, scale_inv, input_cast, input_transpose, otype)

    @classmethod
    @torch.no_grad()
    def cast_to_fp8(cls, inp, fp8_meta_tensor, fp8_tensor, otype, out=None):
        """Cast to fp8, support ScalingTensor.

        Args:
            inp (torch.Tensor or ScalingTensor): Input tensor.
            fp8_meta_tensor (tex.FP8TensorMeta): Fp8 meta tensor.
            fp8_tensor (Union[tex.FP8FwdTensors, tex.FP8BwdTensors): Fp8 tensor.
            otype (tex.DType): Output type.
            out (torch.Tensor, optional): Output tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if isinstance(inp, ScalingTensor):
            qtype = TeExtentionOverrider.dtype_map[otype]
            sv = inp.cast(qtype)
            v = sv.value
            if out is not None:
                out.data = v
            fp8_meta_tensor.scale_inv[fp8_tensor].copy_(sv.meta.scale_inv)
            return v

        if out is None:
            return cls.original_cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype)
        return cls.original_cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype, out)

    @staticmethod
    def override():
        """Override transformer engine extension functions."""
        tex.fused_cast_transpose = TeExtentionOverrider.fused_cast_transpose
        te.cpp_extensions.cast_to_fp8 = TeExtentionOverrider.cast_to_fp8


TeExtentionOverrider.override()
