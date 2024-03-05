# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP te.extension module."""

import torch
import transformer_engine.pytorch as te
import transformer_engine_extensions as tex

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.nn import ScalingParameter


class TeExtensionOverrider:
    """An Overrider to override some extension functions in transformer engine."""
    dtype_map = {
        tex.DType.kFloat8E4M3: Dtypes.kfloat8_e4m3,
        tex.DType.kFloat8E5M2: Dtypes.kfloat8_e5m2,
        tex.DType.kBFloat16: Dtypes.kbfloat16,
        tex.DType.kFloat16: Dtypes.kfloat16,
        tex.DType.kFloat32: Dtypes.kfloat32,
    }

    original_cast_to_fp8 = te.cpp_extensions.cast_to_fp8
    original_fp8_cast_transpose_fused = te.cpp_extensions.fp8_cast_transpose_fused
    original_cast_if_needed = te.utils.cast_if_needed

    @staticmethod
    @torch.no_grad()
    def fp8_cast_transpose_fused(inp, fp8_meta_tensor, fp8_tensor, dtype, cast_out=None, transpose_out=None):
        """Cast + Transpose with FP8 output, support ScalingTensor.

        Args:
            inp (torch.Tensor or ScalingTensor): Input tensor.
            fp8_meta_tensor: tex.FP8TensorMeta
            fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors]
            dtype: tex.DType
            cast_out (torch.Tensor, optional): Output tensor.
            transpose_out (torch.Tensor, optional): Output tensor.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], None]: Output tensor.
        """
        if isinstance(inp, ScalingTensor):
            qtype = TeExtensionOverrider.dtype_map[dtype]
            sv = inp.cast(qtype)
            v = sv.value
            t = sv.t().contiguous().value
            if transpose_out is not None:
                transpose_out.data = t
            if cast_out is not None:
                cast_out.data = v
            fp8_meta_tensor.scale_inv[fp8_tensor].copy_(sv.meta.scale_inv)
            return v, t

        return TeExtensionOverrider.original_fp8_cast_transpose_fused(
            inp, fp8_meta_tensor, fp8_tensor, dtype, cast_out, transpose_out
        )

    @staticmethod
    @torch.no_grad()
    def cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype, out=None):
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
            qtype = TeExtensionOverrider.dtype_map[otype]
            sv = inp.cast(qtype)
            v = sv.value
            if out is not None:
                out.data = v
            fp8_meta_tensor.scale_inv[fp8_tensor].copy_(sv.meta.scale_inv)
            return v

        if out is None:
            return TeExtensionOverrider.original_cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype)
        return TeExtensionOverrider.original_cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype, out)

    @staticmethod
    def cast_if_needed(tensor, dtype):
        """Cast tensor to dtype.

        Args:
            tensor (torch.Tensor or ScalingParameter): Input tensor.
            dtype (torch.dtype): Output dtype.

        Returns:
            torch.Tensor: Output tensor.
        """
        with torch.enable_grad():
            if isinstance(tensor, ScalingParameter):
                new_tensor = tensor.to(dtype)
                new_tensor.requires_grad = tensor.requires_grad
                return new_tensor
        return TeExtensionOverrider.original_cast_if_needed(tensor, dtype)

    @staticmethod
    def override():
        """Override transformer engine extension functions."""
        te.cpp_extensions.cast_to_fp8 = TeExtensionOverrider.cast_to_fp8
        te.module.linear.cast_to_fp8 = TeExtensionOverrider.cast_to_fp8
        te.cpp_extensions.fp8_cast_transpose_fused = TeExtensionOverrider.fp8_cast_transpose_fused
        te.module.linear.fp8_cast_transpose_fused = TeExtensionOverrider.fp8_cast_transpose_fused

        te.module.layernorm_linear.cast_if_needed = TeExtensionOverrider.cast_if_needed
        te.module.linear.cast_if_needed = TeExtensionOverrider.cast_if_needed
        te.module.layernorm_mlp.cast_if_needed = TeExtensionOverrider.cast_if_needed


TeExtensionOverrider.override()
