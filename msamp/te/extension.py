# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP te.extension module."""

import torch
import transformer_engine.pytorch as te
import transformer_engine_extensions as tex

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor


class TeExtensionOverrider:
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
    original_fp8_cast_transpose_fused = te.cpp_extensions.fp8_cast_transpose_fused

    @staticmethod
    @torch.no_grad()
    def fused_cast_transpose(input, scale, amax, scale_inv, input_cast, input_transpose, otype):
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
            qtype = TeExtensionOverrider.dtype_map[otype]
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
            TeExtensionOverrider.original_fused_cast_transpose(
                input, scale, amax, scale_inv, input_cast, input_transpose, otype
            )

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
    def override():
        """Override transformer engine extension functions."""
        tex.fused_cast_transpose = TeExtensionOverrider.fused_cast_transpose
        te.cpp_extensions.cast_to_fp8 = TeExtensionOverrider.cast_to_fp8
        te.cpp_extensions.fp8_cast_transpose_fused = TeExtensionOverrider.fp8_cast_transpose_fused


TeExtensionOverrider.override()
