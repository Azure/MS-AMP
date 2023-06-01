# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Transformer engine wrapper module."""

from typing import Tuple

import torch
import torch.nn.functional as F
import transformer_engine as te    # noqa: F401 # pylint:disable=unused-import
import transformer_engine_extensions as tex

from msamp.common.dtype import QType
from msamp.common.tensor import ScalingTensor, ScalingMeta


class PaddingTensor:
    """Padding Tensor to 16-byte aligned for FP8 operators."""
    def __init__(self, val: torch.Tensor, transpose: bool = False, pad_base: int = 16):
        """Constructor, init tensor shape and padding shape.

        Args:
            val (torch.Tensor): input tensor.
            transpose (bool, optional): Whether output transpose or not. Defaults to False.
            pad_base (int, optional): Padding base in number of elements. Defaults to 16.
        """
        self.val = val
        self.dim = val.shape
        self.transpose = transpose
        assert len(self.dim) == 2, 'Input must have 2 dimensions.'
        self.pad = [-d % pad_base for d in self.dim]
        self.require_pad = any(p > 0 for p in self.pad)

    def __enter__(self):
        """Pad input tensor if needed.

        Returns:
            PaddingTensor: return self.
        """
        if self.require_pad:
            self.val = F.pad(self.val, (0, self.pad[1], 0, self.pad[0]))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unpad input tensor when exit the context manager.

        Args:
            exc_type (Type): exception's class.
            exc_val (BaseException): exception instance.
            exc_tb (TracebackType): exception traceback.

        Returns:
            bool: True to suppress an exception raised in the context.
        """
        if self.require_pad:
            if self.transpose:
                self.val = self.val[:self.dim[1], :self.dim[0]]
            else:
                self.val = self.val[:self.dim[0], :self.dim[1]]
        return None


class TransformerEngineWrapper:
    """Wrapper class for transformer engine."""
    @staticmethod
    def _to_te_dtype(qtype):
        """Convert qtype to te dtype.

        Args:
            qtype (msamp.QType): qtype defined in msamp package.

        Return:
            tex.DType: return the type defined in transformer_engine_extensions package.
        """
        return getattr(tex.DType, qtype.name)

    @staticmethod
    def _to_compatible_args(args):
        """Convert all qtype to te dtype in the args list.

        Args:
            args (List): args of the gemm operator.

        Return:
            List: return the te-compatible arguments.
        """
        def fn(a):
            if isinstance(a, QType):
                return TransformerEngineWrapper._to_te_dtype(a)
            return a

        new_args = [fn(a) for a in args]
        return new_args

    @staticmethod
    def te_gemm(*args):
        """GEMM operator by calling te_gemm.

        Args:
            args (List): args of the te_gemm operator.
        """
        new_args = TransformerEngineWrapper._to_compatible_args(args)
        tex.te_gemm(*new_args)

    @staticmethod
    def cast_to_fp8(input, scale, amax, scale_inv, otype):
        """Cast input to FP8 format.

        Args:
            input (torch.Tensor): the tensor to cast.
            scale (torch.Tensor): the tensor scale.
            amax (torch.Tensor): the tensor amax.
            scale_inv (torch.Tensor): the inverses of tensor scale.
            otype (tex.DType): the output tensor type.

        Return:
            torch.Tensor: the output tensor in FP8 format.
        """
        otype = TransformerEngineWrapper._to_te_dtype(otype)
        return tex.cast_to_fp8(input, scale, amax, scale_inv, otype)

    @staticmethod
    def cast_from_fp8(input, scale_inv, itype, otype):
        """Cast input from fp8 format.

        Args:
            input (torch.Tensor): the tensor to cast.
            scale_inv (torch.Tensor): the inverses of tensor scale.
            itype (tex.DType): the input tensor type.
            otype (tex.DType): the output tensor type.

        Return:
            torch.Tensor: the output tensor in otype format.
        """
        itype = TransformerEngineWrapper._to_te_dtype(itype)
        otype = TransformerEngineWrapper._to_te_dtype(otype)
        return tex.cast_from_fp8(input, scale_inv, itype, otype)

    @staticmethod
    def fp8_fused_cast_transpose(input: torch.Tensor, qtype: QType,
                                 meta: ScalingMeta) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused cast and transpose for input tensor.

        Args:
            input (torch.Tensor): input tensor.
            qtype (QType): qtype to cast.
            meta (ScalingMeta): scaling meta.

        Returns:
            torch.Tensor, torch.Tensor: casted and transposed tensors.
        """
        with PaddingTensor(input) as pad_input:
            out_cast = torch.empty_like(pad_input.val, dtype=torch.uint8)
            out_t = torch.empty(out_cast.shape[1], out_cast.shape[0], device='cuda', dtype=torch.uint8)
            tex.fused_cast_transpose(
                pad_input.val, meta.scale, meta.amax, 1.0 / meta.scale, out_cast, out_t,
                TransformerEngineWrapper._to_te_dtype(qtype)
            )
            if pad_input.require_pad:
                out_cast = out_cast[:input.shape[0], :input.shape[1]]
                out_t = out_t[:input.shape[1], :input.shape[0]]
        return out_cast, out_t

    @staticmethod
    def fp8_transpose(input: ScalingTensor) -> ScalingTensor:
        """Transpose the input tensor.

        Args:
            input (ScalingTensor): input scaling tensor.

        Returns:
            ScalingTensor: transposed scaling tensor.
        """
        with PaddingTensor(input.value, transpose=True) as pad_input:
            pad_input.val = tex.fp8_transpose(pad_input.val, TransformerEngineWrapper._to_te_dtype(input.meta.qtype))
        return ScalingTensor(pad_input.val.contiguous(), input.meta)
