# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Transformer engine wrapper module."""

import transformer_engine as te  # noqa: F401 # pylint:disable=unused-import
import transformer_engine_extensions as tex

from msamp.common.dtype import QType


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
        """gemm operator by calling te_gemm.

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
