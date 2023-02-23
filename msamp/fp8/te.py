# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP TeFp8."""

import transformer_engine
import transformer_engine_extensions as tex


class TeFp8:
    """Transformer engine fp8 class."""
    @staticmethod
    def _to_te_dtype(dtype):
        """Convert Dtypes.QType to transformer_engine.DType.

        Args:
            dtype (Dtypes.QType): Dtypes.QType to convert.

        Returns:
            transformer_engine.DType: Converted DType.
        """
        return getattr(tex.DType, dtype.name)

    @staticmethod
    def cast_to_fp8(input, scale, amax, scale_inv, otype):
        """Cast tensor to fp8 using transformer engine.

        Args:
            input (torch.Tensor): Tensor to cast.
            scale (torch.Tensor): Scaling factor.
            amax (torch.Tensor): Absolute maximum value in input.
            scale_inv (torch.Tensor): Inverse of scale.
            otype (Dtypes.QType): Type of output, could be Dtypes.kfloat8_e4m3 or Dtypes.kfloat8_e5m2.

        Return:
            torch.Tensor: Fp8 format tensor whose dtype is torch.uint8.
        """
        otype = TeFp8._to_te_dtype(otype)
        return tex.cast_to_fp8(input, scale, amax, scale_inv, otype)

    @staticmethod
    def cast_from_fp8(input, scale_inv, itype, otype):
        """Cast fp8 tensor to pytorch native tensor.

        Args:
            input (torch.Tensor): Fp8 tensor to cast, usually come from cast_to_fp8.
            scale_inv (torch.Tensor): Inverse of scale.
            itype (Dtypes.QType): Input tensor's type.
            otype (Dtypes.QType): Output tensor's type.

        Return:
            torch.Tensor: Pytorch native tensor.
        """
        itype = TeFp8._to_te_dtype(itype)
        otype = TeFp8._to_te_dtype(otype)
        return tex.cast_from_fp8(input, scale_inv, itype, otype)
