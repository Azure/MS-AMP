# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FP8 Arithmetic module."""

import torch

import msamp_arithmetic
from msamp.common.dtype import Dtypes


class Arithmetic:
    """Arithmetic operator for FP8 tensor."""
    @staticmethod
    def add_to_fp8(fp8_tensor, meta, other):
        """Add high presicon tensor to fp8_tensor in-place.

        Args:
            fp8_tensor (torch.Tensor): fp8 tensor to add to.
            meta (ScalingTensorMeta): meta data of fp8_tensor.
            other (torch.Tensor): high precision tensor to add.
        """
        if not (fp8_tensor.is_cuda and fp8_tensor.is_contiguous):
            raise ValueError('The fp8 tensor is not in cuda memory or contiguous.')
        if not (other.is_cuda and other.is_contiguous):
            raise ValueError('The other tensor is not in cuda memory or contiguous.')
        if not (fp8_tensor.dtype == torch.uint8 or fp8_tensor.dtype == torch.int8):
            raise ValueError('The fp8 tensor is not in uint8 or int8.')

        if not (meta.qtype == Dtypes.kfloat8_e4m3 or meta.qtype == Dtypes.kfloat8_e5m2):
            raise ValueError('The fp8 tensor is not in e4m3 or e5m2 format.')

        is_e4m3 = meta.qtype == Dtypes.kfloat8_e4m3

        msamp_arithmetic.add_to_fp8(fp8_tensor, meta.scale, meta.scale_inv, meta.amax[0], other, is_e4m3)
