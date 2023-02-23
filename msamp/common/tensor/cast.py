# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP TypeCast."""

import torch
import torch.distributed as dist

from msamp.common.dtype import Dtypes
from msamp.common.utils import DistUtil
from msamp.fp8.te import TeFp8


class TypeCast:
    """Type cast helper class."""
    @staticmethod
    def cast_to_fp8(input, meta, sync=False):
        """Cast pytorch tensor to fp8.

        Args:
            input (torch.Tensor): Input tensor to cast whose dtype should not be torch.uint8/torch.int8.
            meta (ScalingMeta): Scaling meta data used for cast.
            sync (bool, optional): Sync or not. Defaults to False.

        Return:
            torch.Tensor: tensor whose dtype is torch.uint8.
        """
        if not (input.is_cuda and input.is_contiguous):
            raise ValueError('The input tensor is not in cuda memory or contiguous.')
        if not meta.scale.is_cuda:
            raise ValueError('The meta.scale is not in cuda memory.')
        in_time = meta.is_in_time_scaling()
        if in_time:
            meta.amax[0] = input.abs().max()
            meta.reset_scaling_factor()
        if sync:
            world_size = DistUtil.get_world_size()
            if world_size > 1:
                dist.all_reduce(meta.scale, op=dist.ReduceOp.MIN)
        input_fp8 = TeFp8.cast_to_fp8(
            input.view(1, -1),
            meta.scale,
            meta.amax[0],
            1.0 / meta.scale,
            meta.qtype,
        )

        shape = input.shape
        return input_fp8.view(shape)

    @staticmethod
    def cast_to_fp16(input, meta, sync=False):
        """Cast pytorch tensor to to fp16.

        Args:
            input (torch.Tensor): Input tensor to cast.
            meta (ScalingMeta): Scaling meta data used for cast.
            sync (bool): Sync or not. Defaults to False.

        Return:
            torch.Tensor: tensor whose dtype is torch.float16.
        """
        meta.amax[0] = input.abs().max()
        in_time = meta.is_in_time_scaling()
        if in_time:
            # notice: we scale the tensor with qtype FP8-E4M3.
            meta.reset_scaling_factor(qtype=Dtypes.kfloat8_e4m3)
        if sync:
            world_size = DistUtil.get_world_size()
            if world_size > 1:
                dist.all_reduce(meta.scale, op=dist.ReduceOp.MIN)

        input_fp16 = (input * meta.scale).to(torch.float16)
        return input_fp16

    @staticmethod
    def cast_from_fp8(input, meta, otype):
        """Cast from fp8 tensor to pytorch tensor.

        Args:
            input (torch.Tensor): Input fp8 tensor to cast from.
            meta (ScalingMeta): Scaling meta data used for cast.
            otype (Dtypes.QType): The output type.

        Return:
            torch.Tensor: tensor whose dtype is otype.
        """
        if not (input.is_cuda and input.is_contiguous):
            raise ValueError('The input tensor is not in cuda memory or contiguous.')
        if input.dtype != torch.uint8:
            raise ValueError('The dtype of input tensor is not torch.uint8.')

        shape = input.shape
        return TeFp8.cast_from_fp8(
            input.view(1, -1),
            1.0 / meta.scale,
            meta.qtype,
            otype,
        ).view(shape)

    @staticmethod
    def cast_from_fp16(input, meta, otype):
        """Cast from fp16 tensor to pytorch tensor.

        Args:
            input (torch.Tensor): Input fp16/fp32 tensor to cast from.
            meta (ScalingMeta): Scaling meta data used for cast.
            otype (Dtypes.Qtype): The output type.

        Return:
            torch.Tensor: tensor whose type is otype.
        """
        dtype = Dtypes.qtype_to_dtype[otype]
        if input.dtype == dtype:
            # return a copy
            input = input.clone()
        else:
            input = input.to(dtype)
        if meta.scale != 1:
            input.mul_(1.0 / meta.scale)
        return input

    cast_from_fp32 = cast_from_fp16