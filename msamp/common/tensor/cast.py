# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP TypeCast."""

import torch
import torch.distributed as dist

from msamp.common.dtype import Dtypes
from msamp.common.utils import DistUtil
from msamp.common.utils import TransformerEngineWrapper


class TypeCast:
    """Type cast helper class."""
    @staticmethod
    def cast_to_fp8(input, meta, sync=False, fuse_transpose=False):
        """Cast pytorch tensor to fp8.

        Args:
            input (torch.Tensor): Input tensor to cast whose dtype should not be torch.uint8/torch.int8.
            meta (ScalingMeta): Scaling meta data used for cast.
            sync (bool, optional): Sync or not. Defaults to False.
            fuse_transpose (bool, optional): Whether fused with transpose. Defaults to False.

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
        sync_amax = None
        if sync:
            # convert NAN to INF since NCCL-ReduceMax ignores NAN
            # notice: nan and posinf must be INF
            meta.amax[0].nan_to_num_(nan=torch.inf, posinf=torch.inf)
            world_size = DistUtil.get_world_size()
            if world_size > 1:
                dist.all_reduce(meta.amax[0], op=dist.ReduceOp.MAX, group=meta.group)
                sync_amax = meta.amax[0].clone()
        if in_time or sync:
            meta.reset_scaling_factor()
        if fuse_transpose:
            input_fp8, input_fp8_t = TransformerEngineWrapper.fp8_fused_cast_transpose(input, meta.qtype, meta)
            meta.scale_inv.data.copy_(torch.reciprocal(meta.scale))
            if sync_amax is not None:
                meta.amax[0].copy_(sync_amax)
            return input_fp8, input_fp8_t
        else:
            input_fp8 = TransformerEngineWrapper.cast_to_fp8(
                input.view(1, -1),
                meta.scale,
                meta.amax[0],
                meta.scale_inv,
                meta.qtype,
            )
            # scale_inv will not be set to inverse of scale in transformer-engine v0.7.
            meta.scale_inv.data.copy_(torch.reciprocal(meta.scale))    # scale_inv = 1 / scale
            if sync_amax is not None:
                meta.amax[0].copy_(sync_amax)
            return input_fp8.view_as(input)

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
        in_time = meta.is_in_time_scaling()
        if in_time or sync:
            meta.amax[0] = input.abs().max()
        if sync:
            # convert NAN to INF since NCCL-ReduceMax ignores NAN
            # notice: nan and posinf must be INF
            meta.amax[0].nan_to_num_(nan=torch.inf, posinf=torch.inf)
            world_size = DistUtil.get_world_size()
            if world_size > 1:
                dist.all_reduce(meta.amax[0], op=dist.ReduceOp.MAX, group=meta.group)
        if in_time or sync:
            meta.reset_scaling_factor()
            meta.scale_inv.data.copy_(torch.reciprocal(meta.scale))    # scale_inv = 1 / scale
        dtype = Dtypes.get_dtype_from_qtype(meta.qtype)
        # reshape scale to the tensor with the shape of (1,)
        # to avoid overflow when scale is larger than the maximum of qtype
        return (input * meta.scale.view((1, ))).to(dtype)

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

        return TransformerEngineWrapper.cast_from_fp8(
            input.view(1, -1),
            meta.scale_inv,
            meta.qtype,
            otype,
        ).view_as(input)

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
        dtype = Dtypes.get_dtype_from_qtype(otype)
        return (input * meta.scale_inv.view((1, ))).to(dtype)

    cast_from_fp32 = cast_from_fp16
