# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP tensor distribution module."""

import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from msamp.common.utils import DistUtil
from msamp.common.tensor import ScalingTensor


class TensorDist:
    """Distribute tensors(including ScalingTensor) across processes."""
    BROADCAST_BUCKET_SIZE = 512 * (1024**2)

    @staticmethod
    def _dist_tensors_after_flatten(tensors, dist_fn):
        """Distribute tensors after flatten.

        Args:
            tensors (list of torch.Tensor or ScalingTensor): Tensors to be distributed.
            dist_fn (function): Distribution function.
        """
        if len(tensors) == 0:
            return
        # TODO: Check contiguous buffer
        flat = _flatten_dense_tensors(tensors)
        dist_fn(flat)
        for p, q in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
            p.data = q.data

    @classmethod
    def _dist_tensors_by_bucket(cls, tensors, dist_fn, bucket_size):
        """Distribute tensors by bucket.

        Args:
            tensors (list of torch.Tensor or ScalingTensor): Tensors to be distributed.
            dist_fn (function): Distribution function.
            bucket_size (int): Bucket size in bytes.
        """
        world_size = DistUtil.get_world_size()
        if world_size == 1:
            return
        if not isinstance(tensors, list):
            tensors = [tensors]
        buffer = []
        n = 0
        for t in tensors:
            if t is None:
                continue
            if isinstance(t, ScalingTensor):
                t = t.value
            buffer.append(t)
            n += t.numel()
            if n >= bucket_size:
                cls._dist_tensors_after_flatten(buffer, dist_fn)
                buffer.clear()
                n = 0
        cls._dist_tensors_after_flatten(buffer, dist_fn)

    @classmethod
    def broadcast(cls, tensors, src, bucket_size=BROADCAST_BUCKET_SIZE):
        """Broadcast tensors across processes in one process group.

        Args:
            tensors (list of torch.Tensor or ScalingTensor): Tensors to be broadcasted.
            src (int): Source rank.
            bucket_size (int): Bucket size in bytes.
        """
        world_size = DistUtil.get_world_size()
        if world_size == 1:
            return

        def dist_fn(x):
            return dist.broadcast(x, src=src)

        if isinstance(tensors[0], ScalingTensor):
            values = [p.value for p in tensors]
            cls._dist_tensors_by_bucket(values, dist_fn, bucket_size)
            scales = [p.meta.scale for p in tensors]
            cls._dist_tensors_by_bucket(scales, dist_fn, bucket_size)
        else:
            cls._dist_tensors_by_bucket(tensors, dist_fn, bucket_size)
