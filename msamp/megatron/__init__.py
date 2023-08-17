# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP megatron package."""

from msamp.megatron.optimizer.clip_grads import clip_grad_norm_fp8
from msamp.megatron.distributed import FP8DistributedDataParallel
from msamp.common.utils.lazy_import import LazyImport

FP8LinearWithGradAccumulationAndAsyncCommunication = LazyImport(
    'msamp.megatron.layers', 'FP8LinearWithGradAccumulationAndAsyncCommunication'
)
FP8DistributedOptimizer = LazyImport('msamp.megatron.optimizer.distrib_optimizer', 'FP8DistributedOptimizer')

__all__ = [
    'clip_grad_norm_fp8', 'FP8DistributedDataParallel', 'FP8LinearWithGradAccumulationAndAsyncCommunication',
    'FP8DistributedOptimizer'
]
