# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP megatron package."""

from msamp.megatron.optimizer.clip_grads import clip_grad_norm_fp8
from msamp.megatron.layers import FP8LinearWithGradAccumulationAndAsyncCommunication
from msamp.megatron.distributed import FP8DistributedDataParallel
from msamp.common.utils.lazy_import import LazyImport

FP8DistributedOptimizer =  LazyImport('msamp.megatron.optimizer.distrib_optimizer', 'FP8DistributedOptimizer')

__all__ = ['clip_grad_norm_fp8', 'FP8LinearWithGradAccumulationAndAsyncCommunication', 'FP8DistributedDataParallel', 'FP8DistributedOptimizer']
