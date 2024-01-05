# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP fsdp package."""

from msamp.fsdp.flat_param import FP8FlatParamHandle
from msamp.fsdp.fully_sharded_data_parallel import FP8FullyShardedDataParallel

__all__ = ['FP8FlatParamHandle', 'FP8FullyShardedDataParallel']