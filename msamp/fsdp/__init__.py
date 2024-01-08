# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP fsdp package."""

from msamp.fsdp.replacer import FsdpReplacer
from msamp.fsdp.fully_sharded_data_parallel import FP8FullyShardedDataParallel

__all__ = ['FsdpReplacer', 'FP8FullyShardedDataParallel']
