# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP tensor package."""

from msamp.common.tensor.cast import TypeCast
from msamp.common.tensor.hook import HookManager
from msamp.common.tensor.meta import ScalingMeta
from msamp.common.tensor.tensor import ScalingTensor
from msamp.common.tensor.tensor_dist import TensorDist

__all__ = ['TypeCast', 'HookManager', 'ScalingMeta', 'ScalingTensor', 'TensorDist']
