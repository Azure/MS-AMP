# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Utils with MS-AMP support."""

from msamp.common.tensor import ScalingTensor
# flake8: noqa: F403
from deepspeed.runtime.utils import *

_origin_CheckOverflow = CheckOverflow


# flake8: noqa: F405
class CheckOverflow(_origin_CheckOverflow):
    """CheckOverflow with MS-AMP support."""
    @staticmethod
    def _has_inf_or_nan(x, i):
        """Check if the input tensor has inf or nan values."""
        if isinstance(x, ScalingTensor):
            return x.has_inf_or_nan()
        return _origin_CheckOverflow._has_inf_or_nan(x, i)
