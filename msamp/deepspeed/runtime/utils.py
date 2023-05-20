# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Utils with MS-AMP support."""

from msamp.common.tensor import ScalingTensor
from deepspeed.runtime.utils import *


_origin_CheckOverflow = CheckOverflow


class CheckOverflow(_origin_CheckOverflow):
    @staticmethod
    def _has_inf_or_nan(x, i):
        if isinstance(x, ScalingTensor):
            return x.has_inf_or_nan()
        return _origin_CheckOverflow._has_inf_or_nan(x, i)
