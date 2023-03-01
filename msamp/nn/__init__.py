# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP nn package."""

from msamp.nn.parameter import ScalingParameter
from msamp.nn.module import ScalingModule
from msamp.nn.state import model_state
from msamp.nn.linear import FP8Linear, LinearReplacer

__all__ = ['ScalingParameter', 'ScalingModule', 'model_state', 'FP8Linear', 'LinearReplacer']
