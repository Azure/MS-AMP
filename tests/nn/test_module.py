# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ScalingModule."""

import torch

from msamp.common.dtype import Dtypes
from msamp.nn import ScalingModule
from msamp.nn import ScalingParameter
from tests.helper import decorator


@decorator.cuda_test
def test_scaling_module():
    """Test functions in ScalingModule."""
    tensor = torch.randn((4, 4), device='cuda')
    scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
    parameter = ScalingParameter(scaling_tensor)
    module = ScalingModule()
    module.weight = parameter

    name_para_dict = {}
    for k, v in module.named_parameters():
        name_para_dict[k] = v

    assert 'weight' in name_para_dict
    assert name_para_dict['weight'] == parameter
