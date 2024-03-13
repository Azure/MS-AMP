# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for floating module."""

import torch

from msamp.common.utils import Device, GPUType
from msamp.common.dtype import Floating
from msamp.common.dtype import Dtypes


def test_fp_max():
    """Test fp_max in Floating class."""
    fp8e4m3_max = Device.get_gpu_type() == GPUType.AMD and 240.0 or 448.0
    expect_fp_maxs = {torch.fp8e4m3: fp8e4m3_max, torch.fp8e5m2: 57344, torch.float16: 65504}

    for k, v in expect_fp_maxs.items():
        assert Floating.fp_maxs[k] == v


def test_qfp_max():
    """Test fp_max in Floating class."""
    fp8e4m3_max = Device.get_gpu_type() == GPUType.AMD and 240.0 or 448.0
    expected_qfp_maxs = {Dtypes.kfloat8_e4m3: fp8e4m3_max, Dtypes.kfloat8_e5m2: 57344, Dtypes.kfloat16: 65504}

    for k, v in expected_qfp_maxs.items():
        assert Floating.qfp_max[k] == v
