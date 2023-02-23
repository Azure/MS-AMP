# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for dtypes module."""

import torch
from msamp.common.dtype import Dtypes


def test_is_fp8_qtype():
    """Test is_fp8_qtype function in Dtypes."""
    dtype_to_isfp8 = {
        Dtypes.kbyte: False,
        Dtypes.kint32: False,
        Dtypes.kfloat32: False,
        Dtypes.kfloat16: False,
        Dtypes.kbfloat16: False,
        Dtypes.kfloat8_e4m3: True,
        Dtypes.kfloat8_e5m2: True
    }

    for k, v in dtype_to_isfp8.items():
        assert Dtypes.is_fp8_qtype(k) == v


def test_qtype_to_dtype():
    """Test qtype_to_dtype in Dtypes."""
    qtype_to_dtype = {
        Dtypes.kfloat16: torch.float16,
        Dtypes.kbfloat16: torch.bfloat16,
        Dtypes.kfloat32: torch.float32,
        Dtypes.kfloat8_e4m3: torch.fp8e4m3,
        Dtypes.kfloat8_e5m2: torch.fp8e5m2
    }
    assert qtype_to_dtype == Dtypes.qtype_to_dtype
