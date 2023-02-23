# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for TypeCast."""

import torch

from msamp.common.dtype.dtypes import Dtypes
from msamp.common.tensor.meta import ScalingMeta
from msamp.common.tensor.cast import TypeCast
from tests.helper import decorator


@decorator.cuda_test
def test_cast_fp8():
    """Test the cast_to_fp8 and cast_from_fp8 functions in TypeCast."""
    torch.manual_seed(100)
    input_fp16 = torch.rand((4, 4), dtype=torch.float16, device='cuda')
    meta = ScalingMeta(Dtypes.kfloat8_e4m3)
    output_fp8 = TypeCast.cast_to_fp8(input_fp16, meta)
    output_fp16 = TypeCast.cast_from_fp8(output_fp8, meta, Dtypes.kfloat16)

    assert torch.allclose(input_fp16, output_fp16, 0, 0.1)


@decorator.cuda_test
def test_cast_fp16():
    """Test the cast_to_fp16 and cast_from_fp16 functions in TypeCast."""
    torch.manual_seed(100)
    input_fp32 = torch.rand((4, 4), device='cuda')
    meta = ScalingMeta(Dtypes.kfloat16)
    output_fp16 = TypeCast.cast_to_fp16(input_fp32, meta)
    output_fp32 = TypeCast.cast_from_fp16(output_fp16, meta, Dtypes.kfloat32)

    assert torch.allclose(input_fp32, output_fp32, 0, 1e-03)
