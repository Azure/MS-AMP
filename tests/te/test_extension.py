# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.te.extention module."""

import unittest

import torch
import transformer_engine_extensions as tex
import transformer_engine.pytorch.cpp_extensions as texcpp

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingMeta
from msamp.te import extension    # noqa: F401


class TeExtentionOverriderTestCase(unittest.TestCase):
    """Test TeExtention overrider."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)
        self.size = (4, 4)
        self.device = 'cuda'

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_fused_cast_transpose(self):
        """Test fused_cast_transpose."""
        # cast with torch.tensor
        input = torch.randn(self.size, device=self.device)
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        meta.amax[0] = input.abs().max()
        meta.reset_scaling_factor()

        input_cast_1 = torch.empty(self.size, device=self.device, dtype=torch.uint8)
        transpose_cast_1 = torch.empty(self.size, device=self.device, dtype=torch.uint8)

        tex.fused_cast_transpose(
            input, meta.scale, meta.amax, meta.scale_inv, input_cast_1, transpose_cast_1, tex.DType.kFloat8E4M3
        )
        assert torch.equal(input_cast_1.t(), transpose_cast_1)

        # cast with ScalingTensor
        scaling_input = input.cast(Dtypes.kfloat32)
        input_cast_2 = torch.empty(self.size, device=self.device, dtype=torch.uint8)
        transpose_cast_2 = torch.empty(self.size, device=self.device, dtype=torch.uint8)
        scale_inv = torch.ones((), device=self.device)

        tex.fused_cast_transpose(
            scaling_input, None, None, scale_inv, input_cast_2, transpose_cast_2, tex.DType.kFloat8E4M3
        )
        assert torch.equal(input_cast_2.t(), transpose_cast_2)

        assert torch.equal(input_cast_1, input_cast_2)

    @decorator.cuda_test
    def test_cast_to_fp8(self):
        """Test cast_to_fp8."""
        # cast with torch.tensor
        input = torch.randn(self.size, device=self.device)
        scaling_meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        scaling_meta.amax[0] = input.abs().max()
        scaling_meta.reset_scaling_factor()
        scale = scaling_meta.scale.item()

        fp8_type = tex.DType.kFloat8E4M3
        meta = tex.FP8TensorMeta()
        meta.scale = torch.ones(1, dtype=torch.float32, device=self.device) * scale
        meta.scale_inv = torch.ones(1, dtype=torch.float32, device=self.device) / scale
        meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        meta.amax_history[0][0] = scaling_meta.amax[0]

        ret1 = texcpp.cast_to_fp8(input, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)

        # cast with ScalingTensor
        scaling_input = input.cast(Dtypes.kfloat32)
        fp8_type = tex.DType.kFloat8E4M3
        meta = tex.FP8TensorMeta()
        meta.scale = torch.ones(1, dtype=torch.float32, device=self.device) * scale
        meta.scale_inv = torch.ones(1, dtype=torch.float32, device=self.device) / scale
        meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        ret2 = texcpp.cast_to_fp8(scaling_input, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
        assert torch.equal(ret1, ret2)
