# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gemm module."""

import unittest

import torch

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.operators.gemm import Gemm


class GemmTestCase(unittest.TestCase):
    """A class for Gemm test cases.

    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    """
    def test_round2times(self):
        """Test the function Gemm._round2times()."""
        test_cases = [
            {
                'value': 1,
                'base': 16,
                'expected': 16
            }, {
                'value': 16,
                'base': 16,
                'expected': 16
            }, {
                'value': 17,
                'base': 16,
                'expected': 32
            }
        ]

        for case in test_cases:
            assert (Gemm._round2times(case['value'], case['base']) == case['expected'])

    @decorator.cuda_test
    def test_fp8_gemm(self):
        """Test the function Gemm.fp8_gemm()."""
        out_qtype = Dtypes.kfloat8_e4m3
        tensorA = torch.ones((4, 2), dtype=torch.float32, device='cuda')
        tensorB = torch.ones((3, 2), dtype=torch.float32, device='cuda')
        scaling_tensorA = tensorA.cast(out_qtype)
        scaling_tensorB = tensorB.cast(out_qtype)
        out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, Dtypes.kfloat32)

        expected = torch.matmul(tensorB, tensorA.t())
        assert (out.equal(expected))

        out = torch.ones((3, 4), dtype=torch.float32, device='cuda')
        out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, Dtypes.kfloat32, out=out)
        assert (out.equal(expected))
