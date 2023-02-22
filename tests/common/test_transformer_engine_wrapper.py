# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for TransformerEngineWrapper module."""

import os
import unittest

import torch
import transformer_engine_extensions as tex

from tests.helper import decorator
from msamp.common.dtype.dtypes import Dtypes
import msamp.common.utils.TransformerEngineWrapper as tew


class TransformerEngineWrapperTestCase(unittest.TestCase):
    """A class for TransformerEngineWrapper test cases.

    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    """
    def test_to_compatible_args():
        types = ''
        test_cases = [
            {
                input: [1, [2], Dtypes.kbyte],
                exptected : [1, [2], tex.Dtypes.kByte]
            },
            {
                input:[Dtypes.kint32, Dtypes.kfloat32, Dtypes.kfloat16], 
                expected:[tex.Dtypes.kInt32, tex.Dtypes.kFloat32, tex.Dtypes.kFloat16]
            },
            {
                input: [Dtypes.kbfloat16, Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2, '3'],
                expected:[tex.Dtypes.kBFloat16,tex.Dtypes.kFloat8E4M3,tex.Dtypes.kFloat8E5M2, '3']
            }
        ]

        for case in test_cases:
            assert (tew._to_compatible_args(case['input']) == case['expected'])


    @decorator.cuda_test
    def test_cast_fp8():
        torch.manual_seed(100)
        input = torch.randn((4,4), device='cuda')
        amax = input.abs().max()
        scale = torch.ones((), device='cuda')
        fp8_tensor = tew.cast_to_fp8(input, scale, amax, 1.0 / scale, Dtypes.kfloat8_e4m3)
        assert (fp8_tensor.dtype == torch.uint8)
        output = tew.cast_from_fp8(fp8_tensor, 1.0 / scale, Dtypes.kfloat8_e4m3, Dtypes.kfloat32)
        assert (output.dtype == torch.float32)
        torch.allclose(input, output, 0, 0.2)

    def test_te_gemm():
        # TODO - will make up the tests after FP8Tensor module and gemm operator are added.
        pass