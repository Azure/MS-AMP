# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for FP4 quantization operator."""

import itertools
import unittest

import torch

from tests.helper import decorator
from msamp.operators.fp4_quantize import FP4_QUANTIZER


class FP4QuantTestCase(unittest.TestCase):
    '''A class for FP4 quantization test cases.'''
    @decorator.cuda_test
    def test_DGE(self):
        '''Check the DGE item.'''
        total_points = 20
        x_values = torch.linspace(-6.0, 6.0, total_points).to(torch.bfloat16).cuda()
        excepted_y_values = torch.tensor(
            [0.2002, 0.4375, 0.6055, 0.2168, 1.8359, 0.2695, 0.3027, 0.2695, 0.2402, 0.5781, 
             0.5781, 0.2402, 0.2695, 0.3027, 0.2695, 1.8359, 0.2168, 0.6055, 0.4375, 0.2002], dtype=torch.bfloat16).cuda()
        differentiable_quantized_y_derivative = FP4_QUANTIZER.apply_DGE_item(x_values)
        self.assertTrue(torch.allclose(differentiable_quantized_y_derivative, excepted_y_values))


    @decorator.cuda_test
    def test_fp4_quant(self):
        '''Check the quantization of input tensor.'''
        input_tensor = torch.tensor([[[0.001, 0.048, 0.0997], [0.1503, 0.2002, 0.2497], [0.2974, 0.30699, 0.4001]]], dtype=torch.bfloat16).cuda()
        target_tensor = torch.tensor([[[0.0, 0.0625, 0.125], [0.125, 0.1875, 0.25], [0.25, 0.25, 0.375]]], dtype=torch.bfloat16).cuda()
        output_tensor = FP4_QUANTIZER.quantize_simu_fp4_in_bf16(input_tensor, format='e2m1', nan_existed=False)
        self.assertTrue(torch.allclose(output_tensor, target_tensor))

        input_tensor = torch.tensor(
            [ [ [-0.01,  0.48,   -9.67], 
                [1.623,  -2.222, 24.67], ],
              [ [-2.874, 3.699,  -34.57], 
                [0.85,   -1.343, 18.88], ]
            ], dtype=torch.bfloat16).cuda()        # channel-wise outlier. shape: (2, 2, 3)
        target_tensor = torch.tensor(
            [ [ [ 0.0,  0.5,  -8.0],
                [1.5,  -2.0,  24.0], ],
              [ [-3.0,  4.0, -32.0],
                [0.75,  -1.5,  16.0], ]
            ], dtype=torch.bfloat16).cuda()
        output_tensor = FP4_QUANTIZER.quantize_simu_fp4_in_bf16(input_tensor, format='e2m1', nan_existed=False, channel_wise=True)
        self.assertTrue(torch.allclose(output_tensor, target_tensor))
        
        output_tensor = FP4_QUANTIZER.quantize_simu_fp4_in_bf16(input_tensor.view(-1, 3).T, format='e2m1', nan_existed=False, token_wise=True)      # token-wise outlier.
        self.assertTrue(torch.allclose(output_tensor, target_tensor.view(-1, 3).T))
        
