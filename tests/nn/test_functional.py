# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for functional in MS-AMP."""

import itertools
import unittest
import torch
import torch.nn.functional as F

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.nn import LinearReplacer
from tests.helper import decorator


class FunctionalTestCase(unittest.TestCase):
    """Test cases for functional module."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    @decorator.cuda_test
    def test_linear_input_type(self):
        """Test linear input type."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)

        # input, weight, bias
        valid_types = {
            (torch.Tensor, torch.Tensor, torch.Tensor),
            (torch.Tensor, torch.Tensor, type(None)),
            (torch.Tensor, ScalingTensor, torch.Tensor),
            (torch.Tensor, ScalingTensor, type(None)),
        }

        def _is_valid_types(*args):
            assert len(args) == 3
            for valid_type in valid_types:
                if all(isinstance(v, t) for t, v in zip(valid_type, args)):
                    return True
            return False

        values = [None, input, model.weight]
        for a, b, c in itertools.product(values, values, [model.bias, None]):
            with self.subTest(a=type(a), b=type(b), c=type(c)):
                if _is_valid_types(a, b, c):
                    F.linear(a, b, bias=c)
                else:
                    with self.assertRaises(TypeError):
                        F.linear(a, b, bias=c)

        # check weight w/o scaling_metas
        weight2 = model.weight.clone()
        with self.assertRaises(ValueError):
            F.linear(input, weight2, bias=model.bias)

        # check weight w/ scaling_metas
        F.linear(input, model.weight, bias=model.bias)

    @decorator.cuda_test
    def test_linear_forward_backward(self):
        """Test linear forward and backward."""
        input = torch.randn((4, 4), device='cuda')
        input1 = input.clone()
        input2 = input.clone()

        input1.requires_grad = True
        input2.requires_grad = True

        linear = torch.nn.Linear(4, 4).cuda()
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)
        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16)

        output1 = model1(input1)
        output1.sum().backward()

        output2 = F.linear(input2, model2.weight, bias=model2.bias)
        output2.sum().backward()

        self.assertTrue(torch.equal(output1, output2))
        self.assertTrue(torch.equal(model1.weight.grad.float(), model2.weight.grad.float()))
        self.assertTrue(torch.equal(model1.bias.grad, model2.bias.grad))
        self.assertTrue(torch.equal(input1.grad, input2.grad))

        # Check Scaling Metas
        meta_names = set(model1.scaling_metas.keys())
        self.assertEqual(len(meta_names ^ set(model2.scaling_metas.keys())), 0)
        for meta_name in meta_names:
            meta1 = model1.scaling_metas[meta_name]
            meta2 = model2.scaling_metas[meta_name]
            self.assertTrue(torch.equal(meta1.scale, meta2.scale))
            self.assertTrue(torch.equal(meta1.scale_inv, meta2.scale_inv))
            self.assertTrue(torch.equal(meta1.amax, meta2.amax))
