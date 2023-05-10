# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for linear module in MS-AMP."""

import io
import itertools
import copy
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
