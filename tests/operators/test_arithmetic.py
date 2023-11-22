# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for arithmetic module."""

import unittest

import torch

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.operators.arithmetic import Arithmetic
from msamp.common.tensor import TypeCast

class ArithmeticTestCase(unittest.TestCase):
    """A class for Arithmetic test cases."""

    def _check_scaling_tensor(self, scaling_tensor1, scaling_tensor2):
        assert torch.all(torch.eq(scaling_tensor1.value, scaling_tensor2.value))
        assert torch.all(torch.eq(scaling_tensor1.meta.scale, scaling_tensor2.meta.scale))
        assert torch.all(torch.eq(scaling_tensor1.meta.scale_inv, scaling_tensor2.meta.scale_inv))
        assert torch.all(torch.eq(scaling_tensor1.meta.amax, scaling_tensor2.meta.amax))

    @decorator.cuda_test
    def test_add_to_fp8(self):
        """Test the function Arithmetic.add_to_fp8()."""

        torch.manual_seed(100)
        sizes = (10000, 16384)

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            for qtype in [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2]:
                input1 = torch.rand(sizes, dtype=dtype, device='cuda')
                input2 = torch.rand(sizes, dtype=dtype, device='cuda')

                scaling_tensor1 = input1.cast(qtype)
                scaling_tensor2 = input1.cast(qtype)

                self._check_scaling_tensor(scaling_tensor1, scaling_tensor2)

                meta = scaling_tensor1.meta
                Arithmetic.add_to_fp8(scaling_tensor1.value, meta, input2)
                scaling_tensor2.copy_((scaling_tensor2.to(dtype) + input2).cast(qtype, meta=scaling_tensor2.meta))

                self._check_scaling_tensor(scaling_tensor1, scaling_tensor2)
