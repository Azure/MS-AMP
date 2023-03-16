# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ScalingTensor."""

import unittest
import torch
import numpy as np

from msamp.common.dtype import Dtypes
from msamp.common.tensor import TypeCast
from msamp.common.tensor import ScalingMeta
from msamp.common.tensor import ScalingTensor
from tests.helper import decorator


class ScalingTensorTestCase(unittest.TestCase):
    """Test ScalingTensor."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(100)
        self.size = (4, 4)
        self.device = 'cuda'

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_torch_tensor_cast(self):
        """Test overrided tensor.cast functions."""
        tensor = torch.randn(self.size, device=self.device)

        supported_qtype_dtypes = {
            Dtypes.kfloat8_e4m3: torch.uint8,
            Dtypes.kfloat8_e5m2: torch.uint8,
            Dtypes.kfloat16: torch.float16,
            Dtypes.kfloat32: torch.float32
        }

        for qtype, dtype in supported_qtype_dtypes.items():
            scaling_tensor = tensor.cast(qtype)
            self.assertTrue(scaling_tensor.dtype == dtype)
            self.assertTrue(scaling_tensor.qtype == qtype)

        with self.assertRaises(TypeError):
            tensor.cast(Dtypes.kbfloat16)

    @decorator.cuda_test
    def test_torch_unary_funcs(self):
        """Test overrided tensor unary functions."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)

        # test torch.zero_like
        zero_tensor = torch.zeros(self.size, device=self.device)
        self.assertTrue(torch.equal(zero_tensor, torch.zeros_like(scaling_tensor)))
        self.assertTrue(torch.equal(zero_tensor, torch.zeros_like(tensor)))

        # test torch.ones_like
        one_tensor = torch.ones(self.size, device=self.device)
        self.assertTrue(torch.equal(one_tensor, torch.ones_like(scaling_tensor)))
        self.assertTrue(torch.equal(one_tensor, torch.ones_like(tensor)))

    @decorator.cuda_test
    def test_tensor_basic_funcs(self):
        """Test basic functions in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        scaling_tensor = ScalingTensor(TypeCast.cast_to_fp8(tensor, meta), meta=meta)

        self.assertTrue(scaling_tensor.grad is None)
        self.assertTrue(scaling_tensor.is_cuda)
        self.assertEqual(scaling_tensor.shape, self.size)
        self.assertEqual(scaling_tensor.size(), self.size)
        self.assertEqual(scaling_tensor.numel(), self.size[0] * self.size[1])
        self.assertEqual(scaling_tensor.device, tensor.device)
        self.assertEqual(scaling_tensor.dtype, torch.uint8)
        self.assertEqual(scaling_tensor.type(), 'msamp.common.tensor.tensor.ScalingTensor')
        self.assertTrue(scaling_tensor.is_leaf)
        self.assertFalse(scaling_tensor.is_sparse)
        self.assertTrue(scaling_tensor.is_contiguous())
        self.assertFalse(scaling_tensor.is_complex())
        self.assertEqual(len(scaling_tensor), self.size[0])

    @decorator.cuda_test
    def test_is_floating_point(self):
        """Test is_floating_point function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        self.assertEqual(torch.is_floating_point(scaling_tensor), scaling_tensor.is_floating_point())

    @decorator.cuda_test
    def test_tensor_to(self):
        """Test to function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)

        supported_dtypes = [torch.float, torch.float16, torch.bfloat16]
        for dtype in supported_dtypes:
            tensor = scaling_tensor.to(dtype)
            self.assertEqual(type(tensor), torch.Tensor)
            self.assertEqual(tensor.dtype, dtype)

        with self.assertRaises(TypeError):
            scaling_tensor.to(torch.uint8)

        # test unique dtype
        with self.assertRaises(TypeError):
            scaling_tensor.to(torch.float16, torch.float32)

    @decorator.cuda_test
    def test_tensor_cast(self):
        """Test cast function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat32)
        # kfloat32 can cast to any valid qtype
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat8_e4m3).qtype, Dtypes.kfloat8_e4m3)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat8_e5m2).qtype, Dtypes.kfloat8_e5m2)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat16).qtype, Dtypes.kfloat16)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat32).qtype, Dtypes.kfloat32)

        # kfloat16 can cast to kfloat8_e4m3 kfloat16 kfloat32
        scaling_tensor = tensor.cast(Dtypes.kfloat16)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat8_e4m3).qtype, Dtypes.kfloat8_e4m3)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat16).qtype, Dtypes.kfloat16)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat32).qtype, Dtypes.kfloat32)
        with self.assertRaises(TypeError):
            scaling_tensor.cast(Dtypes.kfloat8_e5m2)

        # kfloat8_e4m3 can cast to kfloat8_e4m3 kfloat32
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat8_e4m3).qtype, Dtypes.kfloat8_e4m3)
        self.assertEqual(scaling_tensor.cast(Dtypes.kfloat32).qtype, Dtypes.kfloat32)
        with self.assertRaises(TypeError):
            scaling_tensor.cast(Dtypes.kfloat16)
        with self.assertRaises(TypeError):
            scaling_tensor.cast(Dtypes.kfloat8_e5m2)

    @decorator.cuda_test
    def test_tensor_mul(self):
        """Test mul function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        float_tensor1 = scaling_tensor.float()
        scaling_tensor.mul_(torch.tensor((2.0, ), device=self.device))
        float_tensor2 = scaling_tensor.float()

        self.assertTrue(torch.equal(float_tensor1 * 2.0, float_tensor2))
        scaling_tensor.mul_(2.0)
        scaling_tensor3 = scaling_tensor.float()
        self.assertTrue(torch.equal(float_tensor2 * 2.0, scaling_tensor3))

    @decorator.cuda_test
    def test_tensor_div(self):
        """Test div function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        float_tensor1 = scaling_tensor.float()
        scaling_tensor.div_(torch.tensor((2.0, ), device=self.device))
        float_tensor2 = scaling_tensor.float()
        self.assertTrue(torch.equal(float_tensor1 / 2.0, float_tensor2))
        scaling_tensor.div_(2.0)
        float_tensor3 = scaling_tensor.float()
        self.assertTrue(torch.equal(float_tensor2 / 2.0, float_tensor3))

    @decorator.cuda_test
    def test_tensor_transpose(self):
        """Test transpose function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        float_tensor = scaling_tensor.float()
        transpose_tensor_value = scaling_tensor.t().contiguous().float()
        self.assertTrue(torch.equal(float_tensor.t(), transpose_tensor_value))

    @decorator.cuda_test
    def test_inf_and_nan(self):
        """Test has_inf_or_nan function in ScalingTensor."""
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        self.assertFalse(scaling_tensor.has_inf_or_nan())

        tensor = torch.tensor([1, 2, 3, 4, 5, torch.inf], dtype=torch.float32, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        self.assertTrue(scaling_tensor.has_inf_or_nan())

        tensor = torch.tensor([1, 2, 3, 4, 5, torch.nan], dtype=torch.float32, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        self.assertTrue(scaling_tensor.has_inf_or_nan())

    @decorator.cuda_test
    def test_tensor_zero(self):
        """Test zero function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        scaling_tensor.zero_()
        self.assertTrue((scaling_tensor.float() == 0).all())

    @decorator.cuda_test
    def test_tensor_min_max(self):
        """Test min and max function in ScalingTensor."""
        tensor = torch.randn(self.size, device=self.device)
        scaling_tensor = tensor.cast(Dtypes.kfloat8_e4m3)
        self.assertEqual(scaling_tensor.max().item(), scaling_tensor.float().max().item())
        self.assertEqual(scaling_tensor.min().item(), scaling_tensor.float().min().item())

    @decorator.cuda_test
    def test_tensor_cast_with_updating_factors(self):
        """Test cast function with updating scaling factors."""
        for dtype in [Dtypes.kfloat16, Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2]:
            tensor = torch.randn(self.size, device=self.device)
            scaling_tensor = tensor.cast(dtype)
            # update scale but scale_inv is unchanged.
            scaling_tensor.meta.scale *= 2
            scaling_tensor2 = tensor.cast(dtype)
            self.assertTrue(torch.equal(scaling_tensor.float(), scaling_tensor2.float()))

    def _helper_test_grad_check_unscale(self, device, dtype, qtype=None):
        """Helper function for testing grad scaling and unscale.

        Adapted from https://github.com/pytorch/pytorch/blob/master/test/test_cuda.py
        """
        inv_scale = torch.full((1, ), 0.25, dtype=torch.float, device=device)
        found_inf = torch.full((1, ), 0.0, dtype=torch.float, device=device)

        size = 10
        g = torch.randn((size, size), dtype=dtype, device=device)

        ginf = g.clone()
        ginf[2, 2] = float('inf')
        gnan = g.clone()
        gnan[2, 2] = float('nan')

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            if qtype is not None:
                # convert to ScalingTensor
                grads = [grad.cast(qtype) for grad in grads]
            old_grads = [grad.clone() for grad in grads]
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            assert found_inf.item() == has_inf
            for grad, old_grad in zip(grads, old_grads):
                np.testing.assert_almost_equal(
                    grad.float().data.cpu().numpy(), (old_grad.float() * inv_scale).data.cpu().numpy()
                )

    def test_grad_check_unscale_cpu(self):
        """Test grad scaling and unscale on CPU."""
        dtypes = [torch.float16, torch.float32]
        for dtype in dtypes:
            self._helper_test_grad_check_unscale('cpu', dtype=dtype)

    @decorator.cuda_test
    def test_grad_check_unscale_cuda(self):
        """Test grad scaling and unscale on CUDA."""
        dtypes = [torch.float16, torch.float32]
        qtypes = [Dtypes.kfloat16, Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2]
        for dtype in dtypes:
            self._helper_test_grad_check_unscale('cuda', dtype=dtype)

        for dtype in dtypes:
            for qtype in qtypes:
                self._helper_test_grad_check_unscale('cuda', dtype=dtype, qtype=qtype)
