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
            Dtypes.kbfloat16: torch.bfloat16,
            Dtypes.kfloat32: torch.float32
        }

        for qtype, dtype in supported_qtype_dtypes.items():
            scaling_tensor = tensor.cast(qtype)
            self.assertTrue(scaling_tensor.dtype == dtype)
            self.assertTrue(scaling_tensor.qtype == qtype)

    @decorator.cuda_test
    def test_torch_tensor_fused_cast_transpose(self):
        """Test tensor.fused_cast_transpose function."""
        for qtype in [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2, Dtypes.kfloat16, Dtypes.kfloat32]:
            tensor = torch.randn(self.size, device=self.device).contiguous()
            if Dtypes.is_fp8_qtype(qtype):
                cast, t = tensor.fused_cast_transpose(qtype)
                self.assertTrue(torch.equal(tensor.cast(qtype).float(), cast.float()))
                self.assertTrue(torch.equal(tensor.cast(qtype).fp8_transpose().float(), t.float()))
            else:
                with self.assertRaises(TypeError):
                    tensor.fused_cast_transpose(qtype)

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

        self.assertEqual(scaling_tensor.data_ptr(), scaling_tensor.value.data_ptr())
        self.assertTrue(scaling_tensor.grad is None)
        self.assertTrue(scaling_tensor.is_cuda)
        self.assertEqual(scaling_tensor.shape, self.size)
        self.assertEqual(scaling_tensor.size(), self.size)
        self.assertEqual(scaling_tensor.numel(), self.size[0] * self.size[1])
        self.assertEqual(scaling_tensor.nelement(), self.size[0] * self.size[1])
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
        tensor_bak = tensor.clone()

        qtypes = [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2, Dtypes.kfloat16, Dtypes.kbfloat16, Dtypes.kfloat32]
        dtypes = [torch.uint8, torch.uint8, torch.float16, torch.bfloat16, torch.float32]

        def _allclose(input, other):
            return torch.allclose(input, other, rtol=1e-1, atol=1e-1)

        for qtype1, dtype1 in zip(qtypes, dtypes):
            for qtype2, dtype2 in zip(qtypes, dtypes):
                with self.subTest(qtype1=qtype1, qtype2=qtype2):
                    if not Dtypes.is_fp8_qtype(qtype1):
                        with self.subTest(msg='tensor to scaling tensor'):
                            tensor1 = tensor.to(dtype1)
                            tensor2 = tensor1.cast(qtype2)
                            self.assertTrue(_allclose(tensor2.float(), tensor), (tensor2, tensor))
                    if not Dtypes.is_fp8_qtype(qtype2):
                        with self.subTest(msg='scaling tensor to tensor'):
                            tensor1 = tensor.cast(qtype1)
                            tensor2 = tensor1.to(dtype2)
                            self.assertTrue(_allclose(tensor2.float(), tensor), (tensor2, tensor))
                    with self.subTest(msg='scaling tensor to scaling tensor'):
                        tensor1 = tensor.cast(qtype1)
                        tensor2 = tensor1.cast(qtype2)
                        self.assertTrue(_allclose(tensor2.float(), tensor), (tensor2, tensor))
                    # check if tensor is not changed
                    self.assertTrue(torch.equal(tensor, tensor_bak))

    @decorator.cuda_test
    def test_tensor_cast_to_scaling_fp32(self):
        """Test cast function to ScalingFP32 or ScalingBF16 in ScalingTensor."""
        for dtype in [Dtypes.kfloat32, Dtypes.kbfloat16]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([1.0 / 512], dtype=torch.float32, device=self.device)
                y = x.cast(dtype)
                self.assertTrue(x == y.float())

    @decorator.cuda_test
    def test_tensor_cast_with_exception_value(self):
        """Test cast function in ScalingTensor with exception value."""
        for dtype in [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2, Dtypes.kfloat16, Dtypes.kbfloat16, Dtypes.kfloat32]:
            with self.subTest(dtype=dtype):
                x = torch.randn((2, ), device=self.device)
                t = x.cast(dtype)
                self.assertTrue(torch.isfinite(t.meta.amax[0]))
                for exception_value in [float('nan'), float('inf'), float('-inf')]:
                    for full in [True, False]:
                        with self.subTest(exception_value=exception_value, full=full):
                            x2 = x.clone()
                            if full:
                                x2.fill_(exception_value)
                            else:
                                x2[-1] = exception_value
                            t2 = x2.cast(dtype)
                            self.assertFalse(torch.isfinite(t2.meta.amax[0]))

    @decorator.cuda_test
    def test_tensor_cast_from_scaling_tensor(self):
        """Test tensor cast from ScalingTensor."""
        fp16_value = torch.tensor([1.0 / (2**17)], dtype=torch.float16, device='cuda')
        fp32_value = fp16_value.float()
        for dtype in [torch.float16, torch.float32]:
            value = fp16_value.to(dtype)
            scaling_fp16 = value.cast(Dtypes.kfloat16)
            self.assertTrue(torch.allclose(fp32_value, scaling_fp16.float()), (fp32_value, scaling_fp16))
            # cast ScalingTensor (FP16) to ScalingTensor (FP8E4M3)
            scaling_fp8 = scaling_fp16.cast(Dtypes.kfloat8_e4m3)
            self.assertTrue(torch.allclose(fp32_value, scaling_fp8.float()), (fp32_value, scaling_fp8))

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
    def test_tensor_fp8_transpose(self):
        """Test fp8_transpose function in ScalingTensor."""
        for qtype in [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2, Dtypes.kfloat16, Dtypes.kfloat32]:
            scaling_tensor = torch.randn(self.size, device=self.device).cast(qtype)
            if Dtypes.is_fp8_qtype(qtype):
                self.assertTrue(torch.equal(scaling_tensor.float().t(), scaling_tensor.fp8_transpose().float()))
            else:
                with self.assertRaises(TypeError):
                    scaling_tensor.fp8_transpose()

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
