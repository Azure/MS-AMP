# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for AMP module."""

import unittest
import torch
import numpy as np

from msamp.common.dtype import Dtypes
from tests.helper import decorator


class AMPTestCase(unittest.TestCase):
    """ A class for testing AMP module. """
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(100)
        self.size = (4, 4)

    def _helper_test_grad_scaling_unscale(self, device, dtype, qtype=None):
		# Adapted from https://github.com/pytorch/pytorch/blob/master/test/test_cuda.py
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device=device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device)

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
                np.testing.assert_almost_equal(grad.float().data.cpu().numpy(),
                    (old_grad.float() * inv_scale).data.cpu().numpy())

    def test_grad_scaling_unscale_cpu(self):
        dtypes = [torch.float16, torch.float32]
        for dtype in dtypes: 
            self._helper_test_grad_scaling_unscale("cpu", dtype=dtype)

    @decorator.cuda_test
    def test_grad_scaling_unscale_cuda(self):
        dtypes = [torch.float16, torch.float32]
        qtypes = [Dtypes.kfloat16, Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2]
        for dtype in dtypes:
            self._helper_test_grad_scaling_unscale("cuda", dtype=dtype)

        for dtype in dtypes:
            for qtype in qtypes:
                self._helper_test_grad_scaling_unscale("cuda", dtype=dtype, qtype=qtype)
