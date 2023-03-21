# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.initialize."""

import unittest
import torch

import msamp
from msamp.nn.linear import FP8Linear
from msamp.optim import LBAdamW, LBAdam
from tests.helper import decorator


class InitializeTestCase(unittest.TestCase):
    """Test initialize function in msamp."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_initialize(self):
        """Test initialize function."""
        model = torch.nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            msamp.initialize(model, torch.optim.AdamW(model.parameters()), 'O0')
        with self.assertRaises(ValueError):
            msamp.initialize(model, torch.optim.SGD(model.parameters()))

        for opt_level in ['O1', 'O2']:
            model = torch.nn.Linear(4, 4)
            model, optimizer = msamp.initialize(model, None, opt_level)
            assert isinstance(model, FP8Linear)
            assert isinstance(optimizer, LBAdamW)

        for opt_level in ['O1', 'O2']:
            model = torch.nn.Linear(4, 4)
            optimizer = torch.optim.Adam(model.parameters())
            model, optimizer = msamp.initialize(model, optimizer, opt_level)
            assert isinstance(model, FP8Linear)
            assert isinstance(optimizer, LBAdam)

        for opt_level in ['O1', 'O2']:
            model = torch.nn.Linear(4, 4)
            optimizer = torch.optim.AdamW(model.parameters())
            model, optimizer = msamp.initialize(model, optimizer, opt_level)
            assert isinstance(model, FP8Linear)
            assert isinstance(optimizer, LBAdamW)
