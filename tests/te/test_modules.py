# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.te.modules module."""

import unittest

import torch
import transformer_engine.pytorch as te

from tests.helper import decorator
from msamp.te.modules import MSAMPLinear, MSAMPLayerNormLinear, MSAMPLayerNormMLP


class TeModuleOverriderTestCase(unittest.TestCase):
    """Test TeModule overrider."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_modules(self):
        """Test modules overrided by MS-AMP."""
        te_linear = te.Linear(4, 4)
        assert isinstance(te_linear, MSAMPLinear)

        te_layernorm_linear = te.LayerNormLinear(4, 4)
        assert isinstance(te_layernorm_linear, MSAMPLayerNormLinear)

        te_layernorm_mlp = te.LayerNormMLP(4, 4)
        assert isinstance(te_layernorm_mlp, MSAMPLayerNormMLP)
