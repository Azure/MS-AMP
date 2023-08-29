# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.te.replacer module."""

import unittest

import torch
import transformer_engine.pytorch as te

from tests.helper import decorator
from msamp.te.replacer import TeReplacer


class TeReplacerTestCase(unittest.TestCase):
    """Test TeExtention overrider."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)
        self.hidden_size = 4096
        self.ffn_hidden_size = 16384
        self.num_attention_heads = 32
        self.dtype = torch.float16

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_replace(self):
        """Test replace function in TeReplacer."""
        te_transformer = te.TransformerLayer(self.hidden_size, self.ffn_hidden_size, self.num_attention_heads)
        te_transformer.to(dtype=self.dtype).cuda()
        model = TeReplacer.replace(te_transformer)

        msamp_modules = []

        def _check_model(model):
            if type(model) in TeReplacer.module_weight_names:
                msamp_modules.append(model)
            for _, child in list(model.named_children()):
                _check_model(child)

        _check_model(model)
        assert len(msamp_modules) == 3
