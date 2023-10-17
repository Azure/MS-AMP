# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.te.replacer module."""

import unittest

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from tests.helper import decorator
from msamp.nn import ScalingParameter
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
        self.batch_size = 4
        self.sequence_length = 128

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_replace(self):
        """Test replace function in TeReplacer."""
        # fused attention need cuda version >= 12.1
        if torch.version.cuda < '12.1':
            return
        te_transformer = te.TransformerLayer(
            self.hidden_size, self.ffn_hidden_size, self.num_attention_heads, fuse_qkv_params=True
        )
        te_transformer.to(dtype=self.dtype).cuda()

        model = TeReplacer.replace(te_transformer)
        msamp_module_cnt = 0

        def _check_model(model):
            if type(model) in TeReplacer.module_weight_names:
                nonlocal msamp_module_cnt
                msamp_module_cnt += 1
                weights = TeReplacer.module_weight_names[type(model)]
                for weight in weights:
                    if not hasattr(model, weight):
                        continue
                    weight = getattr(model, weight)
                    assert isinstance(weight, ScalingParameter)

            for _, child in list(model.named_children()):
                _check_model(child)

        _check_model(model)
        assert msamp_module_cnt == 3

        scaling_params = [p for p in model.parameters() if isinstance(p, ScalingParameter)]
        assert len(scaling_params) == 4
        is_fp8_available, _ = te.fp8.is_fp8_available()
        if is_fp8_available:
            # Do a forward pass to make sure the model is working.
            fp8_format = Format.HYBRID
            fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
            x = torch.rand(self.sequence_length, self.batch_size, self.hidden_size).cuda().to(dtype=self.dtype)

            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                y = model(x, attention_mask=None)
                assert y.shape == (self.sequence_length, self.batch_size, self.hidden_size)
            y.sum().backward()
