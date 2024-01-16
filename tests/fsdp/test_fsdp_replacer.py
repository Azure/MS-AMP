# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.fsdp.replacer module."""

import unittest
import copy
import math

import torch
import torch.nn as nn

from tests.helper import decorator
from msamp.fsdp import FsdpReplacer


class FsdpReplacerTestCase(unittest.TestCase):
    """Test TeExtention overrider."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_replace(self):
        """Test replace function in FsdpReplacer."""
        model = nn.Linear(5, 10)
        model1 = copy.deepcopy(model)

        model2 = FsdpReplacer.replace(model)

        params1 = list(model1.parameters())
        params2 = list(model2.parameters())

        assert len(params1) == len(params2)

        param1 = params1[0]
        param2 = params2[0]
        assert isinstance(param1, torch.nn.Parameter)
        assert isinstance(param2, torch.nn.Parameter)
        assert param2.numel() == int(math.ceil(param1.numel() / 4))
        assert param2._original_shape == torch.Size([10, 5])
        assert param2._padded == 4 - (5 * 10) % 4
        assert param2._meta is not None
        assert param2._scaling_metas is not None
