# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for deepspeed zero optimizer for fp8."""
import unittest

import torch
import torch.nn as nn

from msamp import deepspeed
from msamp.common.dtype.dtypes import Dtypes
from msamp.nn import LinearReplacer
from msamp.optim import LBAdam

from tests.helper import decorator


class FP8DeepSpeedZeroOptimizerTestCase(unittest.TestCase):
    """Test Fp8DeepSpeedZeroOptimizer."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)
        pass

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    def _backward(self, ds_config):
        model = nn.Linear(4, 4, device='cuda')
        model = LinearReplacer.replace(model, Dtypes.kfloat16)
        optimizer = LBAdam(list(model.parameters()))
        model, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)

        inputs = []
        num_inputs = 10
        for _ in range(num_inputs):
            inputs.append(torch.rand(4, 4, device='cuda'))

        losses = []
        epoches = 10
        for _ in range(epoches):
            total_loss = 0
            for i in range(num_inputs):
                output = model(inputs[i])
                loss = output.sum()
                model.backward(loss)
                total_loss += loss.item()
                model.step()
            avg_loss = total_loss / 10
            losses.append(avg_loss)

        for i in range(epoches):
            if i > 0:
                assert losses[i] < losses[i - 1]

    @decorator.cuda_test
    def test_stage1(self):
        """Test fp8 deepspeed zero-stage1."""
        config = {
            'train_batch_size': 1,
            'zero_optimization': {
                'stage': 1,
            }
        }
        self._backward(config)

    @decorator.cuda_test
    def test_stage2(self):
        """Test fp8 deepspeed zero-stage2."""
        config = {
            'train_batch_size': 1,
            'zero_optimization': {
                'stage': 2,
            }
        }
        self._backward(config)