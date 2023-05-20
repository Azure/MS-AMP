# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for deepspeed optimizer with MS-AMP."""

import unittest
import torch

from tests.helper import decorator
from msamp import deepspeed
from msamp.optim import LBAdamW


class DeepSpeedTestCase(unittest.TestCase):
    """Test functions in deepspeed optimizer with MS-AMP."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    @decorator.cuda_test
    def test_fused_optimizer(self):
        input = torch.randn(4, 4, device='cuda')
        model = torch.nn.Linear(4, 4, bias=False).cuda()
        model1 = LinearReplacer.replace(model, Dtypes.kfloat16)
        model2 = LinearReplacer.replace(model, Dtypes.kfloat16)
        assert torch.all(model1.weight.float(), model2.weight.float())
        opt1 = LBAdamW(list(model1.parameters()))
        opt2 = LBAdamW(list(model2.parameters()))

        config = {
            'train_batch_size': 1,
            'train_micro_batch_size_per_gpu': 1,
        }

        model2, opt2, _, _ = deepspeed.initialize(
            model=model2,
            optimizer=opt2,
            dist_init_required=False,
            config=config,
        )

        model1(input).sum().backward()
        opt1.step()
        model2.backward(model2(input).sum())
        model2.step()
        assert torch.all(model1.weight.float(), model2.weight.float())
