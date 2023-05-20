# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for deepspeed optimizer with MS-AMP."""

import unittest
import torch

from msamp import deepspeed
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.optim import LBAdamW
from tests.helper import decorator


class DeepSpeedTestCase(unittest.TestCase):
    """Test functions in deepspeed optimizer with MS-AMP."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    @decorator.cuda_test
    def test_fused_optimizer(self):
        """Test fused optimizer."""
        input = torch.randn(4, 4, device='cuda')
        model = torch.nn.Linear(4, 4, bias=False).cuda()
        model1 = LinearReplacer.replace(model, Dtypes.kfloat16)
        model2 = LinearReplacer.replace(model, Dtypes.kfloat16)
        assert torch.equal(model1.weight.float(), model2.weight.float())
        opt1 = LBAdamW(list(model1.parameters()))
        opt2 = LBAdamW(list(model2.parameters()))

        config = {
            'train_batch_size': 1,
            'train_micro_batch_size_per_gpu': 1,
        }

        model2, opt2, _, _ = deepspeed.initialize(
            model=model2,
            optimizer=opt2,
            config=config,
        )

        # In deepspeed, model2.weight has been converted to FP8 E4M3.
        assert torch.equal(model1.weight.cast(Dtypes.kfloat8_e4m3).float(), model2.weight.float())
        model1(input).sum().backward()
        opt1.step()
        model2.backward(model2(input).sum())
        model2.step()
        assert torch.equal(model1.weight.cast(Dtypes.kfloat8_e4m3).float(), model2.weight.float())
