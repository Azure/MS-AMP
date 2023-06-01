# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for deepspeed optimizer with MS-AMP."""

import os
import unittest
import torch

from msamp import deepspeed
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.optim import LBAdamW
from msamp.deepspeed.runtime.fp8.fused_optimizer import FP8Optimizer
from tests.helper import decorator


class FP8OptimizerTestCase(unittest.TestCase):
    """Test functions in deepspeed optimizer with MS-AMP."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)
        self.checkpoint_path = 'model.pth'

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

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

    @decorator.cuda_test
    def test_state_dict(self):
        """Test state dict."""
        model1 = torch.nn.Linear(4, 4).cuda()
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        optimizer1 = LBAdamW(list(model1.parameters()))
        fp8_optimizer1 = FP8Optimizer(optimizer1, static_loss_scale=128.0)

        for _ in range(10):
            input = torch.randn(4, 4, device='cuda')
            loss = model1(input).sum()
            loss.backward()
            optimizer1.step()

        checkpoint1 = {}
        checkpoint1['model'] = model1.state_dict()
        checkpoint1['optimizer'] = fp8_optimizer1.state_dict()
        torch.save(checkpoint1, self.checkpoint_path)

        model2 = torch.nn.Linear(4, 4).cuda()
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)
        optimizer2 = LBAdamW(list(model2.parameters()))
        fp8_optimizer2 = FP8Optimizer(optimizer2, dynamic_loss_scale=True)
        checkpoint2 = torch.load(self.checkpoint_path)
        model2.load_state_dict(checkpoint2['model'])
        fp8_optimizer2.load_state_dict(checkpoint2['optimizer'])

        assert torch.equal(model1.weight.float(), model2.weight.float())
        assert fp8_optimizer1.cur_scale == fp8_optimizer2.cur_scale

        # check master groups
        assert len(fp8_optimizer1.fp8_master_groups) == 1
        assert len(fp8_optimizer2.fp8_master_groups) == 1

        for i in range(len(fp8_optimizer1.fp8_master_groups[0])):
            assert torch.equal(
                fp8_optimizer1.fp8_master_groups[0][i].float(), fp8_optimizer2.fp8_master_groups[0][i].float()
            )
