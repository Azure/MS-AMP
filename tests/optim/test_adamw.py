# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for adamw optimizer."""

import copy
import unittest
import torch

from msamp.common.dtype import Dtypes
from msamp.optim import LBAdamW, LBAdam, LBAdamWBase
from msamp.nn import LinearReplacer
from tests.helper import decorator


class LBAdamwTestCase(unittest.TestCase):
    """Test LBAdamW optimizers."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_adamw_step(self):
        """Test adamw optimizer step function."""
        self.check_optimizer_step(LBAdamWBase)
        self.check_optimizer_step(LBAdamW)
        self.check_optimizer_step(LBAdam)

    @decorator.cuda_test
    def test_state_dict(self):
        """Test state dict of LBAdamW and LBAdam."""
        self.check_optimizer_state_dict(LBAdamW)
        self.check_optimizer_state_dict(LBAdam)

    def check_optimizer_step(self, optimizer_class, diff=3e-4):
        """Check the difference between torch.optim.AdamW and optimizer_class optimizers.

        Args:
            optimizer_class (class): LBAdamWBase, LBAdamW or LBAdam.
            diff (float, optional): The difference between torch.optim.AdamW and optimizer_class optimizers.
        """
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()

        # test torch.optim.AdamW
        model1 = copy.deepcopy(linear)
        opt1 = torch.optim.AdamW(model1.parameters())

        for _ in range(4):
            output = model1(input)
            output.sum().backward()
            opt1.step()
            opt1.zero_grad()

        # test msamp.optim.LBAdamWBase
        model2 = copy.deepcopy(linear)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)

        opt2 = optimizer_class(model2.parameters())
        opt2.set_model(model2)

        for _ in range(4):
            output = model2(input)
            output.sum().backward()
            opt2.step()
            opt2.zero_grad()

        self.assertTrue(torch.allclose(model1.weight, model2.weight.float(), 0, diff))

    def check_optimizer_state_dict(self, lbadam_class):
        """Save and load state dict of lbadam_class optimizer and check if the value is excepted.

        Args:
            lbadam_class (class): LBAdamW or LBAdam.
        """
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()

        # build model1 and update weight 4 times.
        model1 = copy.deepcopy(linear)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        opt1 = lbadam_class(model1.parameters())
        opt1.set_model(model1)

        for _ in range(4):
            output = model1(input)
            opt1.zero_grad()
            output.sum().backward()
            opt1.step()

        state_dict1 = opt1.state_dict()

        # Check if the content of state_dict is expected.
        self.assertEqual(list(state_dict1.keys()), ['state', 'param_groups'])

        state = state_dict1['state']
        self.assertEqual(list(state.keys()), [0, 1])
        self.assertEqual(list(state[0].keys()), ['step', 'exp_avg', 'exp_avg_sq'])
        self.assertEqual(state[0]['step'], 4)

        for i in range(0, 2):
            exp_avg = state[i]['exp_avg']['state']
            self.assertEqual(type(exp_avg), torch.Tensor)
            self.assertEqual(exp_avg.dtype, torch.uint8)
            exp_avg_sq = state[0]['exp_avg_sq']['state']
            self.assertEqual(type(exp_avg_sq), torch.Tensor)
            self.assertEqual(exp_avg_sq.dtype, torch.float16)

        param_groups = state_dict1['param_groups']
        self.assertEqual(len(param_groups), 1)
        self.assertEqual(param_groups[0]['lr'], 0.001)
        self.assertEqual(param_groups[0]['betas'], (0.9, 0.999))
        self.assertEqual(param_groups[0]['eps'], 1e-08)

        # Backup state dict and update weight once.
        state_dict2 = copy.deepcopy(state_dict1)
        opt1.zero_grad()
        model1(input).sum().backward()
        opt1.step()

        # Build model2 and update 4 times.
        model2 = copy.deepcopy(linear)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)

        opt2 = lbadam_class(model2.parameters())
        opt2.set_model(model2)

        for _ in range(4):
            output = model2(input)
            opt2.zero_grad()
            output.sum().backward()
            opt2.step()

        # Load state dict to op2 and check if the weight is same as model1 after update weigth once.
        opt2 = lbadam_class(model2.parameters())
        opt2.load_state_dict(state_dict2)

        opt2.zero_grad()
        model2(input).sum().backward()
        opt2.step()

        self.assertTrue(torch.equal(model1.weight.value, model2.weight.value))
