# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for adamw optimizer."""

import copy
import itertools
import unittest
import torch

from functools import partial

from msamp.common.dtype import Dtypes
from msamp.common.tensor import TensorDist, ScalingTensor
from msamp.optim import LBAdamW, LBAdam, LBAdamWBase, DSAdam
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
        dtypes = [torch.uint8, torch.float16]
        pairs = list(itertools.product(dtypes, dtypes)) + [[torch.float32, torch.float32]]
        for exp_avg_dtype, exp_avg_sq_dtype in pairs:
            with self.subTest(exp_avg_dtype=exp_avg_dtype, exp_avg_sq_dtype=exp_avg_sq_dtype):
                kwargs = dict(exp_avg_dtype=exp_avg_dtype, exp_avg_sq_dtype=exp_avg_sq_dtype)
                self.check_optimizer_step(torch.optim.AdamW, partial(LBAdamWBase, **kwargs))
                self.check_optimizer_step(torch.optim.AdamW, partial(LBAdamW, **kwargs))
                self.check_optimizer_step(torch.optim.AdamW, partial(DSAdam, **kwargs))
                self.check_optimizer_step(torch.optim.Adam, partial(LBAdam, **kwargs))

    @decorator.cuda_test
    def test_state_dict(self):
        """Test state dict of LBAdamW and LBAdam."""
        self.check_optimizer_state_dict(LBAdamW)
        self.check_optimizer_state_dict(LBAdam)

    def check_optimizer_step(self, optimizer_class1, optimizer_class2, diff=3e-4):
        """Check the difference between optimizer_class1 and optimizer_class2 optimizers.

        Args:
            optimizer_class1 (class): Optimizer Class
            optimizer_class2 (class): Optimizer Class
            diff (float, optional): The difference between optimizer_class1 and optimizer_class2 optimizers.
        """
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()
        wd = 1e-3
        steps = 4

        # test torch.optim.AdamW
        model1 = copy.deepcopy(linear)
        opt1 = optimizer_class1(model1.parameters(), weight_decay=wd)

        for _ in range(steps):
            output = model1(input)
            output.sum().backward()
            opt1.step()
            opt1.zero_grad()

        # test msamp.optim.LBAdamWBase
        model2 = copy.deepcopy(linear)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)

        opt2 = optimizer_class2(model2.parameters(), weight_decay=wd)

        for _ in range(steps):
            output = model2(input)
            output.sum().backward()
            opt2.all_reduce_grads(model2)
            opt2.step()
            opt2.zero_grad()

        self.assertTrue(torch.allclose(model1.weight, model2.weight.float(), 0, diff))

    def test_all_reduce_grads(self):
        """Test the function `all_reduce_grads`."""
        input = torch.randn(4, 4, device='cuda')
        model1 = torch.nn.Linear(4, 4).cuda()
        model2 = torch.nn.Linear(4, 4).cuda()
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)
        opt = LBAdamW(list(model1.parameters()) + list(model2.parameters()))
        loss = (model1(input) + model2(input)).sum()
        loss.backward()
        old_all_reduce_avg = TensorDist.all_reduce_avg
        num_grads = 0

        def debug_all_reduce_avg(grads):
            nonlocal num_grads
            num_grads += len(grads)
            return old_all_reduce_avg(grads)

        TensorDist.all_reduce_avg = debug_all_reduce_avg
        opt.all_reduce_grads(model1)
        self.assertEqual(num_grads, 1)
        opt.all_reduce_grads(model2)
        self.assertEqual(num_grads, 2)
        TensorDist.all_reduce_avg = old_all_reduce_avg

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

        for _ in range(4):
            output = model1(input)
            opt1.zero_grad()
            output.sum().backward()
            opt1.all_reduce_grads(model1)
            opt1.step()

        state_dict1 = opt1.state_dict()

        # Check if the content of state_dict is expected.
        self.assertEqual(list(state_dict1.keys()), ['state', 'param_groups'])

        state = state_dict1['state']
        self.assertEqual(list(state.keys()), [0, 1])
        self.assertEqual(list(state[0].keys()), ['step', 'exp_avg', 'exp_avg_sq'])
        self.assertEqual(state[0]['step'], 4)

        for i in range(0, 2):
            exp_avg = state[i]['exp_avg']
            self.assertEqual(type(exp_avg), ScalingTensor)
            self.assertEqual(exp_avg.dtype, torch.uint8)
            exp_avg_sq = state[i]['exp_avg_sq']
            self.assertEqual(type(exp_avg_sq), ScalingTensor)
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
        opt1.all_reduce_grads(model1)
        opt1.step()

        # Build model2 and update 4 times.
        model2 = copy.deepcopy(linear)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)

        opt2 = lbadam_class(model2.parameters())

        for _ in range(4):
            output = model2(input)
            opt2.zero_grad()
            output.sum().backward()
            opt2.all_reduce_grads(model2)
            opt2.step()

        # Load state dict to op2 and check if the weight is same as model1 after update weigth once.
        opt2 = lbadam_class(model2.parameters())
        opt2.load_state_dict(state_dict2)

        opt2.zero_grad()
        model2(input).sum().backward()
        opt2.all_reduce_grads(model2)
        opt2.step()

        self.assertTrue(torch.equal(model1.weight.value, model2.weight.value))

    def test_lb_adamw_base_state_dtypes(self):
        """Check the dtype of LBAdamWBase optimizer state."""
        dtypes = [torch.uint8, torch.int8, torch.float16]
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        x = torch.randn((4, 4), device='cuda', dtype=torch.float32)
        pairs = list(itertools.product(dtypes, dtypes)) + [[torch.float32, torch.float32]]
        for exp_avg_dtype, exp_avg_sq_dtype in pairs:
            with self.subTest(exp_avg_dtype=exp_avg_dtype, exp_avg_sq_dtype=exp_avg_sq_dtype):
                y = model(x)
                y.sum().backward()
                opt = LBAdamWBase(model.parameters(), exp_avg_dtype=exp_avg_dtype, exp_avg_sq_dtype=exp_avg_sq_dtype)
                opt.step()
                opt.zero_grad(set_to_none=True)

    @decorator.cuda_test
    def test_historical_window_quantization(self):
        """Test historical window quantization."""
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        opt = LBAdamW(model.parameters())
        window_size = 16
        windows = []
        for i in list(range(17)) * 6:
            x = torch.full((1, 4), i, device='cuda', dtype=torch.float32)
            windows.append(i)
            while len(windows) > window_size:
                windows.pop(0)
            y = model(x)
            self.assertTrue((model.scaling_metas['input'].amax.max() == max(windows)).all())
            y.sum().backward()
            opt.all_reduce_grads(model)
            opt.step()
