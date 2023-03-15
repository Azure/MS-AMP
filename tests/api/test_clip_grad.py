# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for clip_grad api in MS-AMP."""

import unittest
import copy

import torch

import msamp
from msamp.nn import LinearReplacer


class ClipGradTestCast(unittest.TestCase):
    """Test functions in clip_grad module."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    def test_clip_grad_norm(self):
        """Test clip_grad_norm_ function."""
        model = torch.nn.Linear(4, 4, bias=False).to('cuda')
        model2 = copy.deepcopy(model)

        input = torch.randn(4, 4, device='cuda')
        model(input).sum().backward()

        grads = []
        sum = 0.
        for param in model.parameters():
            sum += param.grad.abs().sum().item()
            grads.append(param.grad.clone())

        msamp.clip_grad_norm_(model.parameters(), 1.0, norm_type=1.0)
        i = 0
        for param in model.parameters():
            torch.allclose(param.grad, 1.0 / sum * grads[i], atol=1e-5)
            i += 1

        model2 = LinearReplacer.replace(model2)
        model2(input).sum().backward()

        grads = []
        sum = 0.
        for param in model2.parameters():
            sum += param.grad.float().abs().sum().item()
            grads.append(param.grad.clone())

        msamp.clip_grad_norm_(model2.parameters(), 1.0, norm_type=1.0)
        i = 0
        for param in model2.parameters():
            torch.allclose(param.grad.float(), 1.0 / sum * grads[i].float(), atol=1e-5)
            i += 1
