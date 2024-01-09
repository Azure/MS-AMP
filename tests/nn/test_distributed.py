# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.nn.distributed module."""

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu, requires_nccl

from msamp.nn import LinearReplacer
from msamp.nn.distributed import FP8DistributedDataParallel
from tests.helper import decorator


class FakeNet(nn.Module):
    """A fake network for testing."""
    def __init__(self):
        """Constructor."""
        super().__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        """Forward function."""
        return self.fc1(x)


class DistributedTestCase(MultiProcessTestCase):
    """Test functions in distributed module."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        """Return the number of processes."""
        return 2

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @decorator.cuda_test
    def test_fp8_ddp(self):
        """Test FP8DistributedDataParallel."""
        rank = self.rank
        store = dist.FileStore(self.file_name, self.world_size)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)

        fake_model = FakeNet().cuda()
        model = LinearReplacer.replace(fake_model)
        try:
            # ddp_with_replicated_tensor is set in MultiProcessTestCase and should disabled. We catch exception because
            # replicated_tensor_ddp_utils is not available in torch 2.
            from torch.nn.parallel._replicated_tensor_ddp_utils import _set_ddp_with_replicated_tensor
            _set_ddp_with_replicated_tensor(False)
        except Exception:
            pass

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        assert isinstance(model, FP8DistributedDataParallel)
        # input is different for each rank.
        input = torch.randn(20, 10).cuda() + rank
        output = model(input)
        output.sum().backward()
        assert fake_model.fc1.weight.grad.value.dtype == torch.uint8
        # after backward, the grad should be same for each rank. And we call all_reduce to check it.
        expect_grad = self.world_size * fake_model.fc1.weight.grad.value.clone()
        dist.all_reduce(fake_model.fc1.weight.grad.value)
        assert torch.equal(expect_grad, fake_model.fc1.weight.grad.value)
