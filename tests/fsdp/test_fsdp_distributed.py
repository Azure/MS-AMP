# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.fsdp package."""

import os

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu, requires_nccl

from tests.helper import decorator
from msamp.fsdp import FsdpReplacer, FP8FullyShardedDataParallel


class FsdpDistributedTestCast(MultiProcessTestCase):
    """Test functions in distributed module with FSDP."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        super().setUp()
        torch.manual_seed(1000)

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
    def test_fp8_fsdp(self):
        """Test forward and backward functionality in FP8 FSDP."""
        rank = self.rank
        store = dist.FileStore(self.file_name, self.world_size)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)
        model = torch.nn.Sequential(torch.nn.Linear(10000, 20000), torch.nn.Dropout(), torch.nn.Linear(20000, 10000))
        model = FsdpReplacer.replace(model)
        model = FP8FullyShardedDataParallel(model, use_orig_params=True)
        for _ in range(10):
            input = torch.randn(128, 10000).cuda()
            output = model(input)
            loss = output.sum()
            loss.backward()
            for param in model.parameters():
                if param.numel() > 0:
                    assert param.grad is not None
                    param.grad.zero_()
