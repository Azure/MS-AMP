# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for fp8_op module."""

import os

import torch
import torch.distributed as dist

from tests.helper import decorator
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu, requires_nccl

from msamp.common.dtype import Dtypes
from msamp.operators.dist_op import DistOp


class DistOpTestCase(MultiProcessTestCase):
    """A class for FP8Op test cases."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
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
    def test_all_reduce(self):
        """Test all reduce without fp8 disable."""
        rank = self.rank
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(rank)

        tensors = [torch.rand(4).cuda(), torch.rand(4).cuda()]
        target = tensors[0] + tensors[1]
        tensor = tensors[rank].clone()

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.assertEqual(tensor, target)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @decorator.cuda_test
    def test_all_reduce_fp8(self):
        """Test all reduce with fp8 enabled."""
        rank = self.rank
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(rank)

        tensors = [
            torch.tensor([0b01001010, 0b01011000], dtype=torch.uint8, device='cuda'),
            torch.tensor([0b01010000, 0b01011100], dtype=torch.uint8, device='cuda')
        ]
        target = torch.tensor([0b01010101, 0b01100010], dtype=torch.uint8, device='cuda')

        tensor = tensors[rank].clone()
        DistOp.enable_fp8(Dtypes.kfloat8_e4m3)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.assertEqual(tensor, target)
        DistOp.disable_fp8()
