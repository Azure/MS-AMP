# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for msamp.te.replacer module."""

import os
import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu, requires_nccl
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from tests.helper import decorator
from msamp import deepspeed
from msamp.nn import ScalingParameter
from msamp.optim import LBAdamW
from msamp.te.replacer import TeReplacer


class TeReplacerTestCase(unittest.TestCase):
    """Test TeExtention overrider."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)
        self.hidden_size = 4096
        self.ffn_hidden_size = 16384
        self.num_attention_heads = 32
        self.dtype = torch.float16
        self.batch_size = 4
        self.sequence_length = 128

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_replace(self):
        """Test replace function in TeReplacer."""
        te_transformer = te.TransformerLayer(
            self.hidden_size, self.ffn_hidden_size, self.num_attention_heads, fuse_qkv_params=True
        )
        te_transformer.to(dtype=self.dtype).cuda()

        model = TeReplacer.replace(te_transformer)
        msamp_module_cnt = 0

        def _check_model(model):
            if type(model) in TeReplacer.module_weight_names:
                nonlocal msamp_module_cnt
                msamp_module_cnt += 1
                weights = TeReplacer.module_weight_names[type(model)]
                for weight in weights:
                    if not hasattr(model, weight):
                        continue
                    weight = getattr(model, weight)
                    assert isinstance(weight, ScalingParameter)

            for _, child in list(model.named_children()):
                _check_model(child)

        _check_model(model)
        assert msamp_module_cnt == 3

        scaling_params = [p for p in model.parameters() if isinstance(p, ScalingParameter)]
        assert len(scaling_params) == 4
        is_fp8_available, _ = te.fp8.is_fp8_available()
        if is_fp8_available:
            # Do a forward pass to make sure the model is working.
            fp8_format = Format.HYBRID
            fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
            x = torch.rand(self.sequence_length, self.batch_size, self.hidden_size).cuda().to(dtype=self.dtype)

            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                y = model(x, attention_mask=None)
                assert y.shape == (self.sequence_length, self.batch_size, self.hidden_size)
            y.sum().backward()

    @decorator.cuda_test
    def test_te_with_deepspeed(self):
        """Test TransformerEngine + MS-AMP with DeepSpeed."""
        te_transformer = te.TransformerLayer(
            self.hidden_size, self.ffn_hidden_size, self.num_attention_heads, fuse_qkv_params=True
        )
        te_transformer.to(dtype=self.dtype).cuda()

        model = TeReplacer.replace(te_transformer)

        ds_config = {
            'train_batch_size': self.batch_size,
            'train_micro_batch_size_per_gpu': self.batch_size,
            'zero_optimization': {
                'stage': 2,
            }
        }

        optimizer = LBAdamW(model.parameters(), lr=1e-3, weight_decay=0)
        model, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)

        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            input = torch.randn(self.sequence_length, self.batch_size, self.hidden_size).cuda().to(dtype=self.dtype)
            output = model(input, attention_mask=None)
            loss = output.sum()
            model.backward(loss)
            model.step()


class TeReplacerDistributedTestCast(MultiProcessTestCase):
    """Test functions in distributed module with TransformerEngine."""
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
    def test_fp8_ddp_with_te(self):
        """Test FP8DistributedDataParallel with TransformerEngine."""
        hidden_size = 4096
        ffn_hidden_size = 16384
        num_attention_heads = 32
        dtype = torch.float16
        batch_size = 4
        sequence_length = 128

        rank = self.rank
        store = dist.FileStore(self.file_name, self.world_size)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', store=store, rank=self.rank, world_size=self.world_size)

        te_transformer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, fuse_qkv_params=True)
        te_transformer.to(dtype=dtype).cuda()
        model = TeReplacer.replace(te_transformer)
        try:
            # ddp_with_replicated_tensor is set in MultiProcessTestCase and should disabled. We catch exception because
            # replicated_tensor_ddp_utils is not available in torch 2.
            from torch.nn.parallel._replicated_tensor_ddp_utils import _set_ddp_with_replicated_tensor
            _set_ddp_with_replicated_tensor(False)
        except Exception:
            pass

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        # input is different for each rank.
        x = torch.randn(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            output = model(x, attention_mask=None)
            output.sum().backward()
