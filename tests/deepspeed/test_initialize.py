# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for deepspeed initialize with MS-AMP."""

import unittest
import torch
import torch.nn as nn

from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine

from deepspeed.pipe import PipelineModule

from msamp import deepspeed
from msamp.deepspeed.runtime.engine import MSAMPDeepSpeedEngine

from tests.helper import decorator


class DeepSpeedInitializeTestCase(unittest.TestCase):
    """Test DeepSpeed.initialize with MS-AMP."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_initialize(self):
        """Test DeepSpeed.initialize method."""
        model1 = torch.nn.Linear(4, 4)
        config = {
            'train_batch_size': 1,
        }
        model1, _, _, _ = deepspeed.initialize(model=model1, config=config)
        assert isinstance(model1, MSAMPDeepSpeedEngine)

        model2 = nn.Sequential(nn.Linear(4, 10), nn.ReLU(inplace=True), nn.Linear(10, 4))

        model2 = PipelineModule(layers=model2, num_stages=1)
        model2, _, _, _ = deepspeed.initialize(model=model2, config=config)
        assert isinstance(model2, PipelineEngine)

        config = {'train_batch_size': 1, 'hybrid_engine': {'enabled': True}}
        model3 = torch.nn.Linear(4, 4)
        model3, _, _, _ = deepspeed.initialize(model=model3, config=config)
        assert isinstance(model3, DeepSpeedHybridEngine)
