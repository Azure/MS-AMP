# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unittest decorator helpers."""

import os
import unittest

cuda_test = unittest.skipIf(os.environ.get('TEST_WITH_CUDA', '1') == '0', 'Skip CUDA tests.')
rocm_test = unittest.skipIf(os.environ.get('TEST_WITH_ROCM', '0') == '0', 'Skip ROCm tests.')
fused_attention_supported = unittest.skipIf(torch.version.cuda >= '12.1', 'Skip fused attention tests.')
