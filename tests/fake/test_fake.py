# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fake test."""

import os
import unittest

from tests.helper import decorator


class FakeTestCase(unittest.TestCase):
    """A class for fake test cases.

    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    """
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        pass

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    def test_add(self):
        """Test add."""
        self.assertEqual(1 + 2, 3)

    @decorator.cuda_test
    def test_nvidia_smi(self):
        """Test nvidia-smi."""
        smi = os.popen('nvidia-smi').read().strip()
        self.assertIn('NVIDIA-SMI', smi)
