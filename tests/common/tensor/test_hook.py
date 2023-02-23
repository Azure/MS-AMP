# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for HookManager."""

import torch

from msamp.common.tensor import HookManager

inputs = []
outputs = []


def record_input_output(_, input, output):
    """Put input and output tensor to global inputs and outpus.

    Args:
        _  (torch.nn.Module): Module to hook.
        input (torch.Tensor): Input tensor of the module.
        output (torch.Tensor): Output tensor of the module.
    """
    inputs.append(input[0])
    outputs.append(output)


def test_hook_manager():
    """Test HookManager."""
    hook_manager = HookManager()
    handle1 = hook_manager.register_hook(record_input_output)
    handle2 = hook_manager.register_hook(record_input_output)

    x = torch.randn(2, 3)
    hook_manager(x)
    assert len(inputs) == 2
    assert len(outputs) == 2
    assert inputs[0].equal(x) and inputs[1].equal(x)
    assert outputs[0].equal(x) and outputs[1].equal(x)
    handle1.remove()
    handle2.remove()

    inputs.clear()
    outputs.clear()
    hook_manager(x)
    assert len(inputs) == 0
    assert len(outputs) == 0
