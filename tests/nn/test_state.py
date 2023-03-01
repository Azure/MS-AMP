# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ModelState."""

import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingMeta
from msamp.nn import ScalingModule
from msamp.nn import model_state
from tests.helper import decorator


class FakeModel:
    """A fake model for testing."""
    def __init__(self, modules):
        """Constructor.

        Args:
            modules (list): a list of (name, module) tuples.
        """
        self.modules = modules

    def named_modules(self):
        """Return a list of (name, module) tuples."""
        for name, module in self.modules:
            yield name, module


@decorator.cuda_test
def test_model_state():
    """Test functions in ModelState."""
    module1 = ScalingModule()
    module1.scaling_metas = dict(
        input=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=16),
        wgrad=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=1),
        ograd=ScalingMeta(Dtypes.kfloat8_e5m2, window_size=16)
    )
    module2 = ScalingModule()
    module2.scaling_metas = dict(
        input=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=16),
        wgrad=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=1),
        ograd=ScalingMeta(Dtypes.kfloat8_e5m2, window_size=16)
    )

    model = FakeModel([('module1', module1), ('module2', module2)])
    model_state.register_scaling_metas(model)

    scaling_metas = model_state.flattened_scaling_metas
    assert len(scaling_metas.keys()) == 3
    assert 'input' in scaling_metas
    assert 'wgrad' in scaling_metas
    assert 'ograd' in scaling_metas

    scaling_metas['input']['qtype'] == Dtypes.kfloat8_e4m3
    scaling_metas['input']['scales'].size() == torch.Size([2])
    scaling_metas['input']['amaxs'] == torch.Size([2, 16])
    scaling_metas['input']['amax_counters'] = torch.Size([2])

    scaling_metas['wgrad']['qtype'] == Dtypes.kfloat8_e4m3
    scaling_metas['wgrad']['scales'].size() == torch.Size([2])
    scaling_metas['wgrad']['amaxs'] == torch.Size([2, 1])
    scaling_metas['wgrad']['amax_counters'] = torch.Size([2])

    scaling_metas['ograd']['qtype'] == Dtypes.kfloat8_e5m2
    scaling_metas['ograd']['scales'].size() == torch.Size([2])
    scaling_metas['ograd']['amaxs'] == torch.Size([2, 16])
    scaling_metas['ograd']['amax_counters'] = torch.Size([2])
