# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP parameter module."""

import torch
from msamp.common.tensor import ScalingTensor
import msamp.common.tensor.tensor as tensor_py


class ScalingParameter(ScalingTensor):
    """Parameter class for ScalingTensor."""
    @property
    def __class__(self): return torch.Tensor if tensor_py.should_pretend_to_be_tt else ScalingParameter
    def __init__(self, tensor, requires_grad=True):
        """Constructor.

        Args:
            tensor (ScalingTensor): the tensor to be wrapped as a parameter.
            requires_grad (bool): whether the parameter requires gradient.
        """
        super().__init__(tensor.value, tensor.meta)
        self._is_param = True
        self.requires_grad = requires_grad

    def __repr__(self):
        """String representation of the parameter."""
        return f'ScalingParameter({super().__repr__()}, requires_grad={self.requires_grad})'

    def clone(self):
        """Clone the parameter."""
        return ScalingParameter(super().clone(), requires_grad=self.requires_grad)
