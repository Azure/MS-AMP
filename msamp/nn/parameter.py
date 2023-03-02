# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP parameter module."""

from msamp.common.tensor import ScalingTensor


class ScalingParameter(ScalingTensor):
    """Parameter class for ScalingTensor."""
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
