# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP ScalingModule class."""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from msamp.nn import ScalingParameter


class ScalingModule(nn.Module):
    """A module that supports scaling parameters."""
    def __init__(self):
        """Constructor."""
        super().__init__()
        self.scaling_metas = None

    def __setattr__(self, name, value):
        """Set attributes. Add ScalingParameter to the module's parameters.

        Args:
            name (str): Name of the attribute.
            value (Any): Value of the attribute.
        """
        if isinstance(value, ScalingParameter):
            self._parameters[name] = value
        else:
            super().__setattr__(name, value)

    def _apply(self, fn):    # noqa: C901
        """Apply a function to all parameters(including ScalingParameter) and buffers.

        We copied this function from torch.nn.Module._apply() and modified it to support ScalingParameter.

        Args:
            fn (Callable): Function to apply.

        Returns:
            Module: self.
        """
        for module in self.children():
            module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if isinstance(tensor, ScalingParameter) or torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            should_use_set_data = compute_should_use_set_data(param, param_applied)
            if should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                if should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param.grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self
