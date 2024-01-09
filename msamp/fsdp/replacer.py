# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP fsdp.replacer module."""

import torch

from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer


class FsdpReplacer:
    """A replacer to replace the FP8 weights with FP32 nn.Parameter and attributes."""
    @classmethod
    def replace(cls, model):
        """Replace the weights with ScalingParameter in modules."""

        model = LinearReplacer.replace(model, weight_qtype=Dtypes.kfloat8_e4m3)
        for _, submodule in model.named_modules():
            params_to_process = list(submodule.named_parameters(recurse=False))
            for param_name, param in params_to_process:
                if not isinstance(param, torch.Tensor):
                    data = param.value.view(-1)
                    padded = 0
                    if data.numel() % 4 != 0:
                        padded = 4 - data.numel() % 4
                        data = torch.nn.functional.pad(data, (0, padded))

                    data = data.view(dtype=torch.float32)
                    new_param = torch.nn.Parameter(data)
                    new_param._original_shape = param.shape
                    new_param._padded = padded
                    new_param._meta = param.meta
                    new_param._scaling_metas = param._scaling_metas

                    setattr(submodule, param_name, new_param)
        return model
