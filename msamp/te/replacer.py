# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP te.replacer module."""

import torch

from msamp.common.dtype import Dtypes
from msamp.nn import ScalingParameter
from msamp.te.modules import MSAMPLinear, MSAMPLayerNormLinear, MSAMPLayerNormMLP


class TeReplacer:
    """A replacer to replace the weights with ScalingParameter in transformer engine modules."""
    module_weight_names = {
        MSAMPLinear: ['weight'],
        MSAMPLayerNormLinear: ['weight'],
        MSAMPLayerNormMLP: ['fc1_weight', 'fc2_weight'],
    }

    @classmethod
    def _replace(cls, model):
        for mod in TeReplacer.module_weight_names:
            if isinstance(model, mod):
                mod.is_msamp_module = True
                weight_names = TeReplacer.module_weight_names[mod]
                for wname in weight_names:
                    if not hasattr(model, wname):
                        continue
                    # assert hasattr(model, wname), [k for k, v in model.named_parameters()]
                    weight = getattr(model, wname)
                    requires_grad = weight.requires_grad
                    sp = ScalingParameter(weight.data.cast(Dtypes.kfloat16), requires_grad=requires_grad)
                    # release the old weight
                    weight.data = torch.tensor([])
                    setattr(model, wname, sp)
        for child_name, child in list(model.named_children()):
            setattr(model, child_name, cls._replace(child))
        return model

    @classmethod
    def replace(cls, model):
        """Replace the weights with ScalingParameter in transformer engine modules."""
        model = cls._replace(model)
        fp8_named_weights = [(k, p) for k, p in model.named_parameters() if isinstance(p, ScalingParameter)]
        fp8_names = [k for k, _ in fp8_named_weights]
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, fp8_names)
        # empty cache
        torch.cuda.empty_cache()
        return model
