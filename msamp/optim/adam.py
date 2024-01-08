# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP adam module."""

import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.optim import LBAdamW, FSDPAdamW

class LBAdam(LBAdamW):
    """Implements Adam algorithm with weight decay fix."""
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize: bool = False,
        *args,
        **kwargs
    ):
        """Constructor. See LBAdamW class docstring for details."""
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            *args,
            **kwargs
        )
        self.use_adam = True


class DSAdam(LBAdamW):
    """An low-bits adam optimizer used as basic optimizer in DeepSpeed."""
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.,
        amsgrad=False,
        set_grad_none=True,
        **kwargs
    ):
        """Constructor. See LBAdamW class docstring for details."""
        super().__init__(
            params,
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs
        )
        self.use_adam = not adam_w_mode
        self.set_grad_none = set_grad_none



class FSDPAdam(LBAdam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            *args,
            **kwargs
        )

        self.original_params = []
        self.master_weights = []

        for group in self.param_groups:
            params = []
            for param in group['params']:
                if param is None or param.numel() == 0:
                    continue
                if hasattr(param, '_meta') and param._meta is not None:
                    self.original_params.append(param)
                    dtype = Dtypes.qtype_to_dtype[param._meta.qtype]
                    param = ScalingTensor(param.view(dtype), param._meta)
                    master_weight = param.cast(Dtypes.kfloat16)
                    master_weight.requires_grad = True
                    self.master_weights.append(master_weight)
                    params.append(master_weight)    
                else:
                    self.original_params.append(param)
                    self.master_weights.append(None)
                    params.append(param)

            group['params'] = params


    def zero_grad(self, set_to_none=False):
        for param in self.original_params:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

    def step(self):
        # cast gradient to ScalingTensor
        for i, param in enumerate(self.original_params):
            if param.grad is None:
                continue

            if hasattr(param, '_meta') and param._meta is not None:
                grad_meta = param._grad_meta
                dtype = Dtypes.qtype_to_dtype[grad_meta.qtype]
                self.master_weights[i].grad = ScalingTensor(param.grad.view(dtype), grad_meta)
                param.grad = None

        # call step() to update master weight
        super().step()

        # sync params and copy master weight to weight
        for i, param in enumerate(self.original_params):
            if hasattr(param, '_meta') and param._meta is not None and param.numel() > 0:
                data = self.master_weights[i].float().cast(param._meta.qtype, param._meta, True).value.view(torch.float32)
                param.data.copy_(data)