# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP adam module."""

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


class FSDPAdam(FSDPAdamW):
    """Implements Adam algorithm for FSDP."""
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
