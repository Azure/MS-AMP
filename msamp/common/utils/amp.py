# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Automatic Mixed Precision (AMP) module."""

import torch


torch_amp_foreach_non_finite_check_and_unscale_ = torch._amp_foreach_non_finite_check_and_unscale_


@torch.no_grad()
def _amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale):
    '''
    This function is a wrapper around torch._foreach_non_finite_check_and_unscale_ that
    checks if a non-finite value exists in the grads. Meanwhile, all gradients are multiplied by inv_scale.
    (grad *= inv_scale).

    Args:
        grads (list): list of grads
        found_inf (Tensor): Tensor that contains a single element that is set to 1 if a non-finite is found
        inv_scale (Tensor): Tensor that contains a single element that is set to 1 / scale

    Returns:
        None
    '''
    from msamp.common.tensor import ScalingTensor
    cpu_torch_grads = []
    cuda_torch_grads = []
    scaling_grads = []
    for g in grads:
        if isinstance(g, ScalingTensor):
            scaling_grads.append(g)
        elif g.is_cuda:
            cuda_torch_grads.append(g)
        else:
            cpu_torch_grads.append(g)

    # torch.Tensor on GPU
    if len(cuda_torch_grads) > 0:
        torch_amp_foreach_non_finite_check_and_unscale_(
            cuda_torch_grads, found_inf, inv_scale)

    # torch.Tensor on CPU
    if len(cpu_torch_grads) > 0:
        for grad in cpu_torch_grads:
            grad *= inv_scale
            if not torch.isfinite(grad).all():
                found_inf.fill_(True)

    # ScalingTensor
    if len(scaling_grads) > 0:
        for grad in scaling_grads:
            grad.mul_(inv_scale)
            if not torch.isfinite(grad.meta.amax[0]):
                found_inf.fill_(True)


# Monkey patch torch._foreach_non_finite_check_and_unscale_ with our own function
torch._amp_foreach_non_finite_check_and_unscale_ = _amp_foreach_non_finite_check_and_unscale_
