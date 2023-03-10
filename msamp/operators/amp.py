# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Automatic Mixed Precision (AMP) module."""

import torch
from msamp.common.tensor import ScalingTensor

# pylint: disable=protected-access
torch_amp_foreach_non_finite_check_and_unscale_ = torch._amp_foreach_non_finite_check_and_unscale_


@torch.no_grad()
def _amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale):
    """A wrapper of torch._amp_foreach_non_finite_check_and_unscale_ for ScalingTensor.

    This function is a wrapper around torch._foreach_non_finite_check_and_unscale_ that
    checks if a non-finite value exists in the grads.
    Meanwhile, all gradients are multiplied by inv_scale (grad *= inv_scale).

    Args:
        grads (list): list of grads
        found_inf (Tensor): a single element that is set to 1 if a non-finite is found
        inv_scale (Tensor): a single element that is set to 1 / scale

    Returns:
        None
    """
    cpu_torch_grads = []
    cuda_torch_grads = []
    scaling_grads = []
    for grad in grads:
        if isinstance(grad, ScalingTensor):
            scaling_grads.append(grad)
        elif grad.is_cuda:
            cuda_torch_grads.append(grad)
        else:
            cpu_torch_grads.append(grad)

    # torch.Tensor on GPU
    if cuda_torch_grads:
        torch_amp_foreach_non_finite_check_and_unscale_(cuda_torch_grads, found_inf, inv_scale)

    # torch.Tensor on CPU
    for grad in cpu_torch_grads:
        grad *= inv_scale
        if not torch.isfinite(grad).all():
            found_inf.fill_(True)

    # ScalingTensor
    for grad in scaling_grads:
        grad.mul_(inv_scale)
        if not torch.isfinite(grad.meta.amax[0]):
            found_inf.fill_(True)


# Monkey patch torch._foreach_non_finite_check_and_unscale_ with our own function
torch._amp_foreach_non_finite_check_and_unscale_ = _amp_foreach_non_finite_check_and_unscale_
