# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP clip_grad module."""

import torch

from msamp.common.tensor import ScalingTensor


def _compute_total_norm(parameters, norm_type=2.0):
    """Computes the total norm of the parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensor or ScalingTensor.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Return:
        Total norm of the parameters (viewed as a single vector).
    """
    grad_dtype = torch.float32
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.)

    device = grads[0].device

    def map_grads(fn, grads):
        outs = []
        for grad in grads:
            outs.append(fn(grad.to(grad_dtype)))
        return outs

    norm_type = float(norm_type)
    if norm_type == float('inf'):
        norms = list(map_grads(lambda g: g.detach().abs().max().to(device), grads))
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norm_grads = list(map_grads(lambda g: torch.norm(g.detach(), norm_type).to(device), grads))
        total_norm = torch.norm(torch.stack(norm_grads), norm_type)

    return total_norm


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a single Tensor that will have
            gradients normalized.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total norm of the gradients from
            :attr:`parameters` is ``nan``,``inf``, or ``-inf``. Default: False (will switch to True
            in the future).

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    # convert to list to avoid parameters is an iterator
    if isinstance(parameters, (torch.Tensor, ScalingTensor)):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    total_norm = _compute_total_norm(parameters, norm_type=norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`'
        )

    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)

    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        if clip_coef_clamped.item() < 1.0:
            for g in grads:
                if isinstance(g, ScalingTensor):
                    g.meta.scale /= clip_coef_clamped
                else:
                    g.mul_(clip_coef_clamped)

    return total_norm
