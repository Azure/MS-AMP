# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP Python module."""

from msamp.nn import clip_grad_norm_

__version__ = '0.1.0'
__author__ = 'Microsoft'

__all__ = ['clip_grad_norm_', 'initialize']

import torch

from msamp.nn import LinearReplacer
from msamp.optim import LBAdam, LBAdamW

opt_levels = ['O1', 'O2']


def initialize(model, optimizer=None, opt_level='O1'):    # noqa: C901
    """Initialize your model, optimizer according to the optimization level.

    msamp.initialize() should be called after you have finished constructing your model and optimizer.
    Currently, msamp.initialize() should be called only once.

    Args:
        model (torch.nn.Module): Model to cast.
        optimizer (torch.optim.Optimizer): Optimizer to cast.
        opt_level (str): Optimization level. Currently supports 'O1', 'O2'.
            opt_level || Gemm || Communication || Weight || Weight Gradient || Optimizer States
            'O1'      || fp8  || fp8           || fp16   || fp8             || fp32 + FP32
            'O2'      || fp8  || fp8           || fp16   || fp8             || fp8 + fp16
    Return:
        return the casted model and optimizer.
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Model must be an instance of torch.nn.Module')

    if opt_level not in opt_levels:
        raise ValueError('Invalid optimization level. Please choose from {}'.format(opt_levels))

    if not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        raise ValueError('Optimizer {} is not supported in optimization level {}'.format(optimizer, opt_level))

    cast_model = LinearReplacer.replace(model)
    if not optimizer:
        # default optimizer.
        optimizer = LBAdamW(cast_model.parameters())

    default_args = optimizer.defaults

    exp_avg_dtype = torch.float32
    exp_avg_sq_dtype = torch.float32
    if opt_level == 'O2':
        exp_avg_dtype = torch.uint8
        exp_avg_sq_dtype = torch.float16

    default_args['exp_avg_dtype'] = exp_avg_dtype
    default_args['exp_avg_sq_dtype'] = exp_avg_sq_dtype

    # currently, we don't support foreach and capturable.
    if 'foreach' in default_args:
        del default_args['foreach']
    if 'capturable' in default_args:
        del default_args['capturable']

    parameters = cast_model.parameters()
    cast_optimizer = None
    if isinstance(optimizer, torch.optim.Adam):
        cast_optimizer = LBAdam(parameters, **default_args)
    elif isinstance(optimizer, torch.optim.AdamW):
        cast_optimizer = LBAdamW(parameters, **default_args)

    cast_optimizer.set_model(cast_model)

    return cast_model, cast_optimizer
