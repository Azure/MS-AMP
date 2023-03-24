# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP Python module."""

import torch

from msamp.nn import clip_grad_norm_
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

    if not optimizer:
        # default optimizer.
        optimizer = torch.optim.AdamW(model.parameters())

    if not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        raise ValueError('Optimizer {} is not supported in optimization level {}'.format(optimizer, opt_level))

    # We record the index of parameters in the original optimizer and fill new optimizer's parameter groups
    # with parameters from cast model. The index should not change.
    param_index_map = {}
    index = 0
    for p in model.parameters():
        param_index_map[id(p)] = index
        index += 1

    index_list = []
    for group in optimizer.param_groups:
        for param in group['params']:
            assert id(param) in param_index_map
            index = param_index_map[id(param)]
            index_list.append(index)

    cast_model = LinearReplacer.replace(model)
    parameters = list(cast_model.parameters())

    index = 0
    for group in optimizer.param_groups:
        params = group['params']
        for i in range(len(params)):
            params[i] = parameters[index_list[index]]
            index += 1

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

    cast_optimizer = None
    if isinstance(optimizer, torch.optim.Adam):
        cast_optimizer = LBAdam(optimizer.param_groups, **default_args)
    elif isinstance(optimizer, torch.optim.AdamW):
        cast_optimizer = LBAdamW(optimizer.param_groups, **default_args)

    return cast_model, cast_optimizer


__version__ = '0.1.0'
__author__ = 'Microsoft'
__all__ = ['clip_grad_norm_', 'initialize']
