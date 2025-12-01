# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP Python module."""

import torch
from deepspeed.ops.adam import FusedAdam

from msamp.common.dtype import Dtypes
from msamp.nn import clip_grad_norm_
from msamp.nn import LinearReplacer
from msamp.optim import LBAdam, LBAdamW, DSAdam, FSDPAdamW
from msamp.te import TeReplacer

opt_levels = ['O1', 'O2']


def initialize(
    model, 
    optimizer=None, 
    opt_level='O1', 
    use_te=False, 
    weight_qtype=Dtypes.kfloat16,
    use_fsdp=False,
):    # noqa: C901
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
        use_te (bool): Whether to use Transformer Engine.
        weight_qtype (Dtypes): Weight quantization type.
        use_fsdp (bool): Whether to prepare the model for FSDP wrapping.
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

    if not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam, FusedAdam)):
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
    if not use_te:
        cast_model = LinearReplacer.replace(model, weight_qtype=weight_qtype)
    else:
        cast_model = TeReplacer.replace(model)
    
    if use_fsdp:
        # When using FSDP, the named parameters of the model are different now and we need to adjust the param groups
        old_named_params = {n: p for n, p in cast_model.named_parameters()}
        for _, submodule in cast_model.named_modules():
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
        # Map our new parameters to the optimizer param groups
        new_named_params = {n: p for n, p in cast_model.named_parameters()}
        mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
        for param_group in optimizer.param_groups:
            param_group["params"] = [mapping.get(p, p) for p in param_group["params"]]

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

    # Currently, we don't support foreach, capturable, differentiable, and fused.
    for k in ['foreach', 'capturable', 'differentiable', 'fused']:
        default_args.pop(k, None)

    cast_optimizer = None
    if isinstance(optimizer, torch.optim.Adam):
        cast_optimizer = LBAdam(optimizer.param_groups, **default_args)
    elif isinstance(optimizer, torch.optim.AdamW):
        cast_optimizer = LBAdamW(optimizer.param_groups, **default_args)
    elif isinstance(optimizer, FusedAdam):
        adam_w_mode = optimizer.adam_w_mode
        cast_optimizer = DSAdam(optimizer.param_groups, **default_args, adam_w_mode=adam_w_mode)

    return cast_model, cast_optimizer


__version__ = '0.4.0'
__author__ = 'Microsoft'
__all__ = ['clip_grad_norm_', 'initialize']
