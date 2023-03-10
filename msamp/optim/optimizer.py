# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP optimizer module in which defines low-bits optimizer base class."""

from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain
import warnings

import torch
from torch.optim.optimizer import Optimizer, required

from msamp.common.dtype import Floating
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.common.tensor import TensorDist
from msamp.nn import model_state


class LBOptimizer(Optimizer):
    """Low-bit optimizer base class.

    This class is adapted from https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py.
    """
    def __init__(self, params, defaults):
        """Constructor.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            defaults: dict containing default values of optimization hyperparameters.
        """
        super().__init__(params, defaults)
        self.set_grad_none = False
        self.model = None

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        rtn = self.lb_step(closure)
        self._update_scaling_factors()
        self._all_reduce_grads()
        return rtn

    def set_model(self, model):
        """Set model to optimizer.

        Args:
            model: model to be set.
        """
        if model is None:
            return
        while hasattr(model, 'module'):
            model = model.module
        self.model = model

    def _all_reduce_grads(self):
        """All-reduce gradients of parameters."""
        if self.model is None:
            return
        get_fp8_wgrads_fn = getattr(self.model, 'get_fp8_wgrads', None)
        if get_fp8_wgrads_fn is not None:
            wgrads = get_fp8_wgrads_fn()
            TensorDist.all_reduce_avg(wgrads)

    def lb_step(self, closure=None):
        """Performs a single optimization step. The subclass needs to implement this method.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        raise NotImplementedError('lb_step is not implemented')

    def _update_scaling_factors(self, training=True):
        """Update scaling factors of parameters.

        Args:
            training (bool, optional): whether in training mode, defaults to True.
        """
        if not model_state.ready_to_scale_tensor:
            return
        model_state.ready_to_scale_tensor = False

        metas = model_state.flattened_scaling_metas
        if metas is None:
            return

        meta_names = ['input']
        if training:
            meta_names.append('ograd')

        margin = 0

        for name in meta_names:
            meta = metas[name]
            qtype, scales, amaxs = meta['qtype'], meta['scales'], meta['amaxs']
            # amaxs: (n, window_size)
            amax_counters = meta['amax_counters']
            fp_max = Floating.qfp_max[qtype]
            # compute scaling factor before rolling amaxs
            sf = ScalingMeta.compute_scaling_factor(amaxs.max(1).values, scales, fp_max, margin)

            mask_valid = torch.isfinite(amaxs[:, 0])
            mask_inf_nan = ~mask_valid
            amaxs.copy_(amaxs.roll(1, dims=1))
            amax_counters += 1
            amax_counters[mask_inf_nan] = 0
            scales[mask_valid] = sf[mask_valid]

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update(
                {id(p): i
                 for i, p in enumerate(group['params'], start_index) if id(p) not in param_mappings}
            )
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, (torch.Tensor, ScalingTensor)) else k): v
            for k, v in self.state.items()
        }
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):    # noqa: C901
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError('loaded state dict has a different number of parameter groups')
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                'loaded state dict contains a parameter group '
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g['params']
                                     for g in saved_groups)), chain.from_iterable((g['params'] for g in groups))
            )
        }

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if (key != 'step'):
                    # LBOptimizer does not cast the state type

                    # if param.is_floating_point():
                    #     value = value.to(param.dtype)

                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none=None):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool, optional): instead of setting to zero, set the grads to None.
        """
        if set_to_none is None:
            set_to_none = self.set_grad_none
        super().zero_grad(set_to_none)

    def add_param_group(self, param_group):    # noqa: C901
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), 'param group must be a dict'

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, (torch.Tensor, ScalingTensor)):
                raise TypeError(
                    'optimizer can only optimize Tensors, '
                    'but one of the params is ' + torch.typename(param)
                )
            if not self.defaults.get('differentiable', None) and not (param.is_leaf or param.retains_grad):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                'optimizer contains a parameter group with duplicate parameters; '
                'in future, this will cause an error; '
                'see github.com/pytorch/pytorch/issues/40967 for more information',
                stacklevel=3
            )

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError('some parameters appear in more than one parameter group')

        self.param_groups.append(param_group)
