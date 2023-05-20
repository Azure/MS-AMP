"""Copyright 2019 The Microsoft DeepSpeed Team.

Copyright NVIDIA/apex
This file is adapted from FP8_Optimizer in NVIDIA/apex.
"""

from itertools import chain
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.runtime import DeepSpeedOptimizer
from deepspeed.runtime.utils import get_global_norm, get_grad_norm
from ..utils import CheckOverflow
from deepspeed.runtime.fp16.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, MIN_LOSS_SCALE
from deepspeed.utils import groups, logger, log_dist
from deepspeed import comm as dist
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, CLIP_GRAD

import msamp
from msamp.common.tensor import ScalingTensor
from msamp.common.dtype import Dtypes

MASTER_WEIGHT_QTYPE = Dtypes.kfloat16
WEIGHT_QTYPE = Dtypes.kfloat8_e4m3
WEIGHT_GRAD_QTYPE = Dtypes.kfloat8_e4m3

COMPUTE_NORM = 'compute_norm'
OVERFLOW_CHECK = 'overflow_check'
OVERFLOW_TIMERS = [COMPUTE_NORM, OVERFLOW_CHECK]
UNSCALE_AND_CLIP = 'unscale_and_clip'
BASIC_STEP = 'basic_step'
UPDATE_FP16 = 'update_fp16'
STEP_TIMERS = OVERFLOW_TIMERS + [UNSCALE_AND_CLIP, BASIC_STEP, UPDATE_FP16]


class FP8_Optimizer(DeepSpeedOptimizer):
    """FP8 Optimizer."""
    def __init__(
        self,
        init_optimizer,
        deepspeed=None,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        initial_dynamic_scale=2**32,
        dynamic_loss_args=None,
        verbose=True,
        mpu=None,
        clip_grad=0.0,
        fused_adam_legacy=False,
        has_moe_layers=False,
        timers=None
    ):
        """This init function wraps the user optimizer."""
        self.fused_adam_legacy = fused_adam_legacy
        self.timers = timers
        self.deepspeed = deepspeed
        self.has_moe_layers = has_moe_layers
        self.using_pipeline = self.deepspeed.pipeline_parallelism if self.deepspeed is not None else None
        if not torch.cuda.is_available():
            raise SystemError('Cannot use fp16 without CUDA.')
        self.optimizer = init_optimizer

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []

        # MSAMP
        self.fp8_groups = []
        self.fp8_master_groups = []

        self._global_grad_norm = 0.

        # Divide out FP8 param groups
        # create FP8 weight and FP16 master weight
        # Note: DO NOT CHANGE THE DICT OBJECT IN PARAM_GROUPS SINCE LR_SCHEDULER WILL CHANGE THEIR LR
        for i, param_group in enumerate(self.optimizer.param_groups):
            fp8_params = []    # weight
            fp8_master_params = []    # master weight
            fp16_params = []
            for p in param_group['params']:
                if isinstance(p, ScalingTensor):
                    # FP8 Tensor
                    # high-precision param
                    master_weight = p.clone().cast(MASTER_WEIGHT_QTYPE)
                    master_weight.requires_grad = True
                    master_weight._param_name = getattr(p, '_param_name', '')
                    # low-precision param
                    p.data = p.data.cast(WEIGHT_QTYPE)
                    fp8_params.append(p)
                    fp8_master_params.append(master_weight)
                else:
                    # torch.Tensor
                    fp16_params.append(p)

            self.fp16_groups.append(fp16_params)
            # init fp16 weight buffer, flattened
            self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]]))
            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data

            # init fp32 master weight, flattened
            self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
            # modify optimizer of have flat master weight
            self.fp32_groups_flat[i].requires_grad = True    # keep this in case internal optimizer uses it

            # FP8
            self.fp8_groups.append(fp8_params)
            self.fp8_master_groups.append(fp8_master_params)

            # update param group
            param_group['params'] = [self.fp32_groups_flat[i]] + fp8_master_params

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2

            if dynamic_loss_args is None:
                self.cur_scale = initial_dynamic_scale
                self.scale_window = 1000
                self.min_loss_scale = 1
            else:
                self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
                self.scale_window = dynamic_loss_args[SCALE_WINDOW]
                self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        self.clip_grad = clip_grad
        self.norm_type = 2
        self.step_count = 0

        self.clip_grad_norm = msamp.nn.clip_grad_norm_

        # model parallel object
        self.mpu = mpu

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups + self.fp8_groups, mpu=self.mpu, deepspeed=deepspeed)
        self.initialize_optimizer_states()

    def initialize_optimizer_states(self):
        """Initialize optimizer states."""
        # optimizer.step() may change the parameters (like l2_reg).
        return

    def zero_grad(self, set_grads_to_None=True):
        """Zero FP16 parameter grads."""
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups + self.fp8_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def step_fused_adam(self, closure=None):
        """Not supporting closure."""
        raise NotImplementedError('[TODO] support ScalingTensor')

    def start_timers(self, name_list):
        """Start timers."""
        if self.timers is not None:
            for name in name_list:
                self.timers(name).start()

    def stop_timers(self, name_list):
        """Stop timers."""
        if self.timers is not None:
            for name in name_list:
                self.timers(name).stop()

    def log_timers(self, name_list):
        """Log timers."""
        if self.timers is not None:
            self.timers.log(name_list)

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def override_loss_scale(self, loss_scale):
        """Override loss scale."""
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def step(self, closure=None):
        """Step optimizer. Not supporting closure."""
        # flake8: noqa: C901
        if self.fused_adam_legacy:
            return self.step_fused_adam()

        # First determine if there is overflow.
        self.start_timers([OVERFLOW_CHECK])
        fp16_params = []
        for i, group in enumerate(self.fp16_groups):
            fp16_params.extend([p for p in group if p.grad is not None])

        fp8_params = []
        for i, group in enumerate(self.fp8_groups):
            fp8_params.extend([p for p in group if p.grad is not None])

        self.overflow = self.overflow_checker.has_overflow(fp16_params + fp8_params)
        self.stop_timers([OVERFLOW_CHECK])
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                log_dist(
                    'Overflow detected. Skipping step. Attempted loss '
                    f'scale: {prev_scale}, reducing to {self.cur_scale}',
                    ranks=[0]
                )
            # Clear gradients
            for i, group in enumerate(self.fp16_groups + self.fp8_groups):
                for p in group:
                    p.grad = None

            self.log_timers(OVERFLOW_TIMERS)
            return self.overflow

        # move weight.grad to master_weight.grad
        grads_groups_flat = []
        for i, group in enumerate(self.fp16_groups):
            data_type = self.fp32_groups_flat[i].dtype

            grads_groups_flat.append(
                _flatten_dense_tensors(
                    [
                        torch.zeros(p.size(), dtype=data_type, device=p.device)
                        if p.grad is None else p.grad.to(data_type) for p in group
                    ]
                )
            )

            for p in group:
                p.grad = None

            self.fp32_groups_flat[i].grad = grads_groups_flat[i]

        # FP8
        assert len(self.fp8_groups) == len(self.fp8_master_groups)
        for ps, master_ps in zip(self.fp8_groups, self.fp8_master_groups):
            assert len(ps) == len(master_ps)
            for p, m in zip(ps, master_ps):
                m.grad = p.grad
                p.grad = None

        self.start_timers([COMPUTE_NORM])
        fp8_groups_flat = list(chain.from_iterable(self.fp8_master_groups))

        all_groups_norm = get_grad_norm(self.fp32_groups_flat + fp8_groups_flat, mpu=self.mpu)

        self.stop_timers([COMPUTE_NORM])

        if self.has_moe_layers:
            all_groups_norm = self._get_norm_with_moe_layers(all_groups_norm)

        scaled_global_grad_norm = get_global_norm(norm_list=[all_groups_norm])

        # the gradients are not unscaled.

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_global_grad_norm / self.cur_scale

        self.start_timers([UNSCALE_AND_CLIP])

        # grad has been moved into master
        fp8_master_grads = [p.grad for p in fp8_groups_flat]
        self.unscale_and_clip_grads(grads_groups_flat + fp8_master_grads, scaled_global_grad_norm)

        self.stop_timers([UNSCALE_AND_CLIP])

        self.start_timers([BASIC_STEP])
        self.optimizer.step()
        self.stop_timers([BASIC_STEP])

        # get rid of the fp32 gradients. Not needed anymore
        for group in self.fp32_groups_flat + fp8_groups_flat:
            group.grad = None

        self.start_timers([UPDATE_FP16])

        # copy master params to params
        for i in range(len(self.fp16_groups)):
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data.copy_(q.data)

        for ps, master_ps in zip(self.fp8_groups, self.fp8_master_groups):
            for p, m in zip(ps, master_ps):
                p.data = m.data.cast(WEIGHT_QTYPE)

        self.stop_timers([UPDATE_FP16])

        self.log_timers(STEP_TIMERS)

        self.step_count += 1

        return self.overflow

    def _get_norm_with_moe_layers(self, all_groups_norm):
        """Get the norm of the gradients with moe layers."""
        # all_groups_norm_old = all_groups_norm
        # Need to allreduce (avg) the norms across
        # different ranks because moe params
        # will not be synced during allreduce
        if self.using_pipeline:
            pg = self.deepspeed.mpu.get_data_parallel_group()
        else:
            pg = groups._get_data_parallel_group()
        scaled_norm = all_groups_norm * 1.0 / float(dist.get_world_size(group=pg))
        scaled_norm_tensor = torch.tensor(scaled_norm, device=self.fp32_groups_flat[0].device, dtype=torch.float)
        dist.all_reduce(scaled_norm_tensor, group=pg)
        all_groups_norm = scaled_norm_tensor.item()
        # print(f"old = {all_groups_norm_old} and new = {all_groups_norm} at rank: {deepspeed.comm.get_rank()}")
        return all_groups_norm

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm, apply_scale=True):
        """Unscale and clip the gradients."""
        # compute combined scale factor for this group
        combined_scale = self.cur_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.cur_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale

        if apply_scale:
            for grad in grad_groups_flat:
                grad.data.mul_(1. / combined_scale)

        return combined_scale

    def backward(self, loss, create_graph=False, retain_graph=False):
        """Backward function.

        :attr:`backward` performs the following steps:
        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients
           into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            scaled_loss = (loss.float()) * self.cur_scale
            scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    def _update_scale(self, skip):
        """Update the scale factor."""
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    logger.info(f'\nGrad overflow on iteration {self.cur_iter}')
                    logger.info(f'Reducing dynamic loss scale from {prev_scale} to {self.cur_scale}')
            else:
                # Ensure self.scale_window updates since last overflow
                stable_interval = (self.cur_iter - self.last_overflow_iter) - 1
                if (stable_interval > 0) and (stable_interval % self.scale_window == 0):
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        logger.info(f'No Grad overflow for {self.scale_window} iterations')
                        logger.info(f'Increasing dynamic loss scale from {prev_scale} to {self.cur_scale}')
        else:
            if skip:
                logger.info('Grad overflow on iteration: %s', self.cur_iter)
                logger.info('Using static loss scale of: %s', self.cur_scale)
        self.cur_iter += 1
        return

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        """Get the state of the optimizer."""
        return self.optimizer.state

    def _set_state(self, value):
        """Set the state of the optimizer."""
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        """Get the param_groups of the optimizer."""
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        """Set the param_groups of the optimizer."""
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.

        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        state_dict['fp8_master_groups'] = self.fp8_master_groups
        state_dict[CLIP_GRAD] = self.clip_grad
        return state_dict

    # Refresh fp32 master params from fp16 copies
    def refresh_fp32_params(self):
        """Refresh fp32 master params from fp16 copies."""
        for current, saved in zip(self.fp32_groups_flat, self.fp16_groups_flat):
            current.data.copy_(saved.data)

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """Loads a state_dict created by an earlier call to state_dict().

        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
        self.clip_grad = state_dict[CLIP_GRAD]
        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 2.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.
        for current, saved in zip(self.fp32_groups_flat, state_dict['fp32_groups_flat']):
            current.data.copy_(saved.data)

        assert len(self.fp8_master_groups) == len(state_dict['fp8_master_groups'])
        for ps, ms in zip(self.fp8_master_groups, state_dict['fp8_master_groups']):
            assert len(ps) == len(ms)
            for p, m in zip(ps, ms):
                p.data.copy_(m.data)

    def __repr__(self):
        """Returns a string containing a pretty-printed representation of the optimizer."""
        return repr(self.optimizer)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        """Get the loss scale of the optimizer."""
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.cur_scale

    def _set_loss_scale(self, value):
        """Set the loss scale of the optimizer."""
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
