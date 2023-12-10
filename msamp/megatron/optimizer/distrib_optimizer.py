# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""distrib_optimizer module in msamp.megatron package."""

import math

import torch
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.optimizer.optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper
from megatron.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.utils import print_rank_0

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.operators.dist_op import DistOp


class FP8DistributedOptimizer(MixedPrecisionOptimizer):
    """Distributed optimizer, for all data types (fp8, fp16, bf16, and fp32).

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        use_contiguous_buffers_in_local_ddp: if true, the local DDP model
            is using a contiguous buffer to hold the model grads.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
    """

    weight_qtype = Dtypes.kfloat8_e4m3
    master_weight_qtype = Dtypes.kfloat16
    wgrad_qtype = Dtypes.kfloat8_e4m3
    wgrad_dtype = torch.fp8e4m3

    @classmethod
    def build_model_gbuf_range(cls, model, dtype):
        """Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the DDP's grad buffer for
        each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer range.
        grad_buffer = model._grad_buffers[dtype]
        gbuf_size = grad_buffer.numel
        max_gbuf_range_size = int(math.ceil(gbuf_size / data_parallel_world_size))

        # All world ranges. (i.e., across all data parallel ranks)
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            gbuf_world_range = Range(gbuf_world_start, gbuf_world_end)
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]
        gbuf_local_range = gbuf_world_range.normalize()

        # Get each param's ranges.
        param_range_map = DistributedOptimizer.build_model_gbuf_param_range_map(model, dtype, gbuf_world_range)

        # Group into dict.
        data = {
            'local': gbuf_local_range,
            'world': gbuf_world_range,
            'world_all': gbuf_world_all_ranges,
            'param_map': param_range_map,
            'max_range_size': max_gbuf_range_size,
        }

        # MS-AMP: Add all partitions' param range map in data.
        if dtype == cls.wgrad_dtype:
            partitions = []
            for i in range(data_parallel_world_size):
                gbuf_world_range = gbuf_world_all_ranges[i]
                param_range_map = DistributedOptimizer.build_model_gbuf_param_range_map(model, dtype, gbuf_world_range)
                partitions.append(param_range_map)
                assert len(param_range_map) == model._grad_buffer_num_params[i]
            data['partitions'] = partitions

        return data

    @classmethod
    def build_model_gbuf_range_map(cls, model):
        """Create param-to-grad-buffer mappings, for grad buffer data types within a specific virtual model."""
        return {dtype: cls.build_model_gbuf_range(model, dtype) for dtype in model._grad_buffers}

    @classmethod
    def build_model_and_main_param_groups(cls, model_gbuf_ranges, param_gbuf_map, opt_group_ranges):
        """Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """
        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # MS-AMP
        model_fp8_groups = []
        shard_fp8_groups = []
        shard_hp_from_fp8_groups = []
        link_lp_params = dict()

        # Allocate (or slice) each group's param shard.
        for group_index, group_range in enumerate(opt_group_ranges):

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

            # MS-AMP: fp8 params of this group.
            model_fp8_groups_this_group = []
            shard_fp8_groups_this_group = []
            shard_hp_from_fp8_groups_this_group = []

            model_fp8_groups.append(model_fp8_groups_this_group)
            shard_fp8_groups.append(shard_fp8_groups_this_group)
            shard_hp_from_fp8_groups.append(shard_hp_from_fp8_groups_this_group)

            for model_param in group_range['params']:

                assert model_param.requires_grad

                model_index, dtype = param_gbuf_map[model_param]
                gbuf_range = model_gbuf_ranges[model_index][dtype]
                param_range = gbuf_range['param_map'][model_param]['param']

                # MS-AMP: fp8 param.
                if not torch.is_tensor(model_param):
                    # Scaling Tensor
                    assert isinstance(model_param, ScalingTensor)
                    assert model_param.numel() == param_range.end - param_range.start
                    shard_main_param = model_param.clone().cast(cls.master_weight_qtype)
                    shard_main_param.requires_grad = True
                    shard_model_param = model_param.cast_(cls.weight_qtype)

                    # link
                    link_lp_params[shard_main_param] = shard_model_param

                    assert shard_main_param.qtype == cls.master_weight_qtype
                    assert shard_model_param.qtype == cls.weight_qtype
                    model_param.data = shard_model_param.data
                    assert hasattr(model_param, 'main_grad')

                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_model_param, model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_main_param, model_param)

                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    model_fp8_groups_this_group.append(model_param)
                    shard_fp8_groups_this_group.append(shard_model_param)
                    shard_hp_from_fp8_groups_this_group.append(shard_main_param)

                # fp16, bf16 params.
                elif model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[param_range.start:param_range.end]
                    shard_main_param = shard_model_param.clone().float()
                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_model_param, model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_main_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start:param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(shard_model_param, model_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

            # Update optimizer's params.
            group_range['orig_group']['params'] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
                *shard_hp_from_fp8_groups_this_group    # MS-AMP
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
            model_fp8_groups,    # MS-AMP
            shard_fp8_groups,
            shard_hp_from_fp8_groups,
            link_lp_params,
        )

    def __init__(
        self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        fp16, bf16, params_dtype, grad_scaler, models
    ):
        """See top of class definition for argument descriptions.

        The steps in this method create the core mapping between DDP grad
        buffers, parameters, and parameter shard ranges, that is needed for
        converting between model param indexes and main parameter shard
        indexes. This method also updates the optimizer parameter groups
        with the newly created shards.
        """
        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models
        )

        # Verify that contiguous buffers are being used.
        # - Note: this should already be checked in arguments.py.
        assert use_contiguous_buffers_in_local_ddp

        # Model grad buffer ranges.
        self.model_gbuf_ranges = []
        for model_index, model in enumerate(self.models):
            self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model))
        self.model_param_gbuf_map = \
            DistributedOptimizer.build_model_param_gbuf_map(self.model_gbuf_ranges)

        # Optimizer ranges.
        self.model_param_group_index_map, self.opt_group_ranges = \
            DistributedOptimizer.build_optimizer_group_ranges(self.optimizer.param_groups,
                                                              self.model_gbuf_ranges)

        # Allocate main param shards.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
            self.model_fp8_groups,    # MS-AMP
            self.shard_fp8_groups,
            self.shard_hp_from_fp8_groups,
            self.link_lp_params,
        ) = self.build_model_and_main_param_groups(
            self.model_gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
        )

        # MS-AMP: Cast all ScalingTensor params to FP8.
        for _, model in enumerate(self.models):
            for p in model.parameters():
                if not torch.is_tensor(p):
                    p.cast_(self.weight_qtype)

        # Initialize param buffers.
        # - These are views on the DDP model's grad buffers, that share
        #   storage & have their own dtype. This is safe because the param
        #   dtype size is always <= grad dtype size.
        self.param_buffers = []
        for model_index, model in enumerate(self.models):
            current_param_buffers = {}
            for dtype, grad_buffer in model._grad_buffers.items():

                # Handle older/newer method for getting untyped storage.
                try:
                    storage = grad_buffer.data.storage()._untyped()
                except Exception:
                    storage = grad_buffer.data.storage().untyped()

                # Typed param buffer.
                param_buffer = torch.tensor(
                    storage,
                    dtype=params_dtype if dtype != self.wgrad_dtype else self.wgrad_dtype,
                    device=grad_buffer.data.device
                )
                param_buffer = param_buffer[:grad_buffer.numel_padded]
                current_param_buffers[dtype] = param_buffer
            self.param_buffers.append(current_param_buffers)

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = [g['orig_group'] for g in self.opt_group_ranges]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def get_model_param_range_map(self, param):
        """Given a model param, get the index sub-range of the param that this data-parallel rank owns."""
        model_index, dtype = self.model_param_gbuf_map[param]
        gbuf_range_map = self.model_gbuf_ranges[model_index][dtype]
        param_range_map = gbuf_range_map['param_map'][param]
        return param_range_map

    def get_model_parallel_group(self):
        """With the distributed optimizer, the model parallel group is the entire world."""
        return None

    def state_dict(self):
        """The state dict must contain the fp32-from-float16 and fp16-from-fp8 shards."""
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        # shared master weight
        state_dict['shard_fp32_from_float16_groups'] = \
            self.shard_fp32_from_float16_groups
        state_dict['shard_hp_from_fp8_groups'] = \
            self.shard_hp_from_fp8_groups
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state dict."""
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from '
                         'an old checkpoint ...')
        # convert optimizer states
        ckpt_state_dict = state_dict[optimizer_key]
        self.optimizer.load_state_dict(ckpt_state_dict)

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.fp16:
                print_rank_0('***WARNING*** found an old checkpoint, will not '
                             'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        # Copy data for the main params.
        for current_group, saved_group in zip(
            self.shard_fp32_from_float16_groups, state_dict['shard_fp32_from_float16_groups']
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)

        for current_group, saved_group in zip(self.shard_hp_from_fp8_groups, state_dict['shard_hp_from_fp8_groups']):
            for current_param, saved_param in zip(current_group, saved_group):
                if current_param.data.qtype == saved_param.data.qtype:
                    current_param.data.copy_(saved_param.data)
                else:
                    # when the data type of optimizer's master weight and checkpoint's is different
                    current_param.data.copy_(
                        saved_param.data.to(current_param.data.device).cast(current_param.data.qtype)
                    )

    def zero_grad(self, set_to_none=True):
        """Zero grads.

        We only need to zero the model related parameters, i.e.,
        model_float16_groups & model_fp32_groups. We additionally zero
        the remaining groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point.
        """
        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,    # grad empty/unused here?
            self.shard_fp32_groups,    # throws grad-access warning
            self.shard_fp32_from_float16_groups,
            self.model_fp8_groups,    # MS-AMP
            self.shard_fp8_groups,
            self.shard_hp_from_fp8_groups
        ):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)

    @staticmethod
    def get_model_buffer_dp_views(model_buffers):
        """Get shard views of each of the DDP's param/grad buffers.

        In this nested list, the top level is grouped by the virtual model
        index and the buffer's data type. The sub-level is a list of
        shards of that buffer, where each shard in the list represents
        a contiguous view of the buffer, that is owned by a data-parallel
        rank. The shard boundary does not respect parameter boundaries, and
        so the elements of some parameters are split across data parallel
        ranks.

        Additionally, return references to the entire buffers, for use
        in _reduce_scatter_base and _all_gather_base.
        """
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Buffer views.
        view_items = []
        for model_index, buffers in enumerate(model_buffers):
            for dtype, buf in buffers.items():

                assert buf.numel() % data_parallel_world_size == 0
                shard_size = int(buf.numel() / data_parallel_world_size)
                buf_views = [buf[(r * shard_size):((r + 1) * shard_size)] for r in range(data_parallel_world_size)]
                view_items.append((model_index, dtype, buf, buf_views))

        return view_items

    def get_model_grad_buffer_dp_views(self):
        """Get shard views of each of the DDP's grad buffers."""
        return self.get_model_buffer_dp_views(
            [{
                dtype: mem_buffer.data
            } for model in self.models for dtype, mem_buffer in model._grad_buffers.items()]
        )

    def get_model_param_buffer_dp_views(self):
        """Get shard views of each of the DDP's param buffers."""
        return self.get_model_buffer_dp_views(self.param_buffers)

    def reduce_model_grads(self, args, timers):    # noqa: C901
        """Reduce-scatter model grads.

        The DDP's grad buffer is used for the reduce-scatter, and thus no
        tensors are dynamically allocated.

        Note: this is a different order of reduction, versus the non-
        distributed optimizer, which reduces: 1) layernorm grads, 2) all
        grads, 3) embedding grads.
        """
        # All-reduce layer-norm grads (for sequence parallelism).
        timers('layernorm-grads-all-reduce', log_level=1).start(barrier=args.barrier_with_L1_time)
        self.allreduce_layernorm_grads(args)
        timers('layernorm-grads-all-reduce').stop()

        # All-reduce embedding grads.
        timers('embedding-grads-all-reduce', log_level=1).start(barrier=args.barrier_with_L1_time)
        self.allreduce_embedding_grads(args)
        timers('embedding-grads-all-reduce').stop()

        # Reduce-scatter setup.
        timers('grads-reduce-scatter', log_level=1).start(barrier=args.barrier_with_L1_time)
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()

        # Scale grad buffers by '1 / data_parallel_world_size'.
        for model in self.models:
            for dtype, gbuf in model._grad_buffers.items():
                # MS-AMP: Ignore FP8 for special process.
                if dtype != self.wgrad_dtype:
                    gbuf.data /= data_parallel_world_size

        # MS-AMP: For FP8, there is a scaling factor for each parameter. So we need to all reduce scaling factors,
        # re-quantize and then reduce-scatter.
        if hasattr(model, '_fp8_main_grad_scales'):
            with ScalingMeta.in_time_scaling_context(enabled=False):
                for model_id, model in enumerate(self.models):
                    old_fp8_main_grad_scales = model._fp8_main_grad_scales.clone()
                    torch.distributed.all_reduce(
                        model._fp8_main_grad_scales,
                        op=torch.distributed.ReduceOp.MIN,
                        group=data_parallel_group,
                        async_op=False
                    )
                    # re-quantize FP8 wgrad
                    need_requant = (old_fp8_main_grad_scales != model._fp8_main_grad_scales).tolist()
                    for need, g in zip(need_requant, model._scaling_grads):
                        if need:
                            g.copy_(g.float().cast(g.qtype, meta=g.meta))

                    model._fp8_main_grad_scale_invs.mul_(1.0 / data_parallel_world_size)

        # Reduce-scatter all grads.
        gbuf_view_items = self.get_model_grad_buffer_dp_views()
        for index, (model_index, dtype, gbuf, gbuf_views) in enumerate(gbuf_view_items):
            # Call DistOp.enable_fp8/disable_fp8 for FP8 parameters.
            if gbuf.dtype == self.wgrad_dtype:
                DistOp.enable_fp8(self.wgrad_qtype)
            torch.distributed._reduce_scatter_base(
                gbuf_views[data_parallel_rank],
                gbuf,
                group=data_parallel_group,
            )
            if gbuf.dtype == self.wgrad_dtype:
                DistOp.disable_fp8()

        timers('grads-reduce-scatter').stop()

        if args.wgrad_auto_scaling:
            # Weight Gradient Auto Scaling
            if args.curr_iteration % args.wgrad_auto_scaling_freq == 0:
                timers('wgrad-auto-scaling', log_level=1).start(barrier=args.barrier_with_L1_time)

                # update pre_scale in this partition
                for model_group in self.model_fp8_groups:
                    for p in model_group:
                        g = p.main_grad
                        if g is not None and not torch.is_tensor(g):
                            if g.qtype != Dtypes.kfloat8_e4m3:
                                raise TypeError('g.qtype != Dtypes.kfloat8_e4m3: {}'.format(g.qtype))
                            # stat overflow ratio
                            num_infs = torch.count_nonzero((g.value & 0x7f) == 126)
                            overflow_ratio = num_infs / g.numel()
                            if overflow_ratio > args.wgrad_auto_scaling_ratio:
                                g.meta.pre_scale.div_(2.0)
                            else:
                                g.meta.pre_scale.mul_(2.0**(1.0 / args.wgrad_auto_scaling_window))

                # synchonize pre_scale in all partitions
                for model_id, model in enumerate(self.models):
                    # all fp8 gradients
                    partitions = self.model_gbuf_ranges[model_id][torch.uint8]['partitions']
                    fp8_grads = [[p.main_grad for p in part.keys()] for part in partitions]
                    # pre_scales in the partition `data_parallel_rank`
                    pre_scales = [g.meta.pre_scale for g in fp8_grads[data_parallel_rank]]
                    max_elems_per_rank = max(model._grad_buffer_num_params)
                    pre_scales = torch.cat(pre_scales)
                    # padding to max_elems_per_rank
                    pad = max_elems_per_rank - pre_scales.numel()
                    pre_scales = F.pad(pre_scales, (0, pad))
                    output_pre_scales = pre_scales.new_empty((data_parallel_world_size, max_elems_per_rank))
                    torch.distributed._all_gather_base(output_pre_scales, pre_scales, group=data_parallel_group)
                    # assign pre_scale to all fp8 gradients
                    for grads, pre_scales in zip(fp8_grads, output_pre_scales):
                        for g, pre_scale in zip(grads, pre_scales):
                            g.meta.pre_scale.copy_(pre_scale)

                timers('wgrad-auto-scaling').stop()

    def gather_model_params(self, args, timers):    # noqa: C901
        """All-gather updated model params.

        The DDP's param buffer is used for the all-gather, and thus no
        tensors are dynamically allocated. After the all-gather, the params
        can be copied from the param buffer to the param.
        """
        timers('params-all-gather', log_level=1).start(barrier=args.barrier_with_L1_time)

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # All-gather updated main params.
        # - All param buffer views are guaranteed to have the same num elements
        #   across all data parallel ranks, due to grad buffer padding that is
        #   done in distributed.py, and extended to the param buffers. Thus,
        #   all sub-views will have consistent start/end indexes across data
        #   parallel ranks.
        pbuf_view_items = self.get_model_param_buffer_dp_views()
        for index, (model_index, dtype, pbuf, pbuf_views) in enumerate(pbuf_view_items):
            torch.distributed._all_gather_base(
                pbuf,
                pbuf_views[data_parallel_rank],
                group=data_parallel_group,
            )

        # Copy from param buffer to each param.
        for model_id, model in enumerate(self.models):
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                for param, (buf_start, buf_end) in param_map.items():
                    param_buf = self.param_buffers[model_id][dtype]
                    param_buf_shard = param_buf[buf_start:buf_end]
                    # MS-AMP: Use value property for FP8 parameter.
                    if dtype != self.wgrad_dtype:
                        param.view(-1).detach().copy_(param_buf_shard)
                    else:
                        assert param.numel() == param_buf_shard.numel(), \
                                (param.shape, param_buf_shard.shape, dtype, buf_start, buf_end, param_buf.shape)
                        param.value.view(-1).detach().copy_(param_buf_shard)

        # MS-AMP: All-gather scale_invs after all-gather parameters.
        for model_id, model in enumerate(self.models):
            scale_invs = []
            fp8_params = []
            if self.wgrad_dtype not in self.model_gbuf_ranges[model_id]:
                continue
            partitions = self.model_gbuf_ranges[model_id][self.wgrad_dtype]['partitions']
            for part in partitions:
                for p in part.keys():
                    fp8_params.append(p)
            # The scale_inv of weight has been copied to lp.
            scale_invs = [p.meta.scale_inv for p in partitions[data_parallel_rank].keys()]
            max_elems_per_rank = max(model._grad_buffer_num_params)
            if len(scale_invs) > 0:
                scale_invs = torch.stack(scale_invs)
            else:
                scale_invs = torch.tensor([], dtype=torch.float32, device='cuda')
            pad = max_elems_per_rank - scale_invs.numel()
            scale_invs = F.pad(scale_invs, (0, pad))
            output_scale_invs = scale_invs.new_empty((max_elems_per_rank * data_parallel_world_size, ))
            torch.distributed._all_gather_base(output_scale_invs, scale_invs, group=data_parallel_group)
            j = 0
            for i in range(data_parallel_world_size):
                start = i * max_elems_per_rank
                end = start + model._grad_buffer_num_params[i]
                for k in range(start, end):
                    meta = fp8_params[j].meta
                    scale_inv = output_scale_invs[k]
                    meta.scale_inv.copy_(scale_inv)
                    meta.scale.copy_(torch.reciprocal(scale_inv))
                    j += 1

        timers('params-all-gather').stop()

    def _collect_main_grad_data_for_unscaling(self):
        """Collect main grad data for unscaling.

        Note: this should be equivalent to the float-16 optimizer's method,
        but writtent differently, so the two should be combined.
        """
        return [param.grad.data for group in self.optimizer.param_groups for param in group['params']]

    def _get_model_and_main_params_data_float16(self):
        """Get aligned list of model and main params."""
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.shard_float16_groups, self.shard_fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        """Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map['param']
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start:param_range.end]
                    shard_main_param.grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)

        # MS-AMP: Copy FP8 param's grad to master_weight.grad
        def fp8_copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map['param']
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    assert param_range.size == model_grad.numel()
                    shard_main_param.grad = model_grad

        fp8_copy_group_grads(self.model_fp8_groups, self.shard_hp_from_fp8_groups)

    def _copy_main_params_to_model_params(self):
        """Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    world_range = param_range_map['gbuf_world']

                    assert world_range.size == shard_main_param.nelement()

                    model_id, dtype = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.param_buffers[model_id][dtype]

                    shard_model_param = model_param_buffer.view(-1)[world_range.start:world_range.end]

                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)

        # MS-AMP: Copy master weight's params to model FP8 params.
        def fp8_copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    world_range = param_range_map['gbuf_world']

                    assert world_range.size == shard_main_param.nelement()
                    assert world_range.size == model_param.numel()

                    model_id, dtype = self.model_param_gbuf_map[model_param]
                    assert dtype == torch.uint8
                    model_param_buffer = self.param_buffers[model_id][dtype]

                    shard_model_param = model_param_buffer.view(-1)[world_range.start:world_range.end]

                    shard_lp = self.link_lp_params[shard_main_param]
                    lp = shard_main_param.cast(self.weight_qtype)
                    assert shard_model_param.numel() == lp.numel(), \
                           (shard_model_param.numel(), lp.numel(), shard_main_param.shape)
                    shard_model_param.data.copy_(lp.value.view(-1))
                    # copy scale_inv to lp (weight)
                    # note: scale_invs are different in master weight and weight
                    shard_lp.meta.scale_inv.copy_(lp.meta.scale_inv)

        fp8_copy_group_params(self.shard_hp_from_fp8_groups, self.model_fp8_groups)

    def _copy_model_params_to_main_params(self):
        """Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map['param']
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start:param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)
