# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from DeepSpeed (Copyright 2019 The Microsoft DeepSpeed Team)

"""ZeRO Optimizer with MS-AMP support."""

import torch

from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.common.dtype import Dtypes
from msamp.nn import model_state
from msamp.common.utils import TransformerEngineWrapper

from itertools import chain
from deepspeed import comm as dist
from torch._six import inf
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from packaging import version as pkg_version

from deepspeed.runtime.zero.stage_1_and_2 import all_gather_dp_groups, DeepSpeedZeroOptimizer, \
        get_accelerator, get_global_norm, is_model_parallel_parameter, \
        move_to_cpu, _get_padded_tensor, logger, \
        PIPE_REPLICATED, see_memory_usage, version, \
        BASE_OPTIMIZER_STATE, CLIP_GRAD, DS_VERSION, \
        GROUP_PADDINGS, PARAM_SLICE_MAPPINGS, PARTITION_COUNT, \
        SINGLE_PARTITION_OF_FP32_GROUPS, ZERO_STAGE, ZeroStageEnum

_original_DeepSpeedZeroOptimizer = DeepSpeedZeroOptimizer

SINGLE_PARTITION_OF_FP8_GROUPS = 'single_partition_of_fp8_groups'

MASTER_WEIGHT_QTYPE = Dtypes.kfloat16
WEIGHT_GRAD_QTYPE = Dtypes.kfloat8_e4m3
WEIGHT_QTYPE = Dtypes.kfloat8_e4m3


class FP8DeepSpeedZeroOptimizer(_original_DeepSpeedZeroOptimizer):
    """DeepSpeedZeroOptimizer with MS-AMP support.
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """
    def __init__(self, init_optimizer, *args, **kwargs):
        """The initialize function of FP8DeepSpeedZeroOptimizer."""
        # [MSAMP] We handle FP8 and FP16 separately.
        fp8_param_groups = []
        for pg in init_optimizer.param_groups:
            fp8_params = []
            hp_params = []
            for p in pg['params']:
                if not p.requires_grad:
                    continue
                if isinstance(p, ScalingTensor):
                    fp8_params.append(p)
                else:
                    hp_params.append(p)
            fp8_param_groups.append(fp8_params)
            pg['params'] = hp_params

        assert len(fp8_param_groups) == len(init_optimizer.param_groups)

        # FP32 and FP16 weights
        super().__init__(init_optimizer, *args, **kwargs)
        self.fp8_nccl_start_alignment_factor = 4

        # MSAMP variables
        self.fp8_elements_in_ipg_bucket = 0
        self.fp8_extra_large_param_to_reduce = None
        self.fp8_params_already_reduced = []
        self.fp8_grads_in_ipg_bucket = []
        self.fp8_params_in_ipg_bucket = []
        self.fp8_param_to_partition_ids = {}
        self.fp8_params_in_partition = []
        self.fp8_params_not_in_partition = []
        self.fp8_is_param_in_current_partition = {}
        self.fp8_previous_reduced_grads = None
        self.fp8_is_partition_reduced = {}
        self.fp8_is_grad_computed = {}
        self.fp8_param_dict = {}
        self.fp8_remaining_grads_in_partition = {}
        self.fp8_total_grads_in_partition = {}
        self.fp8_averaged_gradients = {}
        self.fp8_groups_flat = []
        self.fp8_param_groups = []
        self.fp8_master_param_groups = []
        self.fp8_param_id = {}
        self.fp8_parallel_partitioned_groups = []
        self.fp8_params_partitions_groups = []
        # [MSAMP] ZeRO2 for FP8
        count = 0
        for ps in fp8_param_groups:
            for p in ps:
                unique_id = id(p)
                self.fp8_param_id[unique_id] = count
                self.fp8_param_dict[count] = p
                self.fp8_params_already_reduced.append(False)
                count += 1

        partition_id = dist.get_rank(group=self.dp_process_group)
        partition_size = dist.get_world_size(group=self.dp_process_group)
        fp8_mems = [0 for _ in range(partition_size)]

        # MSAMP partition
        assert len(fp8_param_groups) == len(self.optimizer.param_groups)
        for group_id, (fp8_params, hp_pg) in enumerate(zip(fp8_param_groups, self.optimizer.param_groups)):
            group_fp8_mems = [0 for _ in range(partition_size)]
            fp8_params_with_size = [(p, (-p.numel(), i % partition_size)) for i, p in enumerate(fp8_params)]
            fp8_params_with_size.sort(key=lambda e: e[1])
            fp8_part_master_params = []

            self.fp8_param_to_partition_ids[group_id] = {}
            self.fp8_total_grads_in_partition[group_id] = {}
            for pi in range(partition_size):
                self.fp8_total_grads_in_partition[group_id][pi] = 0

            params_partitions = [list() for _ in range(partition_size)]
            # len(fp8_params_with_size) may be 0
            for p, _ in fp8_params_with_size:
                target_rank = fp8_mems.index(min(fp8_mems))
                fp8_mems[target_rank] += p.numel()
                group_fp8_mems[target_rank] += p.numel()
                param_id = self.get_fp8_param_id(p)
                self.fp8_param_to_partition_ids[group_id][param_id] = [target_rank]
                self.fp8_total_grads_in_partition[group_id][target_rank] += 1
                if partition_id == target_rank:
                    master_p = p.clone().cast(MASTER_WEIGHT_QTYPE)
                    master_p.requires_grad = True
                    master_p._link_lp_param = p
                    master_p._param_name = getattr(p, '_param_name', '')
                    p._link_hp_param = master_p
                    fp8_part_master_params.append(master_p)
                else:
                    p._link_hp_param = None
                # convert the data type of p
                p.cast_(WEIGHT_QTYPE)
                params_partitions[target_rank].append(p)

            params_in_partition = params_partitions[partition_id]
            params_not_in_partition = list(
                chain(*[part for pi, part in enumerate(params_partitions) if pi != partition_id])
            )
            self.fp8_params_partitions_groups.append(params_partitions)

            values_partitions = [[p.value for p in ps] for ps in params_partitions]
            # flat FP8 weight in values_partitions
            # sizes: fp8_mems
            max_flat_numels = max(group_fp8_mems)
            if max_flat_numels > 0:
                # check type
                ref_value = values_partitions[0][0]
                dtype = ref_value.dtype
                assert all(v.dtype == dtype for v in chain(*values_partitions)), \
                    set(chain(*[[v.dtype for v in vs] for vs in values_partitions]))
                align = self.fp8_nccl_start_alignment_factor
                max_flat_numels = (max_flat_numels + align - 1) // align * align
                # Padding for Alignment
                paddings = []
                for pi in range(partition_size):
                    pad = max_flat_numels - group_fp8_mems[pi]
                    paddings.append(pad)
                    if pad > 0:
                        values_partitions[pi].append(ref_value.new_empty((pad, )))
                logger.info(
                    f'[DeepSpeed ZeRO for MSAMP] group: {group_id}, partitions: {group_fp8_mems}, paddings: {paddings}'
                )

                # the number of elements in each partition is the same.
                values = list(chain(*values_partitions))
                move_to_cpu(values)
                # flat tensors
                # [TODO] align
                flat = _flatten_dense_tensors(values).cuda()
                for p, q in zip(values, _unflatten_dense_tensors(flat, values)):
                    assert isinstance(p, torch.Tensor)
                    p.data = q.data
                fp8_data_parallel_partitions = self.get_data_parallel_partitions(flat, group_id)
            else:
                flat = None
                fp8_data_parallel_partitions = None

            self.fp8_groups_flat.append(flat)
            self.fp8_parallel_partitioned_groups.append(fp8_data_parallel_partitions)

            self.fp8_master_param_groups.append(fp8_part_master_params)
            self.fp8_param_groups.append(fp8_params)
            self.fp8_params_in_partition.append(params_in_partition)
            self.fp8_params_not_in_partition.append(params_not_in_partition)
            # add FP8 master weight into optimizer param_groups
            hp_pg['params'].extend(fp8_part_master_params)

        assert self.fp8_param_groups == fp8_param_groups
        assert len(self.fp8_param_groups) == len(self.fp8_param_to_partition_ids)

        for param_group in self.fp8_params_in_partition:
            for param in param_group:
                self.fp8_is_param_in_current_partition[self.get_fp8_param_id(param)] = True

        for param_group in self.fp8_params_not_in_partition:
            for param in param_group:
                self.fp8_is_param_in_current_partition[self.get_fp8_param_id(param)] = False

        self.fp8_initialize_gradient_partitioning_data_structures()

        # creates backward hooks for gradient partitioning
        if self.partition_gradients or self.overlap_comm:
            self.fp8_create_reduce_and_remove_grad_hooks()

        self.fp8_reset_partition_gradient_structures()

    def _enable_universal_checkpoint(self):
        super()._enable_universal_checkpoint()
        # TODO: support MSAMP

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            self.grads_in_partition = None
            self.grads_in_partition_offset = 0

            self.fp8_ipg_buffer = None
            self.fp8_grads_in_partition = None
            self.fp8_grads_in_partition_offset = 0

    def initialize_optimizer_states(self):
        # TODO: MSAMP
        return

        for i, group in enumerate(self.bit16_groups):
            single_grad_partition = torch.zeros(
                int(self.partition_size[i]), dtype=self.single_partition_of_fp32_groups[i].dtype, device=self.device
            )
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition.pin_memory(
            ) if self.cpu_offload else single_grad_partition

        self.optimizer.step()

        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None    # class init

        return

    #########################################################################
    #################### ZeRO Stage 1 - reduce gradients ####################
    #########################################################################
    def reduce_gradients(self, pipeline_parallel=False):
        # with PP we must create ipg buffer, since backward is handled outside zero
        if pipeline_parallel and self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(
                int(self.reduce_bucket_size), dtype=self.get_acc_dtype(), device=torch.cuda.current_device()
            )
            self.ipg_buffer.append(buf_0)
            self.ipg_index = 0

            # MSAMP
            self.fp8_ipg_buffer = []
            fp8_buf_0 = torch.empty(
                int(self.reduce_bucket_size),
                dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE),
                device=torch.cuda.current_device()
            )
            self.fp8_ipg_buffer.append(fp8_buf_0)
            self.fp8_ipg_index = 0

        if not self.overlap_comm:
            for i, group in enumerate(self.bit16_groups):
                for param in group:
                    if param.grad is not None:
                        self.reduce_ready_partitions_and_remove_grads(param, i)
            for i, group in enumerate(self.fp8_param_groups):
                for param in group:
                    if param.grad is not None:
                        self.fp8_reduce_ready_partitions_and_remove_grads(param, i)
        # reduce any pending grads in either hook/non-hook case
        self.overlapping_partition_gradients_reduce_epilogue()

    #########################################################################
    ######################### ZeRO Partition Gradients########################
    #########################################################################
    def fp8_initialize_gradient_partitioning_data_structures(self):
        for i, param_group in enumerate(self.fp8_param_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            self.fp8_is_partition_reduced[i] = {}
            self.fp8_remaining_grads_in_partition[i] = {}
            self.fp8_is_grad_computed[i] = {}

            for partition_id in range(total_partitions):
                self.fp8_is_grad_computed[i][partition_id] = {}
                self.fp8_initialize_gradient_partition(i, param_group, partition_id)
                self.fp8_is_partition_reduced[i][partition_id] = False

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage('In ipg_epilogue before reduce_ipg_grads', 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage('In ipg_epilogue after reduce_ipg_grads', 0)
        self.fp8_reduce_ipg_grads()

        model_state.ready_to_all_reduce_grads = False

        # if dist.get_rank() == 0:
        #    logger.info("Params already reduced %s", self.params_already_reduced)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        for i in range(len(self.fp8_params_already_reduced)):
            self.fp8_params_already_reduced[i] = False

        if self.overlap_comm:
            torch.cuda.synchronize()
            # It is safe to clear previously reduced grads of other partitions
            self._clear_previous_reduced_grads()

        if self.cpu_offload is False:
            for i, _ in enumerate(self.bit16_groups):

                if not i in self.averaged_gradients or self.averaged_gradients[i] is None:
                    self.averaged_gradients[i] = self.get_flat_partition(
                        self.params_in_partition[i],
                        self.first_offset[i],
                        self.partition_size[i],
                        dtype=self.get_acc_dtype(),
                        device=torch.cuda.current_device(),
                        return_tensor_list=True
                    )
                else:
                    avg_new = self.get_flat_partition(
                        self.params_in_partition[i],
                        self.first_offset[i],
                        self.partition_size[i],
                        dtype=self.get_acc_dtype(),
                        device=torch.cuda.current_device(),
                        return_tensor_list=True
                    )

                    for accumulated_grad, new_avg_grad in zip(self.averaged_gradients[i], avg_new):
                        accumulated_grad.add_(new_avg_grad)

            for i, _ in enumerate(self.fp8_param_groups):
                if not i in self.fp8_averaged_gradients or self.fp8_averaged_gradients[i] is None:
                    self.fp8_averaged_gradients[i] = self.fp8_get_flat_partition(self.fp8_params_in_partition[i])
                else:
                    avg_new = self.fp8_get_flat_partition(self.fp8_params_in_partition[i])
                    for accumulated_grad, new_avg_grad in zip(self.fp8_averaged_gradients[i], avg_new):
                        accumulated_grad.data = (accumulated_grad.float() +
                                                 new_avg_grad.float()).cast(WEIGHT_GRAD_QTYPE, in_time=True)

        self._release_ipg_buffers()

        # No need to keep the gradients anymore.
        # All gradients required by the step
        # are in self.averaged_gradients
        self.zero_grad()
        see_memory_usage(f"End ipg_epilogue")

    # resets all partition to no reduced
    # sets remaining grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    def fp8_reset_partition_gradient_structures(self):
        # MSAMP
        for i, _ in enumerate(self.fp8_param_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.fp8_is_partition_reduced[i][partition_id] = False
                self.fp8_remaining_grads_in_partition[i][partition_id] = self.fp8_total_grads_in_partition[i][
                    partition_id]

                for param_id in self.fp8_is_grad_computed[i][partition_id]:
                    self.fp8_is_grad_computed[i][partition_id][param_id] = False

    def fp8_initialize_gradient_partition(self, i, param_group, partition_id):
        pass

    def fp8_create_reduce_and_remove_grad_hooks(self):
        # MSAMP hook
        for i, param_group in enumerate(self.fp8_param_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.fp8_reduce_ready_partitions_and_remove_grads(param, i)

                        hook = param.register_backward_post_hook(reduce_partition_and_remove_grads)

                    wrapper(param, i)

    def get_fp8_param_id(self, param):
        unique_id = id(param)
        return self.fp8_param_id[unique_id]

    ############### Independent Partition Gradient ########################
    def fp8_reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        if self.fp8_elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.fp8_reduce_ipg_grads()
            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.fp8_ipg_index = 1 - self.fp8_ipg_index

        param_id = self.get_fp8_param_id(param)
        assert self.fp8_params_already_reduced[param_id] == False, \
            f"The FP8 parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        if param.numel() > self.reduce_bucket_size:
            self.fp8_extra_large_param_to_reduce = param

        elif self.contiguous_gradients:
            new_grad_tensor = self.fp8_ipg_buffer[self.fp8_ipg_index
                                                  ].narrow(0, self.fp8_elements_in_ipg_bucket, param.numel())
            grad = param.grad
            if isinstance(grad, ScalingTensor):
                # only copy ScalingTensor.value
                grad = grad.value
            new_grad_tensor.copy_(grad.view(-1))
            # param: lp
            grad.data = new_grad_tensor.data.view(grad.shape)

        self.fp8_elements_in_ipg_bucket += param.numel()

        assert param.grad is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

        self.fp8_grads_in_ipg_bucket.append(param.grad)
        self.fp8_params_in_ipg_bucket.append((i, param, param_id))

    def fp8_gradient_reduction_w_predivide(self, tensor):
        raise NotImplementedError()

    def fp8_average_tensor(self, tensor):
        if self.overlap_comm:
            stream = self.reduction_stream
            stream.wait_stream(torch.cuda.current_stream())
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            if not self.reduce_scatter:
                self.fp8_gradient_reduction_w_predivide(tensor)
                return

            process_group = self.dp_process_group
            rank_and_offsets = []
            real_dp_process_group = []
            bucket_offset = 0

            partition_size = dist.get_world_size(group=process_group)
            partition_id = dist.get_rank(group=process_group)

            for i, param, param_id in self.fp8_params_in_ipg_bucket:
                partition_ids = self.fp8_param_to_partition_ids[i][param_id]
                assert all(
                    [p_id < partition_size for p_id in partition_ids]
                ), f"world size {partition_size} and p_ids: {partition_ids}"
                assert len(partition_ids) == 1
                param_partition_id = partition_ids[0]
                numel = param.numel()
                rank_and_offsets.append((param_partition_id, bucket_offset, numel))
                bucket_offset += numel
                real_dp_process_group.append(process_group)

                if param_partition_id == partition_id:
                    param.grad.div_(partition_size)

            tensor_to_reduce = tensor

            # MS-AMP does not support distributed operators with process group
            # fallback to FP16 communication
            meta = ScalingMeta(Dtypes.kfloat8_e4m3)

            def _fp8_to_fp16(tensor):
                return TransformerEngineWrapper.cast_from_fp8(
                    tensor.view(1, -1), meta.scale_inv, meta.qtype, Dtypes.kfloat16
                ).view_as(tensor)

            def _fp16_to_fp8(tensor):
                return TransformerEngineWrapper.cast_to_fp8(
                    tensor.view(1, -1), meta.scale, meta.amax[0], meta.scale_inv, meta.qtype
                ).view_as(tensor)

            async_handles = []
            grad_slice_pairs = []
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor_to_reduce.narrow(0, int(bucket_offset), int(numel))
                dst_rank = dist.get_global_rank(real_dp_process_group[i], dst)
                fp16_grad_slice = _fp8_to_fp16(grad_slice)
                async_handle = dist.reduce(fp16_grad_slice, dst=dst_rank, group=real_dp_process_group[i], async_op=True)
                async_handles.append(async_handle)
                grad_slice_pairs.append((fp16_grad_slice, grad_slice))
            for handle in async_handles:
                handle.wait()
            for fp16_grad_slice, grad_slice in grad_slice_pairs:
                grad_slice.copy_(_fp16_to_fp8(fp16_grad_slice))

    ############################################################################################

    def fp8_copy_grads_in_partition(self, param):
        assert not self.cpu_offload
        if self.fp8_grads_in_partition is None:
            self.fp8_grads_in_partition_offset = 0
            total_size = 0
            for group in self.fp8_params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()

            self.fp8_grads_in_partition = torch.empty(
                int(total_size),
                dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE),
                device=torch.cuda.current_device()
            )

        # The allreduce buffer will be rewritten. Copy the gradients in partition to a new buffer
        new_grad_tensor = self.fp8_grads_in_partition.view(-1).narrow(
            0, self.fp8_grads_in_partition_offset, param.numel()
        )
        grad = param.grad
        if isinstance(grad, ScalingTensor):
            grad = grad.value
        new_grad_tensor.copy_(grad.view(-1))
        grad.data = new_grad_tensor.data.view(grad.shape)
        self.fp8_grads_in_partition_offset += param.numel()

    def fp8_reduce_ipg_grads(self):
        if self.contiguous_gradients:
            if self.fp8_extra_large_param_to_reduce is not None:
                assert len(self.fp8_params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
                _, _, param_id = self.fp8_params_in_ipg_bucket[0]
                assert self.get_fp8_param_id(
                    self.fp8_extra_large_param_to_reduce
                ) == param_id, "param in ipg bucket does not match extra-large param"
                self.fp8_average_tensor(self.fp8_extra_large_param_to_reduce.grad.view(-1))
                self.fp8_extra_large_param_to_reduce = None
            else:
                self.fp8_average_tensor(self.fp8_ipg_buffer[self.fp8_ipg_index])
        else:
            raise NotImplementedError('no impl for MSAMP')
            self.buffered_reduce_fallback(
                None, self.fp8_grads_in_ipg_bucket, elements_per_buffer=self.fp8_elements_in_ipg_bucket
            )

        if self.overlap_comm:
            stream = self.reduction_stream
        elif self.cpu_offload:
            # TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            torch.cuda.synchronize()
            #            stream = self.copy_grad_stream
            stream = torch.cuda.current_stream()
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            for _, param, param_id in self.fp8_params_in_ipg_bucket:

                assert self.fp8_params_already_reduced[param_id] == False, \
                    f"The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported"

                self.fp8_params_already_reduced[param_id] = True

                if self.partition_gradients:
                    if not self.fp8_is_param_in_current_partition[param_id]:
                        if self.overlap_comm and self.contiguous_gradients is False:
                            # Clear grads of other partitions during the next reduction
                            # to avoid clearing them before the reduction is complete.
                            if self.fp8_previous_reduced_grads is None:
                                self.fp8_previous_reduced_grads = []
                            self.fp8_previous_reduced_grads.append(param)
                        else:
                            param.grad = None    # only if self.partition_gradients
                    elif self.contiguous_gradients:
                        self.fp8_copy_grads_in_partition(param)
                else:    # zero stage 1 - partition only optimizer state
                    if self.contiguous_gradients and self.fp8_is_param_in_current_partition[param_id]:
                        self.fp8_copy_grads_in_partition(param)

        self.fp8_grads_in_ipg_bucket = []
        self.fp8_params_in_ipg_bucket = []
        self.fp8_elements_in_ipg_bucket = 0

    def fp8_reduce_ready_partitions_and_remove_grads(self, param, i):
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            self.fp8_reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):
        super().zero_reduced_gradients(partition_id, i)

        def fp8_are_all_related_partitions_reduced(params_id):
            for partition_id in self.fp8_param_to_partition_ids[i][params_id]:
                if not self.fp8_is_partition_reduced[i][partition_id]:
                    return False

        for params_id in self.fp8_is_grad_computed[i][partition_id]:
            if fp8_are_all_related_partitions_reduced(params_id):
                self.fp8_param_dict[params_id].grad = None    # dead code

    def set_none_gradients_to_zero(self, i, partition_id):
        super().set_none_gradients_to_zero(i, partition_id)
        for param_id in self.fp8_is_grad_computed[i][partition_id]:
            param = self.fp8_param_dict[param_id]
            if param.grad is None:
                param.grad = ScalingTensor(
                    torch.zero_like(param, dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE)),
                    ScalingMeta(WEIGHT_GRAD_QTYPE)
                )

    def _clear_previous_reduced_grads(self):
        super()._clear_previous_reduced_grads()
        if self.fp8_previous_reduced_grads is not None:
            for param in self.fp8_previous_reduced_grads:
                param.grad = None    # overlap enabled
            self.fp8_previous_reduced_grads = None

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.bit16_groups + self.fp8_param_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None    # epilogue and in step
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    # CHANGE: Why does DeepSpeed use double? I have replaced it with float.
                    param_norm = g.data.float().norm(2)
                    total_norm += param_norm.item()**2
            # Sum across all model parallel GPUs.
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

            total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    def fp8_get_flat_partition(self, tensor_list):
        # return: List[ScalingTensor]
        flat_tensor_list = []
        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                tensor.grad = ScalingTensor(
                    torch.zeros_like(tensor, dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE)),
                    ScalingMeta(WEIGHT_GRAD_QTYPE)
                )
            flat_tensor_list.append(tensor.grad)
        return flat_tensor_list

    def scaled_global_norm(self, norm_type=2):
        assert norm_type == 2, "only L2 norm supported"
        norm_groups = []
        for i, group in enumerate(self.bit16_groups):
            if self.cpu_offload:
                raise NotImplementedError('no impl for MSAMP')
            else:
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))

                norm_groups.append(
                    self.get_grad_norm_direct(self.fp8_averaged_gradients[i], self.fp8_params_in_partition[i])
                )

        if self.has_moe_layers:
            self._average_expert_grad_norms(norm_groups)

        # note that the get_global_norm function only supports l2 norm
        return get_global_norm(norm_list=norm_groups)

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = -1

        see_memory_usage("In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'
        timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if dist.get_rank() == 0:
                logger.info(
                    '[deepspeed] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, '
                    'reducing to {}'.format(dist.get_rank(), prev_scale, self.loss_scale)
                )

            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad()
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}
                self.fp8_averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return

        # Step 1:- Calculate gradient norm using fp-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

        see_memory_usage('After norm before optimizer')
        # Step 2:- run optimizer and upscaling simultaneously
        assert len(self.bit16_groups) == len(self.fp8_param_groups)
        for i, group in enumerate(self.bit16_groups):
            self.start_timers([OPTIMIZER_GRADIENTS])
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                raise NotImplementedError('no impl for MSAMP')
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])
                self.free_grad_in_param_list(self.fp8_params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i], int(self.partition_size[i])
                    ).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]
                                                         ).to(self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    'averaged gradients have different number of elements that partition size {} {} {} {}'.format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])
                self.averaged_gradients[i] = None

                # MSAMP
                # assign grad to master weight
                partition_size = dist.get_world_size(group=self.dp_process_group)
                fp8_master_weight_grads = []
                assert len(self.fp8_master_param_groups[i]) == len(self.fp8_averaged_gradients[i])
                for grad, m in zip(self.fp8_averaged_gradients[i], self.fp8_master_param_groups[i]):
                    m.grad = grad
                    fp8_master_weight_grads.append(m.grad)

                self.free_grad_in_param_list(self.fp8_params_in_partition[i])
                self.fp8_averaged_gradients[i] = None

                self.unscale_and_clip_grads([single_grad_partition] + fp8_master_weight_grads, scaled_global_grad_norm)
                self.stop_timers([OPTIMIZER_GRADIENTS])

                # Step 3:- run the optimizer if no offloading
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                for m in self.fp8_master_param_groups[i]:
                    m.grad = None
                del single_grad_partition
                del fp8_master_weight_grads

                # [TODO] flat weights
                # assign master weights (hp) to weights (lp)

                ids = self.fp8_param_to_partition_ids[i]
                src_collects = [list() for _ in range(partition_size)]
                for lp in self.fp8_param_groups[i]:
                    param_id = self.get_fp8_param_id(lp)
                    partition_ids = ids[param_id]
                    assert len(partition_ids) == 1
                    src = partition_ids[0]
                    if src == partition_id:
                        hp = lp._link_hp_param
                        # DO NOT CHANGE THE POINTER OF `lp`
                        lp.copy_(hp.cast(lp.qtype))
                    src_collects[src].append(lp)

                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.stop_timers([OPTIMIZER_STEP])

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.start_timers([OPTIMIZER_ALLGATHER])
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(
            partitioned_param_groups=self.parallel_partitioned_bit16_groups,
            dp_process_group=self.real_dp_process_group,
            start_alignment_factor=self.nccl_start_alignment_factor,
            allgather_bucket_size=self.allgather_bucket_size
        )

        all_gather_dp_groups(
            partitioned_param_groups=list(filter(lambda g: g is not None, self.fp8_parallel_partitioned_groups)),
            dp_process_group=self.real_dp_process_group,
            start_alignment_factor=self.fp8_nccl_start_alignment_factor,
            allgather_bucket_size=self.allgather_bucket_size
        )

        self.all_gather_fp8_metas()

        self.stop_timers([OPTIMIZER_ALLGATHER])

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.log_timers(timer_names)
        see_memory_usage('After zero_optimizer step')

        return

    def all_gather_fp8_metas(self):
        # all gather meta.scale
        # step 1. flat all meta.scale
        scale_invs_parallel_partitioned_groups = []
        scale_invs_groups = []
        flats = []
        for i, params_partitions in enumerate(self.fp8_params_partitions_groups):
            numels = [len(ps) for ps in params_partitions]
            max_flat_numels = max(numels)
            if max_flat_numels == 0:
                continue
            partition_size = len(params_partitions)
            scale_invs_partitions = [[p.meta.scale_inv for p in ps] for ps in params_partitions]
            ref_scale = scale_invs_partitions[0][0]
            align = self.fp8_nccl_start_alignment_factor
            max_flat_numels = (max_flat_numels + align - 1) // align * align
            for pi in range(partition_size):
                pad = max_flat_numels - numels[pi]
                scale_invs_partitions[pi].append(ref_scale.new_empty((pad, )))
            scales = list(chain(*scale_invs_partitions))
            scale_invs_groups.append(scales)
            flat = _flatten_dense_tensors(scales)
            fp8_data_parallel_partitions = self.get_data_parallel_partitions(flat, i)
            scale_invs_parallel_partitioned_groups.append(fp8_data_parallel_partitions)
            flats.append(flat)

        # step 2. all gather
        all_gather_dp_groups(
            partitioned_param_groups=scale_invs_parallel_partitioned_groups,
            dp_process_group=self.real_dp_process_group,
            start_alignment_factor=self.fp8_nccl_start_alignment_factor,
            allgather_bucket_size=self.allgather_bucket_size
        )

        # step 3. assign scale
        for group_id, (scales, flat) in enumerate(zip(scale_invs_groups, flats)):
            for p, q in zip(scales, _unflatten_dense_tensors(flat, scales)):
                # [TODO] the padding elements could not need copy
                p.data.copy_(q.data)

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True

        for i in range(len(self.fp8_param_groups)):
            for j, grad in enumerate(self.fp8_averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
            overflow_gpu = get_accelerator().ByteTensor([overflow])
            '''This will capture overflow across all data parallel and expert parallel process
            Since expert parallel process are a subset of data parallel process'''
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        else:
            params = []
            for group in self.bit16_groups + self.fp8_param_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = get_accelerator().ByteTensor([overflow])

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor or ScalingTensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        if isinstance(x, ScalingTensor):
            return x.has_inf_or_nan()
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def backward(self, loss, retain_graph=False):
        """Backward function.

        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1

        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(
                int(self.reduce_bucket_size), dtype=self.get_acc_dtype(), device=torch.cuda.current_device()
            )
            self.ipg_buffer.append(buf_0)

            self.fp8_ipg_buffer = []
            fp8_buf_0 = torch.empty(
                int(self.reduce_bucket_size),
                dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE),
                device=torch.cuda.current_device()
            )
            self.fp8_ipg_buffer.append(fp8_buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(
                    int(self.reduce_bucket_size), dtype=self.get_acc_dtype(), device=torch.cuda.current_device()
                )
                self.ipg_buffer.append(buf_1)

                fp8_buf_1 = torch.empty(
                    int(self.reduce_bucket_size),
                    dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE),
                    device=torch.cuda.current_device()
                )
                self.fp8_ipg_buffer.append(fp8_buf_1)

            self.ipg_index = 0
            self.fp8_ipg_index = 0

        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def _fp8_get_groups_without_padding(self, groups_with_padding):
        groups_without_padding = []
        for i, group in enumerate(groups_with_padding):
            new_group = []
            for p in group:
                # shallow copy
                new_group.append(p.detach())
            groups_without_padding.append(new_group)
        return groups_without_padding

    def state_dict(self):
        """Get state dict.

        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = super().state_dict()
        # MSAMP
        fp8_groups_without_padding = self._fp8_get_groups_without_padding(self.fp8_master_param_groups)
        state_dict[SINGLE_PARTITION_OF_FP8_GROUPS] = fp8_groups_without_padding

        state_dict[ZERO_STAGE] = ZeroStageEnum.gradients if self.partition_gradients else ZeroStageEnum.optimizer_states
        state_dict[GROUP_PADDINGS] = self.groups_padding
        state_dict[PARTITION_COUNT] = self.partition_count

        state_dict[DS_VERSION] = version
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings

        return state_dict

    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):
        r"""Loading ZeRO checkpoint.

        Arguments:
            state_dict_list: List of all saved ZeRO checkpoints, one for each saved partition.
                Note that the number of saved partitions may differ from number of loading partitions to support
                changing GPU count, specifically DP world size, between saving and loading checkpoints.
            load_optimizer_states: Boolean indicating whether or not to load base optimizer states
            load_from_fp32_weights: Boolean indicating whether to initialize fp32 master weights from fp32
            copies in checkpoints (no precision loss) or from model's fp16 copies (with precision loss).
        """
        """
        Loads a state_dict created by an earlier call to state_dict().
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
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
        self.loss_scaler = current_rank_sd.get('loss_scaler', self.loss_scaler)
        self.dynamic_loss_scale = current_rank_sd.get('dynamic_loss_scale', self.dynamic_loss_scale)
        self.overflow = current_rank_sd.get('overflow', self.overflow)
        self.clip_grad = current_rank_sd.get(CLIP_GRAD, self.clip_grad)

        ckpt_version = current_rank_sd.get(DS_VERSION, False)
        assert ckpt_version, f'Empty ds_version in checkpoint, not clear how to proceed'
        ckpt_version = pkg_version.parse(ckpt_version)

        # zero stage 1 mode
        if not self.partition_gradients:
            required_version = pkg_version.parse('0.3.17')
            error_str = 'ZeRO stage 1 changed in {required_version} and is not backwards compatible ' \
                'with older stage 1 checkpoints. If you\'d like to load an old ZeRO-1 checkpoint ' \
                'please use an older version of DeepSpeed (<= 0.5.8) and ' \
                'set \'legacy_stage1\': true in your zero config json.'
            assert required_version <= ckpt_version, f'Old version: {ckpt_version} {error_str}'

        ckpt_is_rigid = isinstance(current_rank_sd[BASE_OPTIMIZER_STATE], dict)

        # padding is always at the last rank/partition
        # if DP=1024 and param-group elems=16 -> padding will be 1024-16 across all but one rank
        # scenario-1 (shrink): saving w. 4 gpus -> loading w. 2 gpus
        # scenario-2 (expand): saving w. 2 gpus -> loading w. 4 gpus
        # if load_optimizer_states:
        #     if new_dp_size:
        #         self.strip_padding()
        #         self.add_padding_w_new_dp_size()
        #     self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])

        if load_optimizer_states:
            if ckpt_is_rigid:
                # loading rigid ckpt into either rigid or elastic exec
                self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])
            else:
                if self.elastic_checkpoint:
                    # loading elastic into elastic exec
                    self._restore_elastic_base_optimizer_state(state_dict_list)
                else:
                    # loading an elastic checkpoint into rigid exec
                    self._restore_base_optimizer_state(current_rank_sd[BASE_OPTIMIZER_STATE])

        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 1 if changing DP degree and option 2 otherwise.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.

        if load_from_fp32_weights:
            # option 2 from above
            if self.elastic_checkpoint and not ckpt_is_rigid:
                self._restore_from_elastic_fp32_weights(state_dict_list)
            else:
                # For non-elastic checkpoint, simply copying from saved weights of current rank is sufficient.
                for current, saved in zip(
                    self.single_partition_of_fp32_groups, current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]
                ):
                    src_tensor = _get_padded_tensor(saved, current.numel())
                    current.data.copy_(src_tensor.data)
        else:
            # option 1 from above
            self._restore_from_bit16_weights()

        # MSAMP
        # [TODO] support changing DP degree
        for currents, saveds in zip(self.fp8_master_param_groups, current_rank_sd[SINGLE_PARTITION_OF_FP8_GROUPS]):
            assert len(currents) == len(saveds)
            for current, saved in zip(currents, saveds):
                current.copy_(saved)

        if load_optimizer_states:
            self._link_all_hp_params()

    def get_acc_dtype(self):
        """Get accumulation data type."""
        if self.dtype == torch.bfloat16:
            return torch.float32
        return self.dtype
