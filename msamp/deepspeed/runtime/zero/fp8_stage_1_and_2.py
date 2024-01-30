# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from DeepSpeed (Copyright 2019 The Microsoft DeepSpeed Team)

"""ZeRO Optimizer with MS-AMP support."""

from itertools import chain

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed import comm as dist
from deepspeed.runtime.zero.stage_1_and_2 import all_gather_dp_groups, DeepSpeedZeroOptimizer, \
    get_accelerator, logger, see_memory_usage

from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.common.dtype import Dtypes
from msamp.operators.dist_op import DistOp

SINGLE_PARTITION_OF_FP8_GROUPS = 'single_partition_of_fp8_groups'

MASTER_WEIGHT_QTYPE = Dtypes.kfloat16
WEIGHT_GRAD_QTYPE = Dtypes.kfloat8_e4m3
WEIGHT_QTYPE = Dtypes.kfloat8_e4m3


class FP8DeepSpeedZeroOptimizer(DeepSpeedZeroOptimizer):
    """DeepSpeedZeroOptimizer with MS-AMP support."""
    def __init__(self, init_optimizer, *args, **kwargs):    # noqa: C901
        """Constructor.

        Args:
            init_optimizer (torch.optim.optimizer): existing optimizer to be converted to ZeRO.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.fp8_param_groups = []
        dtype = torch.float16
        for pg in init_optimizer.param_groups:
            for p in pg['params']:
                if p.requires_grad and not isinstance(p, ScalingTensor):
                    dtype = p.dtype
                    break

        fake_param = torch.nn.parameter.Parameter(torch.zeros((), dtype=dtype))
        fake_index = 0
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
            self.fp8_param_groups.append(fp8_params)
            # DeepSpeedZeroOptimizer will crash if there is no parameters in any parameter group,
            # so add a fake parameter.
            if len(hp_params) == 0:
                param_names = args[0]
                param_names[fake_param] = 'fake_' + str(fake_index)
                fake_index += 1
                hp_params.append(fake_param)
            pg['params'] = hp_params

        assert len(self.fp8_param_groups) == len(init_optimizer.param_groups)

        super().__init__(init_optimizer, *args, **kwargs)

        self.fp8_initialize_structures()

        partition_id = dist.get_rank(group=self.dp_process_group)
        partition_size = dist.get_world_size(group=self.dp_process_group)
        fp8_mems = [0] * partition_size    # record the memory usage of each partition.

        # partition fp8 params.
        for group_id, (fp8_params, hp_pg) in enumerate(zip(self.fp8_param_groups, self.optimizer.param_groups)):
            group_fp8_mems = [0] * partition_size

            # Sort fp8 params by size and index.
            fp8_params_with_size = [(p, (-p.numel(), i % partition_size)) for i, p in enumerate(fp8_params)]
            fp8_params_with_size.sort(key=lambda e: e[1])

            self.fp8_total_grads_in_partition[group_id] = {}
            for pi in range(partition_size):
                self.fp8_total_grads_in_partition[group_id][pi] = 0

            self.fp8_param_to_partition_ids[group_id] = {}

            fp8_part_master_params = []
            params_partitions = [list() for _ in range(partition_size)]    # params in each partiton.
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
                flat = self._pad_and_flat(values_partitions, group_fp8_mems, group_id)
                fp8_data_parallel_partitions = self.get_data_parallel_partitions(flat, group_id)
            else:
                flat = None
                fp8_data_parallel_partitions = None

            self.fp8_groups_flat.append(flat)
            self.fp8_parallel_partitioned_groups.append(fp8_data_parallel_partitions)

            self.fp8_master_param_groups.append(fp8_part_master_params)
            # self.fp8_param_groups.append(fp8_params)
            self.fp8_params_in_partition.append(params_in_partition)
            self.fp8_params_not_in_partition.append(params_not_in_partition)
            # add FP8 master weight into optimizer param_groups
            hp_pg['params'].extend(fp8_part_master_params)

        self.set_in_partition_flag()
        self.fp8_initialize_gradient_partitioning_data_structures()
        # creates backward hooks for gradient partitioning
        if self.partition_gradients or self.overlap_comm:
            self.fp8_create_reduce_and_remove_grad_hooks()
        self.fp8_reset_partition_gradient_structures()

    def _pad_and_flat(self, values_partitions, group_fp8_mems, group_id):
        """Pad and flat values_partitions.

        Args:
            values_partitions (list[list[torch.Tensor]]): parameters in each partiton.
            group_fp8_mems (list[int]): fp8 memory size of each partition.
            group_id (int): group id.

        Returns:
            torch.Tensor: flat fp8 groups.
        """
        partition_size = dist.get_world_size(group=self.dp_process_group)
        ref_value = None
        for partition in values_partitions:
            if len(partition) > 0:
                ref_value = partition[0]
                break
        if ref_value is not None:
            dtype = ref_value.dtype
            assert all(v.dtype == dtype for v in chain(*values_partitions))

        align = self.fp8_nccl_start_alignment_factor
        max_flat_numels = max(group_fp8_mems)
        max_flat_numels = (max_flat_numels + align - 1) // align * align
        # Padding for Alignment
        paddings = []
        for pi in range(partition_size):
            pad = max_flat_numels - group_fp8_mems[pi]
            paddings.append(pad)
            if pad > 0:
                values_partitions[pi].append(ref_value.new_empty((pad, )))
        logger.info(f'[DeepSpeed ZeRO for MSAMP] group: {group_id}, partitions: {group_fp8_mems}, paddings: {paddings}')

        # the number of elements in each partition is the same.
        values = list(chain(*values_partitions))
        for value in values:
            value.data = value.data.cpu()
        # flat tensors
        flat = _flatten_dense_tensors(values).cuda()
        for p, q in zip(values, _unflatten_dense_tensors(flat, values)):
            assert isinstance(p, torch.Tensor)
            p.data = q.data
        return flat

    def _release_ipg_buffers(self):
        """Release the buffers used for ipg."""
        super()._release_ipg_buffers()

        if self.contiguous_gradients:
            self.fp8_ipg_buffer = None
            self.fp8_grads_in_partition = None
            self.fp8_grads_in_partition_offset = 0

    def set_in_partition_flag(self):
        """Set the flag for each param to indicate whether it is in the current partition."""
        for param_group in self.fp8_params_in_partition:
            for param in param_group:
                self.fp8_is_param_in_current_partition[self.get_fp8_param_id(param)] = True

        for param_group in self.fp8_params_not_in_partition:
            for param in param_group:
                self.fp8_is_param_in_current_partition[self.get_fp8_param_id(param)] = False

    def initialize_optimizer_states(self):
        """There is no need to initialize optimizer states for fp8."""
        return

    def reduce_gradients(self, pipeline_parallel=False):
        """Reduce the pendding gradients."""
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

    def fp8_initialize_structures(self):
        """Initialize fp8 variables."""
        # fp8 variables.
        self.fp8_nccl_start_alignment_factor = 4
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
        self.fp8_master_param_groups = []
        self.fp8_param_id = {}
        self.fp8_parallel_partitioned_groups = []
        self.fp8_params_partitions_groups = []

        # initialize self.fp8_param_id, fp8_param_dict and fp8_params_already_reduced.
        count = 0
        for pg in self.fp8_param_groups:
            for p in pg:
                unique_id = id(p)
                self.fp8_param_id[unique_id] = count
                self.fp8_param_dict[count] = p
                self.fp8_params_already_reduced.append(False)
                count += 1

    def fp8_initialize_gradient_partitioning_data_structures(self):
        """Initialize data structures for partitioning gradients."""
        for i, _ in enumerate(self.fp8_param_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            self.fp8_is_partition_reduced[i] = {}
            self.fp8_remaining_grads_in_partition[i] = {}
            self.fp8_is_grad_computed[i] = {}

            for partition_id in range(total_partitions):
                self.fp8_is_grad_computed[i][partition_id] = {}
                self.fp8_is_partition_reduced[i][partition_id] = False

    def independent_gradient_partition_epilogue(self):    # noqa: C901
        """Epilogue for igp.

        reduce the gradients in buckect, clear gradient and status. It will be called in model.backward().
        """
        self.report_ipg_memory_usage('In ipg_epilogue before reduce_ipg_grads', 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage('In ipg_epilogue after reduce_ipg_grads', 0)
        self.fp8_reduce_ipg_grads()

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
                if i not in self.averaged_gradients or self.averaged_gradients[i] is None:
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

            # accumulate fp8 gradients in partition.
            for i, _ in enumerate(self.fp8_param_groups):
                if i not in self.fp8_averaged_gradients or self.fp8_averaged_gradients[i] is None:
                    self.fp8_averaged_gradients[i] = self.fp8_get_flat_partition(self.fp8_params_in_partition[i])
                else:
                    avg_new = self.fp8_get_flat_partition(self.fp8_params_in_partition[i])
                    for accumulated_grad, new_avg_grad in zip(self.fp8_averaged_gradients[i], avg_new):
                        accumulated_grad.data = (
                            (accumulated_grad.float() +
                             new_avg_grad.float()).cast(WEIGHT_GRAD_QTYPE, meta=accumulated_grad.meta)
                        )

        self._release_ipg_buffers()

        # No need to keep the gradients anymore.
        # All gradients required by the step
        # are in self.averaged_gradients
        self.zero_grad()
        see_memory_usage('End ipg_epilogue')

    def fp8_reset_partition_gradient_structures(self):
        """Resets all partition to no reduced.

        Sets remaining grads to the total number of grads in each partition, sets is grad computed to false for
        all grads in partition
        """
        for i, _ in enumerate(self.fp8_param_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.fp8_is_partition_reduced[i][partition_id] = False
                self.fp8_remaining_grads_in_partition[i][partition_id] = self.fp8_total_grads_in_partition[i][
                    partition_id]

                for param_id in self.fp8_is_grad_computed[i][partition_id]:
                    self.fp8_is_grad_computed[i][partition_id][param_id] = False

    def fp8_create_reduce_and_remove_grad_hooks(self):
        """Create hooks for reducing gradients and removing gradients from params."""
        for i, param_group in enumerate(self.fp8_param_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.fp8_reduce_ready_partitions_and_remove_grads(param, i)

                        param.register_backward_post_hook(reduce_partition_and_remove_grads)

                    wrapper(param, i)

    def get_fp8_param_id(self, param):
        """Returns the id of the fp8 param.

        Args:
            param: The parameter to get the id for.

        Returns:
            The id of the parameter.
        """
        unique_id = id(param)
        return self.fp8_param_id[unique_id]

    def fp8_reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        """Reduces the gradients by bucket and remove gradients from params.

        Args:
            param (ScalingTensor): The parameter to reduce the gradients for.
            i (int): The index of the parameter group.
        """
        if self.fp8_elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.fp8_reduce_ipg_grads()
            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.fp8_ipg_index = 1 - self.fp8_ipg_index

        param_id = self.get_fp8_param_id(param)
        assert not self.fp8_params_already_reduced[param_id], \
            'The FP8 parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported'

        if param.numel() > self.reduce_bucket_size:
            self.fp8_extra_large_param_to_reduce = param

        elif self.contiguous_gradients:
            # Copy the grad tensor to the ipg buffer.
            new_grad_tensor = self.fp8_ipg_buffer[self.fp8_ipg_index
                                                  ].narrow(0, self.fp8_elements_in_ipg_bucket, param.numel())
            if not isinstance(param.grad, ScalingTensor):
                meta = ScalingMeta(WEIGHT_GRAD_QTYPE, group=self.dp_process_group)
                param.grad = param.grad.cast(WEIGHT_GRAD_QTYPE, meta=meta, sync=True)
            grad = param.grad.value
            new_grad_tensor.copy_(grad.view(-1))
            # param: lp
            grad.data = new_grad_tensor.data.view(grad.shape)

        self.fp8_elements_in_ipg_bucket += param.numel()

        assert param.grad is not None, f'rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient'

        self.fp8_grads_in_ipg_bucket.append(param.grad)
        self.fp8_params_in_ipg_bucket.append((i, param, param_id))

    def fp8_average_tensor(self, tensor):
        """Reduce the average value of each fp8 gradient in the bucket to the it's partition asynchronously.

        Args:
            tensor (torch.Tensor): The tensor to reduce.
        """
        if self.overlap_comm:
            stream = self.reduction_stream
            stream.wait_stream(torch.cuda.current_stream())
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            if not self.reduce_scatter:
                raise NotImplementedError('Reduce scatter is not implemented for fp8')

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
                ), f'world size {partition_size} and p_ids: {partition_ids}'
                assert len(partition_ids) == 1
                param_partition_id = partition_ids[0]
                numel = param.numel()
                rank_and_offsets.append((param_partition_id, bucket_offset, numel))
                bucket_offset += numel
                real_dp_process_group.append(process_group)

                if param_partition_id == partition_id:
                    param.grad.div_(partition_size)

            tensor_to_reduce = tensor

            # Reduce gradient.
            async_handles = []
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor_to_reduce.narrow(0, int(bucket_offset), int(numel))
                dst_rank = dist.get_global_rank(real_dp_process_group[i], dst)
                async_handle = DistOp.reduce(
                    grad_slice, WEIGHT_GRAD_QTYPE, dst=dst_rank, group=real_dp_process_group[i], async_op=True
                )
                async_handles.append(async_handle)
            for handle in async_handles:
                handle.wait()

    def fp8_copy_grads_in_partition(self, param):
        """Copy the gradient of the param to a buffer belong's to it's partition.

        Args:
            param (torch.nn.Parameter): The parameter to copy the gradient from.
        """
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

    def fp8_reduce_ipg_grads(self):    # noqa: C901
        """Reduce the gradients in the bucket.

        Reduce gradients, copy the reduced gradients to the partition buffers and remove the other gradients.
        """
        if self.contiguous_gradients:
            if self.fp8_extra_large_param_to_reduce is not None:
                assert len(self.fp8_params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
                _, _, param_id = self.fp8_params_in_ipg_bucket[0]
                assert self.get_fp8_param_id(
                    self.fp8_extra_large_param_to_reduce
                ) == param_id, 'param in ipg bucket does not match extra-large param'
                self.fp8_average_tensor(self.fp8_extra_large_param_to_reduce.grad.view(-1))
                self.fp8_extra_large_param_to_reduce = None
            else:
                self.fp8_average_tensor(self.fp8_ipg_buffer[self.fp8_ipg_index])
        else:
            raise NotImplementedError('We do not support non-contiguous gradients for fp8')

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
                assert not self.fp8_params_already_reduced[param_id], \
                    f'The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported'

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

        # clean the bucket.
        self.fp8_grads_in_ipg_bucket = []
        self.fp8_params_in_ipg_bucket = []
        self.fp8_elements_in_ipg_bucket = 0

    def fp8_reduce_ready_partitions_and_remove_grads(self, param, i):
        """Reduce gradients and remove grads in parameters.

        Args:
            param (ScalingTensor): The parameter tensor.
            i (int): The index of the parameter group.
        """
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            self.fp8_reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def _clear_previous_reduced_grads(self):
        super()._clear_previous_reduced_grads()
        if self.fp8_previous_reduced_grads is not None:
            for param in self.fp8_previous_reduced_grads:
                param.grad = None    # overlap enabled
            self.fp8_previous_reduced_grads = None

    def zero_grad(self, set_grads_to_None=True):
        """Zero FP16 and FP8 parameter grads."""
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

    def fp8_get_flat_partition(self, tensor_list):
        """Get list of gradient from a list of ScalingTensors. Using zero to initialize if grad is None.

        Args:
            tensor_list (list): list of tensors.

        Returns:
            list: list of gradients.
        """
        flat_tensor_list = []
        for _, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                tensor.grad = ScalingTensor(
                    torch.zeros_like(tensor, dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE)),
                    ScalingMeta(WEIGHT_GRAD_QTYPE)
                )
            flat_tensor_list.append(tensor.grad)
        return flat_tensor_list

    def start_timers(self, timer_names):
        """Start timers."""
        if self.timers is None:
            return

        for name in timer_names:
            self.timers(name).start()

    def stop_timers(self, timer_names):
        """Stop timers."""
        if self.timers is None:
            return

        for name in timer_names:
            self.timers(name).stop()

    def log_timers(self, timer_names):
        """Log timers."""
        if self.timers is None:
            return

        self.timers.log(names=list(timer_names))

    def step(self, closure=None):    # noqa C901
        """Performs a single optimization step. closure is not supported."""
        self.micro_step_id = -1

        see_memory_usage('In step before checking overflow')

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
                # If we are last partition, ensure we have same size grads and partition size,
                # if not pad with zero tensors
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
                # release all the gradient since we have already created a necessary copy in
                # dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])
                self.averaged_gradients[i] = None

                # assign fp8 grad to master weight
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
            groups_flat=self.bit16_groups_flat,
            partitioned_param_groups=self.parallel_partitioned_bit16_groups,
            dp_process_group=self.real_dp_process_group,
            start_alignment_factor=self.nccl_start_alignment_factor,
            allgather_bucket_size=self.allgather_bucket_size
        )

        all_gather_dp_groups(
            groups_flat=list(filter(lambda g: g is not None, self.fp8_groups_flat)),
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
        """All gather fp8 meta data."""
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
            align = self.fp8_nccl_start_alignment_factor
            max_flat_numels = (max_flat_numels + align - 1) // align * align
            for pi in range(partition_size):
                pad = max_flat_numels - numels[pi]
                scale_invs_partitions[pi].append(torch.empty((pad, ), dtype=torch.float32, device='cuda'))

            scales = list(chain(*scale_invs_partitions))
            scale_invs_groups.append(scales)
            flat = _flatten_dense_tensors(scales)
            fp8_data_parallel_partitions = self.get_data_parallel_partitions(flat, i)
            scale_invs_parallel_partitioned_groups.append(fp8_data_parallel_partitions)
            flats.append(flat)

        # step 2. all gather
        all_gather_dp_groups(
            groups_flat=flats,
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
        """Check if there is overflow in any accumulated gradient.

        Returns:
            bool: Whether or not there is overflow in any accumulated grad.
        """
        if super().has_overflow_partitioned_grads_serial():
            return True

        for i in range(len(self.fp8_param_groups)):
            for j, grad in enumerate(self.fp8_averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        """Check if there is overflow in any accumulated gradient.

        Args:
            partition_gradients (bool, optional): Whether or not to partition gradients. Defaults to True.

        Returns:
            bool: Whether or not there is overflow.
        """
        if partition_gradients:
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
            overflow_gpu = get_accelerator().ByteTensor([overflow])
            # This will capture overflow across all data parallel and expert parallel process
            # Since expert parallel process are a subset of data parallel process.
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

    @staticmethod
    def _has_inf_or_nan(x, j=None):
        """Check for inf/nan in a tensor or ScalingTensor.

        Args:
            x (torch.Tensor or ScalingTensor): tensor to check for inf/nan.
            j (int, optional): index of the tensor in the bucket. Defaults to None.

        Returns:
            bool: True if the tensor has inf/nan, False otherwise.
        """
        if isinstance(x, ScalingTensor):
            return x.has_inf_or_nan()
        return DeepSpeedZeroOptimizer._has_inf_or_nan(x, j)

    def backward(self, loss, retain_graph=False):
        """Backward function.

        :attr:`backward` performs the following steps:
        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the
           model's fp16 leaves

        Args:
            loss (torch.Tensor): loss tensor to perform backward pass on.
            retain_graph (bool, optional): If ``True``, graph is retained for another backward pass.
                                           Defaults to ``False``.
        """
        if self.contiguous_gradients:
            self.fp8_ipg_buffer = []
            fp8_buf_0 = torch.empty(
                int(self.reduce_bucket_size),
                dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE),
                device=torch.cuda.current_device()
            )
            self.fp8_ipg_buffer.append(fp8_buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                fp8_buf_1 = torch.empty(
                    int(self.reduce_bucket_size),
                    dtype=Dtypes.get_dtype_from_qtype(WEIGHT_GRAD_QTYPE),
                    device=torch.cuda.current_device()
                )
                self.fp8_ipg_buffer.append(fp8_buf_1)

            self.fp8_ipg_index = 0

        super().backward(loss.float(), retain_graph=retain_graph)

    def _fp8_get_groups(self, groups_with_padding):
        groups_without_padding = []
        for _, group in enumerate(groups_with_padding):
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
        fp8_groups = self._fp8_get_groups(self.fp8_master_param_groups)
        state_dict[SINGLE_PARTITION_OF_FP8_GROUPS] = fp8_groups
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
        super()._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)
        # [TODO] support changing DP degree
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
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
