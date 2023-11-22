# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The DistributedDataParallel which supports FP8."""

import math

import torch
from megatron.core import mpu
from megatron.model.distributed import MemoryBuffer, DistributedDataParallelBase

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingMeta, ScalingTensor
from msamp.operators.arithmetic import Arithmetic


class FP8DistributedDataParallel(DistributedDataParallelBase):
    """A DDP with contiguous buffers and FP8 spport."""
    wgrad_qtype = Dtypes.kfloat8_e4m3
    wgrad_dtype = torch.fp8e4m3

    def __init__(    # noqa: C901
        self, module, accumulate_allreduce_grads_in_fp32, use_contiguous_buffers
    ):
        """DDP with contiguous buffers options to store and accumulate gradients.

        This class:
            - has the potential to reduce memory fragmentation.
            - provides the option to do the gradient accumulation
            in a type other than the params type (for example fp32)

        Arguments:
            module: input model.
            accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
                and the gradient all-reduce all in in float32. If this option is
                true, we require `use_contiguous_buffers` to be true too.
            use_contiguous_buffers: if true, use a contiguous buffer to store the
                gradients.
        """
        super(FP8DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continuous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffer_param_index_map = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._grad_buffer_param_index_map = {}
            data_parallel_world_size = mpu.get_data_parallel_world_size()

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            fp8_params = []
            for param in self.module.parameters():
                if param.requires_grad:
                    if torch.is_tensor(param):
                        dtype = _get_buffer_type(param)
                        type_num_elements[dtype] = type_num_elements.get(dtype, 0) + param.data.nelement()
                    else:
                        fp8_params.append(param)

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():

                # If using distributed optimizer, pad memory buffer to be
                # multiple of data_parallel_world_size. (This padding is done
                # due to a constraint with the reduce_scatter op, which requires
                # all tensors have equal size. See: optimizer.py.)
                num_elements_padded = data_parallel_world_size * \
                    int(math.ceil(num_elements / data_parallel_world_size))

                # Allocate grad buffer.
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, num_elements_padded, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Skip ScalingTensor.
                    if not torch.is_tensor(param):
                        continue
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(param.data.shape, type_num_elements[dtype])
                    if dtype not in self._grad_buffer_param_index_map:
                        self._grad_buffer_param_index_map[dtype] = {}
                    self._grad_buffer_param_index_map[dtype][param] = (
                        type_num_elements[dtype],
                        type_num_elements[dtype] + param.data.nelement(),
                    )

            # Create MemoryBuffer for FP8.
            self._grad_buffer_num_params = [0 for _ in range(data_parallel_world_size)]
            if len(fp8_params) > 0:
                self._grad_buffer_param_index_map[self.wgrad_dtype] = {}
                # Sort fp8 params by size and assign to the shard with latest parameters sequencially.
                fp8_params_with_size = [
                    (p, (-p.numel(), i % data_parallel_world_size)) for i, p in enumerate(fp8_params)
                ]
                fp8_params_with_size.sort(key=lambda e: e[1])
                fp8_mems = [0 for _ in range(data_parallel_world_size)]
                fp8_partitions = [[] for _ in range(data_parallel_world_size)]
                for p, _ in fp8_params_with_size:
                    target_rank = fp8_mems.index(min(fp8_mems))
                    fp8_mems[target_rank] += p.numel()
                    fp8_partitions[target_rank].append(p)
                    self._grad_buffer_num_params[target_rank] += 1
                max_fp8_mems = max(fp8_mems)
                num_elements = max_fp8_mems * data_parallel_world_size
                assert self.wgrad_dtype not in self._grad_buffers

                # get dtype
                dtype = self.wgrad_dtype
                self._grad_buffers[self.wgrad_dtype] = MemoryBuffer(num_elements, num_elements, dtype)
                num_params = len(fp8_params)
                window_size = 1
                scales = torch.ones((num_params, ), device='cuda')
                scale_invs = torch.ones((num_params, ), device='cuda')
                amaxs = torch.zeros((num_params, window_size), device='cuda')
                scaling_grads = []
                t = 0
                pre_scale = 1.0 / math.sqrt(data_parallel_world_size)

                # Create main_grad(ScalingTensor) for each param.
                for pi in range(data_parallel_world_size):
                    start = pi * max_fp8_mems
                    for p in fp8_partitions[pi]:
                        meta = ScalingMeta(self.wgrad_qtype, scale=scales[t], scale_inv=scale_invs[t], amax=amaxs[t])
                        meta.pre_scale = pre_scale
                        t += 1
                        p.main_grad = ScalingTensor(self._grad_buffers[self.wgrad_dtype].get(p.shape, start), meta)
                        self._grad_buffer_param_index_map[self.wgrad_dtype][p] = (start, start + p.numel())
                        start += p.numel()
                        scaling_grads.append(p.main_grad)
                    assert start <= num_elements
                self._fp8_main_grad_scales = scales
                self._fp8_main_grad_scale_invs = scale_invs
                self._fp8_main_grad_amaxs = amaxs
                self._scaling_grads = scaling_grads

            # Backward hook.
            # Accumulation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    if torch.is_tensor(param):
                        # Expand so we get access to grad_fn.
                        param_tmp = param.expand_as(param)
                        # Get the gradient accumulator function.
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        grad_acc.register_hook(self._make_param_hook(param))
                    else:
                        hook = self._fp8_make_param_hook(param)
                        grad_acc = param.register_backward_post_hook(hook)
                    self.grad_accs.append(grad_acc)

    def _fp8_make_param_hook(self, param):
        """Create the all-reduce hook for backprop for FP8 parameter."""

        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                Arithmetic.add_to_fp8(param.main_grad.value, param.main_grad.meta, param.grad)
                # Now we can deallocate grad memory.
                param.grad = None

        return param_hook

    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""

        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None

        return param_hook

    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the beginning of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()

    def broadcast_params(self):
        """Broadcast the parameters from rank zero to all other processes."""
        for param in self.module.parameters():
            torch.distributed.broadcast(
                param.data, src=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group()
            )

    def allreduce_gradients(self):
        """All-reduce gradients, not implemented."""
        raise NotImplementedError
