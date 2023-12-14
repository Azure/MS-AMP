# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""distributed module in MS-AMP."""

import math

import torch
import torch.distributed as dist

from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.common.dtype import Dtypes, Floating
from msamp.common.utils import TransformerEngineWrapper
from msamp.nn.state import model_state
from msamp.operators.dist_op import DistOp


class _ScalingTensorReducer:
    """A reducer for scaling tensors."""
    def __init__(self, parameters, process_group, bucket_bytes_cap):
        """Constructor.

        Args:
            parameters (list): A list of ScalingTensor.
            process_group (torch.distributed.ProcessGroup): The process group to be used for distributed.
            bucket_bytes_cap (int): The capacity of each bucket.
        """
        parameters = list(parameters)
        if not all(isinstance(p, ScalingTensor) for p in parameters):
            raise ValueError('All parameters must be ScalingTensor')
        # Check if the devices of parameters are same.
        if not all(p.device == parameters[0].device for p in parameters):
            raise ValueError('All parameters must be on the same device')
        self.device = parameters[0].device
        self.parameters = parameters
        self.param_to_id = {p: i for i, p in enumerate(parameters)}
        self.buffer = None
        self.bucket_bytes_cap = bucket_bytes_cap
        self.process_group = process_group
        self.reduction_stream = torch.cuda.Stream(device=self.device)
        self._build_buckets(parameters)
        self._register_backward_hooks()
        self.bucket_unreduced_param_ids = dict()
        self.dist_handles = []

    def reset_buckets(self):
        """Reset the buckets after all parameters are reduced."""
        if len(self.bucket_unreduced_param_ids) > 0:
            raise RuntimeError('some gradients not reduced: {}'.format(list(self.bucket_unreduced_param_ids.keys())))
        self.bucket_unreduced_param_ids = {k: set(v) for k, v in self.bucket_to_param_ids.items()}

    def wait(self):
        """Wait for all aysnc operations to complete."""
        for handle in self.dist_handles:
            handle.wait()
        self.dist_handles.clear()

    def _create_buffer(self):
        """Create a buffer to store the flattened gradients."""
        buffer_size = sum(p.numel() for p in self.parameters)
        return torch.empty((buffer_size, ), dtype=torch.uint8, device=self.device)

    def _build_buckets(self, parameters):
        """Split the parameters into multiple buckets in reverse order.

        Args:
            parameters (list): A list of ScalingTensor.
        """
        bucket_bytes = 0
        bucket_id = 0
        bucket_offset = 0
        param_id_to_bucket_id = {}
        bucket_to_param_ids = {}
        bucket_to_range = {}
        for p in parameters[::-1]:
            param_id = self.param_to_id[p]
            nbytes = p.numel()
            param_id_to_bucket_id[param_id] = bucket_id
            bucket_to_param_ids.setdefault(bucket_id, []).append(param_id)
            bucket_bytes += nbytes
            if bucket_bytes >= self.bucket_bytes_cap:
                bucket_to_range[bucket_id] = (bucket_offset, bucket_offset + bucket_bytes)
                bucket_offset += bucket_bytes
                bucket_id += 1
                bucket_bytes = 0

        # Process the last bucket.
        if bucket_bytes > 0:
            bucket_to_range[bucket_id] = (bucket_offset, bucket_offset + bucket_bytes)
        self.param_id_to_bucket_id = param_id_to_bucket_id
        self.bucket_to_param_ids = bucket_to_param_ids
        self.bucket_to_range = bucket_to_range

    def _register_backward_hooks(self):
        """Register backward hooks for all parameters."""
        for p in self.parameters:
            p.register_backward_post_hook(self._get_backward_hook(p))

    def _get_backward_hook(self, param):
        """Get the backward hook for a parameter.

        Args:
            param (ScalingTensor): The parameter.

        Returns:
            The backward hook.
        """
        param_id = self.param_to_id[param]
        bucket_id = self.param_id_to_bucket_id[param_id]

        def hook_fn(*args, **kwargs):
            unreduced_param_ids = self.bucket_unreduced_param_ids[bucket_id]
            try:
                unreduced_param_ids.remove(param_id)
            except KeyError:
                raise RuntimeError('gradient is already reduced')
            if len(unreduced_param_ids) == 0:
                # the bucket is full, reduce it
                self._reduce_bucket(bucket_id)
                self.bucket_unreduced_param_ids.pop(bucket_id)

        return hook_fn

    def _reduce_bucket(self, bucket_id):
        """Reduce the gradients of all the parameters in a bucket.

        Args:
            bucket_id (int): The id of the bucket.
        """
        self.reduction_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.reduction_stream):
            # step 1: collect the gradients
            param_ids = self.bucket_to_param_ids[bucket_id]
            params = [self.parameters[i] for i in param_ids]
            grads = [p.grad for p in params]
            wgrad_qtype = Dtypes.kfloat8_e4m3
            for g in grads:
                if not hasattr(g, 'meta'):
                    meta = ScalingMeta(wgrad_qtype, group=self.process_group)
                    g.meta = meta
            metas = [g.meta for g in grads]

            # step 2: synchronize the amax
            amaxs = torch.stack([g.abs().max().float() for g in grads])
            scales = torch.stack([meta.scale for meta in metas])

            amaxs.nan_to_num_(nan=torch.inf, posinf=torch.inf)    # convert NAN to INF since reduce ignores NAN
            dist.all_reduce(amaxs, op=dist.ReduceOp.MAX, group=self.process_group)

            # step 3: re-compute scaling factor and update meta.amax
            wgrad_qtype = Dtypes.kfloat8_e4m3
            fp_max = Floating.qfp_max[wgrad_qtype]
            world_size = dist.get_world_size(self.process_group)
            pre_scale = 1.0 / math.sqrt(world_size)
            sf = ScalingMeta.compute_scaling_factor(amaxs, scales, fp_max, margin=0)
            sf.mul_(pre_scale)
            for meta, amax, scale in zip(metas, amaxs, sf):
                meta.amax[0] = amax
                meta.scale.copy_(scale)

            # step 4: quantize the gradients to FP8
            bucket_range = self.bucket_to_range[bucket_id]
            bucket_start, bucket_end = bucket_range
            bucket_offset = bucket_start
            dummy_amax = torch.empty((1, ), dtype=torch.float32, device=self.device)
            if self.buffer is None:
                self.buffer = self._create_buffer()
            for i, (grad, meta) in enumerate(zip(grads, metas)):
                fp8_grad = TransformerEngineWrapper.cast_to_fp8(
                    grad.view(1, -1),
                    meta.scale,
                    dummy_amax,
                    meta.scale_inv,
                    meta.qtype,
                ).view_as(grad)
                meta.scale_inv.data.copy_(torch.reciprocal(meta.scale))
                grads[i] = None
                # copy fp8_grad to buffer
                grad_numel = grad.numel()
                buf = self.buffer.narrow(0, bucket_offset, grad_numel).view_as(grad)
                buf.copy_(fp8_grad)
                params[i].grad = ScalingTensor(buf, meta)
                params[i].grad.div_(world_size)
                bucket_offset += grad_numel

            flat_fp8_grads = self.buffer.narrow(0, bucket_start, bucket_end - bucket_start)

            # step 5: allreduce the gradients
            torch.cuda.default_stream().wait_stream(self.reduction_stream)
            dist_handle = DistOp.all_reduce(
                flat_fp8_grads, wgrad_qtype, dist.ReduceOp.SUM, self.process_group, async_op=True
            )
            self.dist_handles.append(dist_handle)


class _DDPSink(torch.autograd.Function):
    """A class for running various functions in DDP.

    Such functions are reset buckets before forward and wait for allreduce after backward.
    """
    @staticmethod
    def forward(ctx, reducer, empty, *inputs):
        """Reset the buckets and return the inputs.

        Argss:
            ctx (Context): The context to store arbitrary data which can be retrieved during the backward pass.
            reducer (_ScalingTensorReducer): The reducer for reducing the gradients.
            empty (torch.Tensor): An empty tensor whose requires_grad is True to trigger the backward function.

        Returns:
            The inputs.
        """
        ctx.set_materialize_grads(False)
        ctx.reducer = reducer
        reducer.reset_buckets()
        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Wait for allreduce complete and return the gradients.

        Args:
            ctx (Context): The context to get the data stored in forward pass.
            grad_outputs (tuple): The gradients of the outputs.

        Returns:
            The gradients of outputs.
        """
        ctx.reducer.wait()
        return (None, None, *grad_outputs)


class FP8DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """A wrapper of DistributedDataParallel to support all-reduce FP8 gradients."""
    def __init__(self, module, **kwargs):
        """Constructor.

        Args:
            module (torch.nn.Module): The module to be wrapped.
            kwargs (dict): The rest arguments for DistributedDataParallel.
        """
        super().__init__(module, **kwargs)

        scaling_params = [p for p in self.parameters() if p.requires_grad and isinstance(p, ScalingTensor)]
        if len(scaling_params) > 0:
            self.scaling_tensor_reducer = _ScalingTensorReducer(
                scaling_params, self.process_group, self.bucket_bytes_cap
            )
            model_state.use_fp8_ddp = True

    def forward(self, *inputs, **kwargs):
        """Apply _DDPSink in forward function.

        Args:
            inputs (tuple): The input tensors.
            kwargs (dict): The keyword arguments.
        """
        if model_state.use_fp8_ddp and torch.is_grad_enabled():
            inputs = _DDPSink.apply(self.scaling_tensor_reducer, torch.tensor([], requires_grad=True), *inputs)
        out = super().forward(*inputs, **kwargs)
        return out


torch.nn.parallel.DistributedDataParallel = FP8DistributedDataParallel
