# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP fsdp.fully_sharded_data_parallel module."""

import torch
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.fsdp._init_utils import _get_default_comm_hook

from msamp.fsdp.flat_param import FP8FlatParamHandle
from msamp.fsdp._runtime_utils import _fp8_post_backward_hook


def _get_fp8_comm_hook(self):
    """Get the communication hook for fp8 gradient."""
    def _fp8_allreduce_hook(state, grad, output):
        start = 0
        end = 0
        has_meta = False
        for meta in self._flat_param._metas:
            if meta is not None:
                has_meta = True
                break
        if has_meta:
            for i, meta in enumerate(self._flat_param._metas):
                start += self._flat_param._numels[i - 1] if i >= 1 else 0
                end += self._flat_param._numels[i]
                if meta is not None:
                    from msamp.common.dtype import Dtypes
                    from msamp.operators.dist_op import DistOp
                    dtype = Dtypes.get_dtype_from_qtype(meta.qtype)
                    DistOp.enable_fp8(meta.qtype)
                    torch.distributed.all_reduce(grad[start:end].view(dtype), group=state.process_group)
                    DistOp.disable_fp8()
                else:
                    default_hooks.allreduce_hook(
                        state=state,
                        grad=grad[start:end],
                    )
            start = self.rank * output.numel()
            end = (self.rank + 1) * output.numel()
            output.copy_(grad[start:end])
        else:
            _get_default_comm_hook()(state, grad, output)

    return _fp8_allreduce_hook


class FP8FullyShardedDataParallel(FullyShardedDataParallel):
    """A FullyShardedDataParallel with supports fp8."""
    def __init__(self, module, *args, **kwargs):
        """Constructor."""
        super().__init__(module, *args, **kwargs)

    @classmethod
    def override(cls):
        """Override FlatParamHandle and _post_backward_hook with class/function which support fp8."""
        torch.distributed.fsdp._init_utils.FlatParamHandle = FP8FlatParamHandle
        torch.distributed.fsdp._runtime_utils._post_backward_hook = _fp8_post_backward_hook
        FullyShardedDataParallel._get_fp8_comm_hook = _get_fp8_comm_hook


FP8FullyShardedDataParallel.override()
