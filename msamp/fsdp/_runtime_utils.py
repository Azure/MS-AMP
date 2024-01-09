# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP fsdp._runtime_utils module."""

from typing import no_type_check

import torch
from torch.distributed.fsdp import FullyShardedDataParallel

old_post_backward_hook = torch.distributed.fsdp._runtime_utils._post_backward_hook


@no_type_check
@torch.no_grad()
def _fp8_post_backward_hook(state, handle, *unused):
    """A post-backward communication hook which supports fp8."""
    if not isinstance(state, FullyShardedDataParallel):
        return old_post_backward_hook(state, handle, *unused)

    accumulate_grad = hasattr(state._flat_param, '_saved_grad_shard')
    if accumulate_grad and not torch.all(state._flat_param._saved_grad_shard == 0):
        raise NotImplementedError('accumulate_grad is not supported for fp8')

    old_communication_hook = state._communication_hook
    state._communication_hook = state._get_fp8_comm_hook()
    old_post_backward_hook(state, handle, *unused)
    state._communication_hook = old_communication_hook
