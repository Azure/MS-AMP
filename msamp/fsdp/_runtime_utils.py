from typing import no_type_check

import torch
from torch.distributed.fsdp import FullyShardedDataParallel

old_post_backward_hook = torch.distributed.fsdp._runtime_utils._post_backward_hook

@no_type_check
@torch.no_grad()
def _post_backward_hook(state, handle, *unused):
    if not isinstance(state, FullyShardedDataParallel):
        return old_post_backward_hook(state, handle, *unused)

    old_communication_hook = state._communication_hook
    state._communication_hook = state._get_fp8_comm_hook()
    old_post_backward_hook(state, handle, *unused)
    state._communication_hook = old_communication_hook

