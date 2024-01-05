import functools

import torch
from torch.distributed.utils import _p_assert
from torch.distributed.fsdp._common_utils import FSDP_PREFIX, TrainingState
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp._runtime_utils import (
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
)
from torch.distributed.fsdp._init_utils import _get_default_comm_hook
from torch.distributed.algorithms._comm_hooks import default_hooks


class FP8FullyShardedDataParallel(FullyShardedDataParallel):
    def __init__(self, module, *args, **kwargs):
        for _, submodule in module.named_modules():
            params_to_process = list(submodule.named_parameters(recurse=False))
            for param_name, param in params_to_process:
                if not isinstance(param, torch.Tensor):
                    data = param.value.view(-1)
                    padded = 0
                    if data.numel() % 4 != 0:
                        padded = 4 - data.numel() % 4 
                        data = torch.nn.functional.pad(data, (0, padded))

                    data = data.view(dtype=torch.float32)
                    new_param = torch.nn.Parameter(data)
                    new_param._fp8 = True
                    new_param._original_shape = param.shape
                    new_param._padded = padded
                    new_param._meta = param.meta
                    new_param._scaling_metas = param._scaling_metas
                    setattr(submodule, param_name, new_param)
        
        super().__init__(module, *args, **kwargs)

        self._communication_hook = self._get_fp8_comm_hook()

    def _get_fp8_comm_hook(self):
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

    def forward(self, *args, **kwargs):
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            unshard_fn = functools.partial(_pre_forward_unshard, self, self._handles)
            reshard_fn = functools.partial(_post_forward_reshard, self, self._handles)
            args, kwargs = _pre_forward(
                self, self._handles, unshard_fn, self._fsdp_wrapped_module, args, kwargs
            )

            for handle in self._handles:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            i = 0
            for _, submodule in self._fsdp_wrapped_module.named_modules():
                for _, param in submodule.named_parameters(recurse=False):
                    if self._flat_param._metas[i] is not None:
                        param._fp8 = True
                        param._scaling_metas = self._flat_param._scaling_metas[i]
                        param._meta = self._flat_param._metas[i]
                        param._padded = self._flat_param._paddeds[i]
                        param._original_shape = self._flat_param._original_shapes[i]
                    i += 1
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(self, self._handles, reshard_fn, self, unused, output)

    
    def named_parameters(self, *args, **kwargs):
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        i = 0
        for param_name, param in super().named_parameters(*args, **kwargs):
            if self._flat_param._metas[i] is not None:
                param._meta = self._flat_param._metas[i]
                param._grad_meta = self._flat_param._scaling_metas[i]['wgrad']
            i += 1
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)