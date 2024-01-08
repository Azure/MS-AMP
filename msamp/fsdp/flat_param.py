
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp.flat_param import FlatParamHandle

class FP8FlatParamHandle(FlatParamHandle):
    def _init_flat_param(
        self,
        params: Sequence[Optional[nn.Parameter]],
        module: nn.Module,
        use_orig_params: bool,
    ) -> None:
        super()._init_flat_param(params, module, use_orig_params)

        metas = []
        paddeds = []
        original_shapes = []
        scaling_metas = []
        
        for param in self.flat_param._params:
            if hasattr(param, '_meta') and param._meta:
                metas.append(param._meta)
                paddeds.append(param._padded)
                original_shapes.append(param._original_shape)
                scaling_metas.append(param._scaling_metas)
            else:
                metas.append(None)
                paddeds.append(0)
                original_shapes.append(None)
                scaling_metas.append(None)

        self.flat_param._metas = metas
        self.flat_param._paddeds = paddeds
        self.flat_param._original_shapes = original_shapes
        self.flat_param._scaling_metas = scaling_metas

    def _init_shard_metadata(
        self,
        numel_padded: int,
        start: int,
        end: int,
    ) -> None:
        super()._init_shard_metadata(numel_padded, start, end)
        start_offset = 0
        end_offset = 0
        sharded_flat_param_numel = self.flat_param.numel()
        for i, meta in enumerate(self.flat_param._metas):
            start_offset += self.flat_param._numels[i-1] if i >=1 else 0
            end_offset += self.flat_param._numels[i]
            if meta is not None:
                start_rank = start_offset // sharded_flat_param_numel
                end_rank = (end_offset-1) // sharded_flat_param_numel
                ranks = list(range(start_rank, end_rank + 1))
                meta.group = dist.new_group(ranks=ranks)


    def _use_unsharded_views(self, as_params: bool) -> None:
        super()._use_unsharded_views(as_params)
        for i, param_info in enumerate(self.flat_param._param_infos):
            if hasattr(param_info.module, param_info.param_name):
                param = getattr(param_info.module, param_info.param_name)
                
                param._scaling_metas = self.flat_param._scaling_metas[i]
                param._meta = self.flat_param._metas[i]
                param._padded = self.flat_param._paddeds[i]
                param._original_shape = self.flat_param._original_shapes[i]
    
    @torch.no_grad()
    def _use_sharded_views(self) -> None:
        super()._use_sharded_views()
        for i, param_info in enumerate(self.flat_param._param_infos):
            if hasattr(param_info.module, param_info.param_name):
                param = getattr(param_info.module, param_info.param_name)
                if self.flat_param._metas[i] is not None:
                    param._meta = self.flat_param._metas[i]
                    param._grad_meta = self.flat_param._scaling_metas[i]['wgrad']


