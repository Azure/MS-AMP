# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP fsdp.flat_param module."""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributed.fsdp.flat_param import FlatParamHandle


class FP8FlatParamHandle(FlatParamHandle):
    """A handle for a flat parameter which may have fp32 and fp8."""
    def _init_flat_param(
        self,
        params: Sequence[Optional[nn.Parameter]],
        module: nn.Module,
        use_orig_params: bool,
    ) -> None:
        """Initialize the flat parameter and save fp8 related metadata."""
        super()._init_flat_param(params, module, use_orig_params)

        metas = []
        paddeds = []
        original_shapes = []
        scaling_metas = []

        for param in self.flat_param._params:
            if hasattr(param, '_meta') and param._meta is not None:
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

    def _use_unsharded_views(self, as_params: bool) -> None:
        """Use unsharded views of the flat parameter.

        It will also set fp8 related attritutes, which will be use in msamp.nn.functional.
        """
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
        """Use sharded views of the flat parameter and set meta of scaling tensor, which will be used in optimizer."""
        super()._use_sharded_views()
        for i, param_info in enumerate(self.flat_param._param_infos):
            if hasattr(param_info.module, param_info.param_name):
                param = getattr(param_info.module, param_info.param_name)
                if self.flat_param._metas[i] is not None:
                    param._meta = self.flat_param._metas[i]
                    param._grad_meta = self.flat_param._scaling_metas[i]['wgrad']
