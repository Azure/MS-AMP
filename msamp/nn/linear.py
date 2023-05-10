# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""linear module in MS-AMP."""

import torch
import torch.nn.functional as F

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor, ScalingMeta, TensorDist
from msamp.nn import ScalingParameter, ScalingModule, model_state


class FP8Linear(ScalingModule):
    """Linear layer with FP8 support."""
    DEFAULT_WINDOW_SIZE = 16
    DEFAULT_WGRAD_WINDOW_SIZE = 1

    def __init__(self, in_features, out_features, use_bias=True, weight_qtype=Dtypes.kfloat16):
        """Constructor.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            use_bias (bool): Whether to use bias. Defaults to True.
            weight_qtype (Dtypes.QType): qtype of weight. Defaults to Dtypes.kfloat16.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_qtye = weight_qtype
        weight_dtype = Dtypes.get_dtype_from_qtype(weight_qtype)

        tensor = torch.empty((out_features, in_features), dtype=weight_dtype)
        self.weight = ScalingParameter(ScalingTensor(tensor, meta=ScalingMeta(weight_qtype, window_size=1)))

        if use_bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, ))
        else:
            self.register_parameter('bias', None)

        self.scaling_metas = dict(
            input=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=FP8Linear.DEFAULT_WINDOW_SIZE),
            wgrad=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=FP8Linear.DEFAULT_WGRAD_WINDOW_SIZE),
            ograd=ScalingMeta(Dtypes.kfloat8_e5m2, window_size=FP8Linear.DEFAULT_WINDOW_SIZE)
        )
        self.weight._scaling_metas = self.scaling_metas

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return F.linear(input, self.weight, bias=self.bias)

    def extra_repr(self):
        """Return the extra representation of this module."""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearReplacer:
    """A class to replace torch.nn.Linear with FP8Linear."""
    @staticmethod
    @torch.no_grad()
    def _build_fp8linear(linear, weight_qtype):
        """Build FP8Linear from torch.nn.Linear.

        Args:
            linear (torch.nn.Linear): Linear module.
            weight_qtype (Dtypes.QType): Qtype of weight.

        Returns:
            FP8Linear: FP8Linear module.
        """
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError('type of m must be torch.nn.Linear')

        fp8_linear = FP8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            use_bias=linear.bias is not None,
        ).cuda()

        linear = linear.cuda()

        if linear.bias is not None:
            fp8_linear.bias.data.copy_(linear.bias)

        weight = linear.weight
        weight = weight.cast(weight_qtype)
        if fp8_linear.weight.dtype != weight.dtype:
            raise ValueError(
                f'weight dtype is not same, dtype ins fp8 linear is {fp8_linear.weight.dtype},'
                f'dtype in weight is {weight.dtype}'
            )
        fp8_linear.weight.copy_(weight)

        return fp8_linear

    @classmethod
    def _replace(cls, model, weight_qtype):
        """Replace torch.nn.Linear with FP8Linear recursively in a model.

        Args:
            model (torch.nn.Module): Model to replace.
            weight_qtype (Dtypes.QType): Qtype of weight.

        Returns:
            model (torch.nn.Module): Model in which all Linear modules are replaced with FP8Linear.
        """
        if isinstance(model, torch.nn.Linear):
            if getattr(model, 'use_fp32_linear', False):
                return model
            fp8_net = cls._build_fp8linear(model, weight_qtype)
            return fp8_net
        else:
            for child_name, child in list(model.named_children()):
                setattr(model, child_name, cls._replace(child, weight_qtype))
        return model

    @classmethod
    def replace(cls, model, weight_qtype=Dtypes.kfloat16):
        """Replace torch.nn.Linear with FP8Linear in a model.

        Besides replace linear modules, it also broadcasts weights and register scaling data to model state.

        Args:
            model (torch.nn.Module): Model to replace.
            weight_qtype (Dtypes.QType, optional): Qtype of weight. Defaults to kfloat16.

        Return:
            model (torch.nn.Module): Model in which all Linear modules are replaced with FP8Linear.
        """
        model = cls._replace(model, weight_qtype)
        fp8_named_weights = [(k, p) for k, p in model.named_parameters() if isinstance(p, ScalingParameter)]

        fp8_names = [k for k, _ in fp8_named_weights]
        fp8_weights = [p for _, p in fp8_named_weights]
        TensorDist.broadcast(fp8_weights, src=0)

        for k, p in fp8_named_weights:
            p._param_name = k

        # DDP ignores the FP8 weights, and the optimizer provides a function `optimizer.all_reduce_grads(model)`
        # to sync them.
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, fp8_names)

        model_state.register_scaling_metas(model)
        return model
