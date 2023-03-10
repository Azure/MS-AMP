# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""linear module in MS-AMP."""

import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor, ScalingMeta, TensorDist
from msamp.nn import ScalingParameter, ScalingModule, model_state
from msamp.operators.gemm import Gemm


class FP8Linear(ScalingModule):
    """Linear layer with FP8 support."""
    DEFAULT_WINDOW_SIZE = 16
    EMPTY_GRAD_TENSOR = torch.nn.Parameter(torch.tensor([]))

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
            wgrad=ScalingMeta(Dtypes.kfloat8_e4m3, window_size=1),
            ograd=ScalingMeta(Dtypes.kfloat8_e5m2, window_size=FP8Linear.DEFAULT_WINDOW_SIZE)
        )

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        model_state.ready_to_scale_tensor = True
        shape = input.shape

        if len(shape) != 2:
            dim = shape[-1]
            input = input.reshape(-1, dim)

        output_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else input.dtype
        out = _FP8GemmFunction.apply(
            input, self.weight, self.scaling_metas, FP8Linear.EMPTY_GRAD_TENSOR.type(output_dtype)
        )
        if self.bias is not None:
            out = out + self.bias.type(output_dtype).view(1, -1)

        if len(shape) != 2:
            out = out.view(shape[:-1] + (-1, ))
        return out

    def extra_repr(self):
        """Return the extra representation of this module."""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class _FP8GemmFunction(torch.autograd.Function):
    """A function provides fp8 gemm forward and backward computations."""
    @staticmethod
    def forward(ctx, input, weight, metas, dtype_holder):
        """Forward function.

        Args:
            ctx: Context to store arbitrary data which can be retrieved during the backward pass.
            input (torch.Tensor): Input tensor.
            weight (ScalingParameter): Weight tensor.
            metas (dict): Scaling meta of input, weight and output.
            dtype_holder (torch.Tensor): A tensor to hold the output dtype. The required_grad of this tensor
                should be if input.required_grad is False.
        """
        ctx.metas = metas
        model_state.check_metas_in_flat(metas)
        input_meta = metas['input']
        input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
        weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)

        ctx.input_fp8 = input_fp8
        ctx.input_fp8.requires_grad = input.requires_grad
        ctx.weight_fp8 = weight_fp8
        ctx.weight = weight

        output_dtype = dtype_holder.dtype
        output_qtype = Dtypes.dtype_to_qtype[output_dtype]

        ctx.output_dtype = output_dtype
        ctx.output_qtype = output_qtype

        out = Gemm.fp8_gemm(weight_fp8, input_fp8, output_qtype, use_split_accumulator=False)
        return out

    @staticmethod
    def backward(ctx, output_grad):
        """Backward function.

        Args:
            ctx: Context to get the data stored in forward pass.
            output_grad (torch.Tensor): Output gradient tensor.

        Returns:
            tuple (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): The gradients of the arguments
                in forward function. None if no gradient.
        """
        # pytorch has a bug that output_grad.strides is 0. Use .contiguous() to fix it.
        output_grad = output_grad.contiguous()

        # We assign gradients to x.grad directly.
        metas = ctx.metas
        ograd_meta = metas['ograd']
        wgrad_meta = metas['wgrad']
        ograd_fp8 = output_grad.cast(Dtypes.kfloat8_e5m2, meta=ograd_meta)

        if ctx.input_fp8.requires_grad:
            weight_fp8_t = ctx.weight_fp8.t().contiguous()
            input_grad = Gemm.fp8_gemm(weight_fp8_t, ograd_fp8, ctx.output_qtype, use_split_accumulator=True)
        else:
            input_grad = None

        if ctx.weight.requires_grad:
            ograd_fp8_t = ograd_fp8.t().contiguous()
            input_fp8_t = ctx.input_fp8.t().contiguous()
            wgrad_qtype = ctx.output_qtype
            # compute weight gradient
            if ctx.weight.grad is None:
                wgrad = Gemm.fp8_gemm(
                    input_fp8_t,
                    ograd_fp8_t,
                    wgrad_qtype,
                    use_split_accumulator=True,
                )
            else:
                # gradient accumulation, old_wgrad is FP32 or FP16 without tensor scaling.
                old_wgrad = ctx.weight.grad.to(ctx.output_dtype)
                wgrad = Gemm.fp8_gemm(
                    input_fp8_t,
                    ograd_fp8_t,
                    wgrad_qtype,
                    accumulate=True,
                    out=old_wgrad,
                    use_split_accumulator=True,
                )
                del old_wgrad

            # wgrad above this line is torch.Tensor w/o tensor scaling
            wgrad = wgrad.cast(Dtypes.kfloat8_e4m3, meta=wgrad_meta, sync=True)

            ctx.weight.backward_grad_update(wgrad)

        return input_grad, None, None, None


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

        # register functions
        get_fp8_wgrads_name = 'get_fp8_wgrads'
        if hasattr(model, get_fp8_wgrads_name):
            raise ValueError(f'`{get_fp8_wgrads_name}` is already in model')

        # DDP ignores the FP8 weights, and the optimizer provides a function `optimizer.all_reduce_grads(model)`
        # to sync them.
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, fp8_names)

        def get_fp8_wgrads():
            return [p.grad for p in fp8_weights]

        setattr(model, get_fp8_wgrads_name, get_fp8_wgrads)

        model_state.register_scaling_metas(model)
        return model
