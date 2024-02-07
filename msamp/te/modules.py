# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP te.modules module."""

import torch
import transformer_engine.pytorch as te
import transformer_engine_extensions as tex
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

from msamp.common.tensor import ScalingTensor
from msamp.nn import ScalingModule

# set the function `untyped_storage` for TransformerEngine
if not hasattr(torch.Tensor, 'untyped_storage'):
    torch.Tensor.untyped_storage = lambda self: self.data.storage().untyped()


def set_activation_dtype(self, inp):
    """Set activation data type for AMP.

    Args:
        self (TransformerEngineBaseModule): Module instance.
        inp (torch.Tensor or ScalingTensor): Input tensor.
    """
    # Native AMP (`torch.autocast`) gets highest priority
    if torch.is_autocast_enabled():
        self.activation_dtype = torch.get_autocast_gpu_dtype()
        return

    # All checks after this have already been performed once, thus skip
    # We assume that user doesn't change input types across iterations
    if hasattr(self, 'activation_dtype'):
        return

    assert all(
        (
            (inp.dtype == param.dtype) if param is not None and not isinstance(param, ScalingTensor) else True
            for param in self.parameters()
        )
    ), ('Data type for activations and weights must '
        'match when outside of autocasted region')
    assert all(((inp.dtype == buf.dtype) if buf is not None else True for buf in self.buffers())
               ), ('Data type for activations and buffers must '
                   'match when outside of autocasted region')
    self.activation_dtype = inp.dtype


class MSAMPTransformerEngineBaseModule:
    """A base module for MS-AMP transformer engine modules."""
    def set_fp8_weights(self):
        """Initializes FP8 weights for the module as class attributes."""
        # when is_first_microbatch is not None
        # call every microbatch
        # cache weight_fp8, weight_t_fp8 for gradient accumulation
        # set_fp8_weights will clean up the cache
        if not self.is_msamp_module:
            super().set_fp8_weights()
        else:
            for i, shape in enumerate(self.fp8_weight_shapes, start=1):
                weight_cast_attr = f'weight{i}_fp8'
                weight_transpose_attr = f'weight{i}_t_fp8'

                if (hasattr(self, weight_cast_attr) and getattr(self, weight_cast_attr).shape == shape):
                    return

                setattr(
                    self, weight_cast_attr,
                    Float8Tensor(
                        data=torch.empty(
                            (0, 0),
                            device=torch.cuda.current_device(),
                            dtype=torch.uint8,
                        ),
                        fp8_dtype=tex.DType.kFloat8E4M3,
                        fp8_scale_inv=1,
                    )
                )

                setattr(
                    self, weight_transpose_attr,
                    Float8Tensor(
                        data=torch.empty(
                            (0, 0),
                            device=torch.cuda.current_device(),
                            dtype=torch.uint8,
                        ),
                        fp8_dtype=tex.DType.kFloat8E4M3,
                        fp8_scale_inv=1,
                    )
                )

    @property
    def is_msamp_module(self):
        """Whether this module is MS-AMP module."""
        if not hasattr(self, '_is_msamp_module'):
            self._is_msamp_module = False
        return self._is_msamp_module

    @is_msamp_module.setter
    def is_msamp_module(self, value):
        """Set whether this module is MS-AMP module.

        Args:
            value (bool): True if this module is MS-AMP module.
        """
        self._is_msamp_module = value

    def get_fp8_weights_empty_tensors(self, is_first_microbatch):
        """Returns empty tensors to be later used to store fp8 version of weights and their transposes.

        Args:
            is_first_microbatch (bool): Whether this is the first microbatch.

        Returns:
            a list of fp8 weight tensors.
        """
        # when is_first_microbatch is None, create empty tensors
        if not self.is_msamp_module:
            return super().get_fp8_weights_empty_tensors(is_first_microbatch)
        # MS-AMP
        old_fp8_weight_shapes = self.fp8_weight_shapes
        self.fp8_weight_shapes = [(0, 0)] * len(old_fp8_weight_shapes)
        # create empty tensor as placeholder
        rtn = super().get_fp8_weights_empty_tensors(is_first_microbatch)
        self.fp8_weight_shapes = old_fp8_weight_shapes
        return rtn


class MSAMPLinear(MSAMPTransformerEngineBaseModule, te.Linear, ScalingModule):
    """MS-AMP Linear module."""
    pass


class MSAMPLayerNormLinear(MSAMPTransformerEngineBaseModule, te.LayerNormLinear, ScalingModule):
    """MS-AMP LayerNormLinear module."""
    pass


class MSAMPLayerNormMLP(MSAMPTransformerEngineBaseModule, te.LayerNormMLP, ScalingModule):
    """MS-AMP LayerNormMLP module."""
    pass


class CtxWrapper:
    """A wrapper of FunctionCtx which supports ScalingTenor."""
    def __init__(self, ctx):
        """Init a CtxWrapper.

        Args:
            ctx (FunctionCtx): Function context.
        """
        self.__dict__['ctx'] = ctx

    def __getattr__(self, name):
        """Get attribute by name.

        Args:
            name (str): Attribute name.

        Returns:
            Attribute value.
        """
        return self.__dict__.get(name, getattr(self.__dict__['ctx'], name))

    def __setattr__(self, name, value):
        """Set attribute by name.

        Args:
            name (str): Attribute name.
            value (object): Attribute value.
        """
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.ctx, name, value)

    def save_for_backward(self, *args):
        """Save tensors for backward.

        Args:
            args (tuple): Tensors to save.
        """
        torch_args = []
        scaling_args = []
        for a in args:
            if isinstance(a, ScalingTensor):
                scaling_args.append(a)
                torch_args.append(None)
            else:
                torch_args.append(a)
                scaling_args.append(None)
        self.ctx.save_for_backward(*torch_args)
        self.ctx.scaling_args = scaling_args

    @property
    def saved_tensors(self):
        """Get saved tensors."""
        tensors = list(self.ctx.saved_tensors)
        for i, v in enumerate(self.ctx.scaling_args):
            if v is not None:
                tensors[i] = v
        return tensors


class TeModuleOverrider:
    """An Overrider to override some modules and functions in Transformer Engine."""
    @classmethod
    def override(cls):
        """Override transformer engine modules and functions."""
        cls._override_funcions()
        cls._override_classes()

    @classmethod
    def _override_funcions(cls):
        TransformerEngineBaseModule.set_activation_dtype = set_activation_dtype
        cls._override_function(te.module.linear, '_Linear')
        cls._override_function(te.module.layernorm_linear, '_LayerNormLinear')
        cls._override_function(te.module.layernorm_mlp, '_LayerNormMLP')

    @classmethod
    def _override_classes(cls):
        """Override some classes in transformer engine."""
        te.Linear = MSAMPLinear
        te.LayerNormLinear = MSAMPLayerNormLinear
        te.LayerNormMLP = MSAMPLayerNormMLP

        te.attention.Linear = MSAMPLinear
        te.attention.LayerNormLinear = MSAMPLayerNormLinear

        te.transformer.Linear = MSAMPLinear
        te.transformer.LayerNormLinear = MSAMPLayerNormLinear
        te.transformer.LayerNormMLP = MSAMPLayerNormMLP

    @staticmethod
    def _override_function(mod, func_name):    # noqa: C901
        """Override a function in a module.

        Args:
            mod (module): Module.
            func_name (str): Function name.
        """
        old_func = getattr(mod, func_name)
        assert issubclass(old_func, torch.autograd.Function), (func_name, old_func)

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, place_holder, *args):
                scaling_tensors = []
                for i, a in enumerate(args):
                    if isinstance(a, ScalingTensor):
                        scaling_tensors.append((i, a))
                if ctx is not None:
                    ctx.scaling_tensors = scaling_tensors
                    ctx = CtxWrapper(ctx)
                return old_func.forward(ctx, *args)

            @staticmethod
            def backward(ctx, *args):
                ctx = CtxWrapper(ctx)
                grads = list(old_func.backward(ctx, *args))
                for i, v in ctx.scaling_tensors:
                    if not v.requires_grad:
                        continue
                    assert grads[i] is not None
                    if v.grad is None:
                        v.grad = grads[i]
                    elif torch.is_tensor(v.grad):
                        v.grad += grads[i]
                    else:
                        assert isinstance(v.grad, ScalingTensor)
                        v.grad = v.grad.to(grads[i].dtype) + grads[i]
                    v.backward_grad_update(v.grad)
                    grads[i] = None
                return (None, ) + tuple(grads)

        class Wrapper:
            EMPTY_TENSOR = torch.tensor([], requires_grad=True)

            @staticmethod
            def forward(ctx, *args):
                return Func.forward(ctx, Wrapper.EMPTY_TENSOR.detach(), *args)

            @staticmethod
            def apply(*args):
                return Func.apply(Wrapper.EMPTY_TENSOR, *args)

        setattr(mod, func_name, Wrapper)


TeModuleOverrider.override()
