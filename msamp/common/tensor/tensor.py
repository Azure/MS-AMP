# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP tensor module."""

import torch
import torch.nn.functional as F
from msamp.common.tensor import ScalingMeta
from msamp.common.tensor import HookManager
from msamp.common.dtype import Dtypes
from msamp.common.tensor import TypeCast
from msamp.common.utils import TransformerEngineWrapper


class ScalingTensor:
    """Customized tensor with scaling."""
    class UniqueDtypeDecorator:
        """A decorator class to check whether dtype is supported and parameters are uniqie."""
        def __init__(self, clsmethod=False, ignore_none=False, **kwargs):
            """Constructor.

            Args:
                clsmethod (bool, optional): is class method, defaults to False.
                ignore_none (bool, optional): ignore none, defaults to False.
                **kwargs (dict): Keyword arguments.
            """
            self.clsmethod = clsmethod
            self.ignore_none = ignore_none
            self.kwargs = kwargs

        def __call__(self, func):
            """Override __call__.

            Args:
                func: function to decorate.

            Return:
                wrapper: a wrapper function to check dtype.
            """
            def wrapper(*args, **kwargs):
                """Check whether dtype is supported or arguments are duplicated.

                Args:
                    *args (list): Arguments.
                    **kwargs (dict): Keyword arguments.

                Return:
                    The result of call to decorated function.
                """
                dtype2name = dict()
                for k, v in self.kwargs.items():
                    dtype2name[v] = k
                instance = None
                if self.clsmethod:
                    instance, args = args[0], args[1:]
                for a in args:
                    if a is None and self.ignore_none:
                        continue
                    t = type(a)
                    name = dtype2name.get(t, None)
                    if name is None:
                        raise TypeError(f'Unsupported dtype in arguments: {t}')
                    if name in kwargs:
                        raise TypeError(f'Duplicated argument: {name}')
                    kwargs[name] = a
                if self.clsmethod:
                    return func(instance, **kwargs)
                return func(**kwargs)

            return wrapper

    def __init__(self, value, meta):
        """Constructor.

        Args:
            value (torch.Tensor): the underlying value data.
            meta (ScalingMeta): the scaling data.
        """
        super().__init__()
        self.value = value
        self._grad = None
        self._backward_post_hooks = HookManager()
        self.meta = meta

        meta_dtype = Dtypes.get_dtype_from_qtype(meta.qtype)
        if meta_dtype != value.dtype:
            raise TypeError(
                f'Type mismatch, value.dtype is {value.dtype}, \
meta.dtype is {meta_dtype} (meta.qtype is {meta.qtype}).'
            )
        self._requires_grad = False

    @property
    def grad(self):
        """Decoration function to access _grad."""
        return self._grad

    @grad.setter
    def grad(self, grad):
        """Set grad.

        Args:
            grad (torch.Tensor): Grad of the tensor.
        """
        self._grad = grad

    @grad.deleter
    def grad(self):
        """Delete grad."""
        del self._grad

    def register_backward_post_hook(self, fn):
        """Register hook function, which will be invoked when backward_grad_update is called.

        Args:
            fn: a function should have the following signature: hook(module, input, output) -> None.

        Return:
            _RemoveHandle: A remove handle which can be used to delete hook.
        """
        return self._backward_post_hooks.register_hook(fn)

    def backward_grad_update(self, grad):
        """Update backward grad.

        Args:
            grad: (torch.Tensor): backward grad.
        """
        self.grad = grad
        self._backward_post_hooks(grad)

    def detach(self):
        """Returns a new ScalingTensor, detached from the current graph.

        Args:
            ScalingTensor: a new detached ScalingTensor.
        """
        return ScalingTensor(self.value, self.meta)

    @UniqueDtypeDecorator(
        dtype=torch.dtype,
        device=torch.device,
        non_blocking=bool,
        clsmethod=True,
        ignore_none=True,
    )
    def to(self, dtype=None, **kwargs):
        """Performs ScalingTensor dtype and device conversion.

        Args:
            dtype (torch.dtype, optional): currently only supports torch.float, torch.float16: and torch.float32.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: tensor with desired dtype and device.
        """
        rtn = self
        if dtype is not None:
            if dtype == torch.float:
                rtn = rtn.float()
            elif dtype == torch.float16:
                rtn = rtn.half()
            elif dtype == torch.bfloat16:
                rtn = rtn.bfloat16()
            else:
                raise TypeError(f'unsupported dtype: {dtype}')
        if isinstance(rtn, ScalingTensor):
            rtn.value = rtn.value.to(**kwargs)
        else:
            rtn = rtn.to(**kwargs)
        return rtn

    @torch.no_grad()
    def mul_(self, other):
        """Multiplies self by other value.

        Args:
            other (torch.Tensor or float): The value to multiplies, could be single-element tensor or float.

        Return:
            ScalingTensor: current object.
        """
        if torch.is_tensor(other):
            if other.numel() != 1:
                raise ValueError('The tensor must by single-element tensor.')
            self.meta.scale_inv *= other.view_as(self.meta.scale_inv)
        else:
            self.meta.scale_inv *= other
        return self

    @torch.no_grad()
    def div_(self, other):
        """Divides tensor by other value.

        Args:
            other (torch.Tensor or float): The value to divide, could be single-element tensor or float.

        Return:
            ScalingTensor: current object.
        """
        if torch.is_tensor(other):
            if other.numel() != 1:
                raise ValueError('The tensor must by single-element tensor.')
            self.meta.scale_inv /= other.view_as(self.meta.scale_inv)
        else:
            self.meta.scale_inv /= other
        return self

    def _get_cast_from_fn(self):
        """Get cast_from function.

        Return:
            func: a function with a signature cast_from_xx(input, meta, otype) => torch.Tensor.
        """
        if Dtypes.is_fp8_qtype(self.meta.qtype):
            return TypeCast.cast_from_fp8
        elif self.meta.qtype in [Dtypes.kfloat16, Dtypes.kbfloat16]:
            return TypeCast.cast_from_fp16
        elif self.meta.qtype == Dtypes.kfloat32:
            return TypeCast.cast_from_fp32
        raise TypeError(f'Unsupported Type: {self.meta.qtype}')

    def cast(self, qtype):
        """Cast the ScalingTensor by qtype.

        Args:
            qtype (Dtypes.QType): the qtype to cast.
                Supported qtype:
                    Dtypes.kfloat8_e4m3 | Dtypes.kfloat8_e5m2 |
                    Dtypes.kfloat16 | Dtypes.kbfloat16 | Dtypes.kfloat32

        Return:
            ScalingTensor: a ScalingTensor with desired qtype.
        """
        if qtype == self.meta.qtype:
            return self

        if Dtypes.is_fp8_qtype(self.meta.qtype):
            return self.float().cast(qtype)

        # Cast ScalingTensor to ScalingTensor with another data type
        old_amax = self.meta.amax[0]
        # clone amax for the new ScalingTensor
        meta = ScalingMeta(qtype=qtype, amax=self.meta.amax.clone(), window_size=self.meta.window_size)
        # re-compute scaling factor with the maximum absolute value `amax`
        meta.reset_scaling_factor()
        # unscale it since self.value has been scaled by `self.scale`
        scale_inv = torch.reciprocal(meta.scale)
        # correct meta.scale
        # since self.value * self.meta.scale_inv = new_value * meta.scale_inv,
        # new_value = self.value * (1 / meta.scale_inv * meta.scale_inv)
        #           = self.value * (meta.scale * meta.scale_inv)
        # where meta.scale * meta.scale_inv is the scaling factor to quantize self.value
        meta.scale.mul_(self.meta.scale_inv)
        # quantize self.value with the current scaling factor, namely meta.scale
        # therefore, disable the update of scaling factor in time
        with ScalingMeta.in_time_scaling_context(enabled=False):
            if Dtypes.is_fp8_qtype(meta.qtype):
                value = TypeCast.cast_to_fp8(self.value, meta)
            else:
                value = TypeCast.cast_to_fp16(self.value, meta)
            # recover the following values since they are changed in cast_to_fp8/fp16.
            meta.amax[0] = old_amax
            meta.scale_inv.copy_(scale_inv)
        return ScalingTensor(value, meta=meta)

    def fp8_transpose(self):
        """FP8 scaling tensor transpose."""
        if not Dtypes.is_fp8_qtype(self.meta.qtype):
            raise TypeError(f'Unsupported FP8 transpose type: {self.meta.qtype}')
        return TransformerEngineWrapper.fp8_transpose(self)

    def t(self):
        """Transpose tensor.

        Return:
            ScalingTensor: a tensor that is a transposed version of this object.
        """
        return ScalingTensor(self.value.t(), self.meta)

    def contiguous(self):
        """Returns a contiguous in memory tensor.

        Return:
            ScalingTensor: a tensor that is contiguous in memory and contains same data as self tensor.
        """
        meta = self.meta
        if not self.value.is_contiguous():
            meta = meta.clone()
        return ScalingTensor(self.value.contiguous(), meta)

    def has_inf_or_nan(self):
        """Check if tensor is infinite or nan.

        Return:
            bool: return True if absolute maxmium value is not finite, otherwise False.
        """
        return not bool(torch.isfinite(self.meta.amax[0]))

    def float(self):
        """Cast value tensor to float.

        Return:
            torch.Tensor: a tensor whose dtype is torch.float.
        """
        fn = self._get_cast_from_fn()
        return fn(self.value, self.meta, Dtypes.kfloat32)

    def half(self):
        """Cast value tensor to float16.

        Return:
            torch.Tensor: a tensor whose dtype is torch.float16.
        """
        fn = self._get_cast_from_fn()
        return fn(self.value, self.meta, Dtypes.kfloat16)

    def bfloat16(self):
        """Cast value tensor to bfloat16.

        Return:
            torch.Tensor: a tensor whose dtype is torch.bfloat16.
        """
        fn = self._get_cast_from_fn()
        return fn(self.value, self.meta, Dtypes.kbfloat16)

    def abs(self):
        """Compute the absolute value.

        Return:
            ScalingTensor: a ScalingTensor instance whose value tensor is absolute value of self.value.
        """
        if self.value.dtype == torch.uint8:
            return ScalingTensor(self.value & 0x7F, self.meta.clone())
        if self.value.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(f'not support {self.value.dtype}')

        return ScalingTensor(self.value.abs(), self.meta.clone())

    def min(self):
        """Get the minimum value.

        Return:
            torch.Tensor: the minimum single-element tensor.
        """
        return self.float().min()

    def max(self):
        """Get the maximum value.

        Return:
            torch.Tensor: the maximum single-element tensor.
        """
        return self.float().max()

    @property
    def qtype(self):
        """Decoration to get qtype in meta."""
        return self.meta.qtype

    @property
    def grad_fn(self):
        """Decoration function to get grad function, currently only returns None."""
        return None

    def requires_grad_(self, requires_grad):
        """Set requires_grad.

        Args:
            requires_grad (bool): whether require gradient.
        """
        self._requires_grad = requires_grad

    def zero_(self):
        """Fill tensor with zero.

        Return:
            ScalingTensor: current object whose value tensor is filled with tensor.
        """
        # [TODO] lazy zero
        self.value.zero_()
        return self

    def flatten(self):
        """Flatten tensor.

        Return:
            ScalingTensor: a new ScalingTensor object whose value tensor is flattened.
        """
        return ScalingTensor(self.value.flatten(), self.meta)

    @property
    def is_cuda(self):
        """Check if value tensor is stored in GPU.

        Return:
            bool: True if value tensor is tored in GPU, otherwise False.
        """
        return self.value.is_cuda

    def isnan(self):
        """Check if tensor is not a number.

        Return:
            bool: True if value tensor is nan, otherwise False.
        """
        if Dtypes.is_fp8_qtype(self.qtype):
            return self.value == 127
        return torch.isnan(self.value)

    @property
    def shape(self):
        """Return shape of tensor.

        Return:
            torch.Size: the shape of value tensor.
        """
        return self.value.shape

    @property
    def is_meta(self):
        """Return is_meta property of tensor.

        Return:
            bool: the is_meta property of value tensor.
        """
        return self.value.is_meta

    @property
    def size(self):
        """Return size function of tensor.

        Return:
            func: the size method of value tensor.
        """
        return self.value.size

    def numel(self):
        """Get number of elements in tensor.

        Return:
            int: The number of elements in value tensor.
        """
        return self.value.numel()

    def nelement(self):
        """Get number of elements in tensor.

        Return:
            int: The number of elements in value tensor.
        """
        # nelement is an alias for numel
        return self.numel()

    @property
    def device(self):
        """Get device.

        Return:
            torch.device: the device of value tensor.
        """
        return self.value.device

    @property
    def data(self):
        """Get underlying data.

        Return:
            ScalingTensor: a detached tensor.
        """
        return self.detach()

    @data.setter
    def data(self, data):
        """Set data.

        Args:
            data (torch.Tensor or ScalingTensor): data to set.
        """
        # ref
        with torch.no_grad():
            if isinstance(data, ScalingTensor):
                if self.meta.locked and self.meta is not data.meta:
                    raise ValueError('This ScalingTensor is locked.')
                self.value.data = data.value
                self.meta = data.meta
            else:
                if not isinstance(data, torch.Tensor):
                    raise TypeError('The type of data is not supported')
                self.value.data = data

    def data_ptr(self):
        """Get data pointer.

        Returns the address of the first element of the tensor.
        """
        return self.value.data_ptr()

    def copy_(self, src):
        """Copy from another tensor.

        Args:
            src (ScalingTensor): source tensor.

        Return:
            ScalingTensor: current instance.
        """
        if not isinstance(src, ScalingTensor):
            raise TypeError(f'The input should be a ScalingTensor rather than {type(src)}')

        # copy
        with torch.no_grad():
            if self.value.dtype != src.value.dtype:
                raise TypeError('The dtype of value must be same.')
            self.value.copy_(src.value)
            self.meta.copy_(src.meta)
        return self

    def cast_(self, qtype):
        """In-place cast tensor.

        Args:
            qtype (Dtypes.QType): type to cast.

        Return:
            ScalingTensor: current instance.
        """
        if self.qtype == qtype:
            return self
        e = self.cast(qtype)
        self.value.data = e.value
        self.meta.copy_(e.meta)
        return self

    @property
    def dtype(self):
        """Get dtype of tensor.

        Return:
            torch.dtype: the dtype of value tensor.
        """
        return self.value.dtype

    def type(self):
        """Get type of tensor.

        Return:
            str: the type of value tensor.
        """
        return self.__class__.__module__ + '.' + self.__class__.__qualname__

    @property
    def is_leaf(self):
        """Check if tensor is leaf.

        Args:
            bool: currently always return True.
        """
        return True

    @property
    def is_sparse(self):
        """Check if tensor is sparse.

        Return:
            bool: True if value tensor is sparse, otherwise False.
        """
        return self.value.is_sparse

    def is_contiguous(self):
        """Check if tensor is contiguous.

        Return:
            bool: True if value tensor is contiguous, otherwise False.
        """
        return self.value.is_contiguous()

    def is_floating_point(self):
        """Check if tensor is floating point.

        Return False to avoid converting to torch.tensor when model.float() is called.

        Return:
            bool: currently always return False.
        """
        return False

    def is_complex(self):
        """Check if tensor is complex.

        Return:
            bool: currently always return False.
        """
        return False

    def pad(self, pad):
        """Pad tensor.

        Args:
            pad (tuple): padding size.

        Return:
            ScalingTensor: a new ScalingTensor object whose value tensor is padded.
        """
        return ScalingTensor(F.pad(self.value, pad), self.meta)

    def clone(self):
        """Clone tensor.

        Return:
            ScalingTensor: a new ScalingTensor object whose value tensor is cloned.
        """
        return ScalingTensor(self.value.clone(), self.meta.clone())

    def cpu(self, *args, **kwargs):
        """Move tensor to CPU.

        Args:
            *args: arguments.
            **kwargs: keyword arguments.

        Return:
            ScalingTensor: a new ScalingTensor object whose value tensor is moved to CPU.
        """
        value = self.value.cpu()
        meta = self.meta.clone()
        return ScalingTensor(value, meta)

    def cuda(self, *args, **kwargs):
        """Move tensor to GPU.

        Args:
            args: arguments.
            kwargs: keyword arguments.

        Return:
            ScalingTensor: a new ScalingTensor object whose value tensor is moved to GPU.
        """
        self.value = self.value.cuda()
        if not self.meta.is_cuda:
            if self.meta.locked:
                raise ValueError('This ScalingTensor is locked.')
            self.meta = self.meta.cuda()
        return self

    def __len__(self):
        """Get length of tensor.

        Return:
            int: the length of value tensor.
        """
        return len(self.value)

    def __repr__(self):
        """Get string representation of tensor.

        Return:
            str: the string representation of tensor.
        """
        return f'ScalingTensor({self.float()}, meta={self.meta}'


class TorchOverider:
    """Class to override torch attributes and functions."""
    one_scales = dict()
    torch_unary_funcs = ['torch.zeros_like', 'torch.ones_like', 'torch.overrides.is_tensor_like']

    @classmethod
    def override(cls):
        """Override torch attributes and functions."""
        torch.Tensor.cast = cls._cast_to_scalingtensor
        torch.Tensor.fused_cast_transpose = cls._fused_cast_transpose_to_scalingtensors
        torch.Tensor.qtype = property(lambda self: Dtypes.dtype_to_qtype[self.dtype])
        cls._override_unary_func()
        torch.is_floating_point = cls._get_wrapper_for_scalingtensor(
            torch.is_floating_point, lambda x: x.is_floating_point()
        )
        torch._amp_foreach_non_finite_check_and_unscale_ = cls._get_wrapper_for_grad_check_and_unscale(
            torch._amp_foreach_non_finite_check_and_unscale_
        )

    @classmethod
    def _override_unary_func(cls):
        """Override unary functions of torch."""
        for func_name in cls.torch_unary_funcs:
            base, name = cls._get_func_base_and_name(func_name)
            setattr(base, name, cls._get_wrapper_for_torch_unary(getattr(base, name)))

    @staticmethod
    def _cast_to_scalingtensor(self, qtype, meta=None, sync=False):
        """Cast pytorch native tensor to ScalingTensor.

        Support below casts:
        torch.float32 | torch.float16 | torch.bfloat16 ->
            Dtypes.kfloat8_e4m3 | Dtypes.kfloat8_e5m2 | Dtypes.kfloat16 | Dtypes.kbfloat16 | Dtypes.kfloat32

        Args:
            self (torch.Tensor): tensor to cast.
            qtype (Qtypes.QType): type to cast.
            meta (Scaling8Meta): meta data of ScalingTensor.
            sync (bool): whether to synchronize the cast operation.

        Return:
            ScalingTensor: a new ScalingTensor object.
        """
        self = self.contiguous()
        if meta is None:
            # default window size: 1
            meta = ScalingMeta(qtype)
        if Dtypes.is_fp8_qtype(qtype):
            return ScalingTensor(TypeCast.cast_to_fp8(self, meta, sync=sync), meta=meta)
        return ScalingTensor(TypeCast.cast_to_fp16(self, meta, sync=sync), meta=meta)

    @staticmethod
    def _fused_cast_transpose_to_scalingtensors(self, qtype, meta=None, sync=False):
        """Fused cast and transpose pytorch native tensor to ScalingTensors.

        Args:
            self (torch.Tensor): input tensor.
            qtype (QType): qtype to cast.
            meta (ScalingMeta): scaling meta.
            sync (bool): whether to synchronize the cast operation.

        Return:
            ScalingTensor, ScalingTensor: casted and transposed scaling tensors.
        """
        self = self.contiguous()
        if meta is None:
            # default window size: 1
            meta = ScalingMeta(qtype)
        if Dtypes.is_fp8_qtype(qtype):
            cast, t = TypeCast.cast_to_fp8(self, meta, sync=sync, fuse_transpose=True)
            return ScalingTensor(cast.contiguous(), meta=meta), ScalingTensor(t.contiguous(), meta=meta)
        raise TypeError(f'Unsupported Cast: {self.dtype} -> {qtype}')

    @staticmethod
    def _get_func_base_and_name(func_name):
        """Get module and function name from complete function name.

        Args:
            func_name (str): complete function name, such as 'torch.zeros_like'.

        Return:
            tuple: module and function name, such as (torch, 'zeros_like').
        """
        sp = func_name.split('.')
        if len(sp) < 2:
            raise ValueError(f'Invalid function name: {func_name}')
        base = globals()[sp[0]]
        for e in sp[1:-1]:
            base = getattr(base, e)
        return base, sp[-1]

    @staticmethod
    def _get_wrapper_for_scalingtensor(old_fn, scaling_fn):
        """Get wrapper for functions that can process torch.Tensor and ScalingTensor.

        Args:
            old_fn (function): original function.
            scaling_fn (function): function that can process ScalingTensor.

        Return:
            function: new function that can process torch.Tensor and ScalingTensor.
        """
        @torch.jit.ignore
        def fn(input, *args, **kwargs):
            if isinstance(input, ScalingTensor):
                return scaling_fn(input, *args, **kwargs)
            return old_fn(input, *args, **kwargs)

        return fn

    @classmethod
    def _get_wrapper_for_torch_unary(cls, old_fn):
        """Get wrapper for torch unary function.

        Args:
            old_fn (function): original function.

        Return:
            function: new function that can process torch.Tensor and ScalingTensor.
        """
        def scaling_fn(input, *args, **kwargs):
            return old_fn(input.value, *args, **kwargs)

        return cls._get_wrapper_for_scalingtensor(old_fn, scaling_fn)

    @classmethod
    def _get_wrapper_for_grad_check_and_unscale(cls, old_fn):
        """Get wrapper for torch._amp_foreach_non_finite_check_and_unscale_.

        Args:
            old_fn (function): original function.

        Return:
            function: new function that can process torch.Tensor and ScalingTensor.
        """
        @torch.no_grad()
        def new_fn(grads, found_inf, inv_scale):
            """A wrapper of torch._amp_foreach_non_finite_check_and_unscale_ for ScalingTensor.

            This function is a wrapper around torch._foreach_non_finite_check_and_unscale_ that
            checks if a non-finite value exists in the grads.
            Meanwhile, all gradients are multiplied by inv_scale (grad *= inv_scale).

            Args:
                grads (list): list of grads
                found_inf (Tensor): a single element that is set to 1 if a non-finite is found
                inv_scale (Tensor): a single element that is set to 1 / scale

            Returns:
                None
            """
            cpu_torch_grads = []
            cuda_torch_grads = []
            scaling_grads_scale_inv, scaling_grads_amax0 = [], []
            for grad in grads:
                if isinstance(grad, ScalingTensor):
                    scaling_grads_scale_inv.append(grad.meta.scale_inv)
                    scaling_grads_amax0.append(grad.meta.amax[0])
                elif grad.is_cuda:
                    cuda_torch_grads.append(grad)
                else:
                    cpu_torch_grads.append(grad)

            # torch.Tensor on GPU
            if cuda_torch_grads:
                old_fn(cuda_torch_grads, found_inf, inv_scale)

            # torch.Tensor on CPU
            for grad in cpu_torch_grads:
                grad.mul_(inv_scale)
                if not torch.isfinite(grad).all():
                    found_inf.fill_(True)

            # ScalingTensor
            if scaling_grads_scale_inv:
                # torch._amp_foreach_non_finite_check_and_unscale_
                old_fn(scaling_grads_scale_inv, found_inf, inv_scale)

                device = found_inf.device
                if device not in cls.one_scales:
                    cls.one_scales[device] = torch.ones(1, device=device)
                old_fn(scaling_grads_amax0, found_inf, cls.one_scales[device])

        return new_fn


TorchOverider.override()
