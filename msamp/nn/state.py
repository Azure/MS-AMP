# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP state module."""

from collections import OrderedDict
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from msamp.common.dtype import Dtypes


class ModelState:
    """Model state class to hold some states of the model."""
    def __init__(self):
        """Constructor."""
        self._ready_to_scale_tensor = False
        # dict[str, dict[str, tensor]], store the flattened scaling metas in all FP8Linear modules.
        # key in input/wgrad/output, value is a dict whose key is scales/amaxs/amax_counters
        # and value is flattened tensors.
        self._flattened_scaling_metas = None
        # OrderedDict[str, dict[str, ScalingMeta]], store the local scaling metas in all FP8Linear modules.
        # key is module name, value is scaling_metas in FP8Linear module.
        self._local_scaling_metas = OrderedDict()

    @property
    def ready_to_scale_tensor(self):
        """Decoration function to access _ready_to_tensor_scale variable."""
        return self.ready_to_tensor_scale

    @ready_to_scale_tensor.setter
    def ready_to_scale_tensor(self, value):
        """Set the value of _ready_to_tensor_scale variable.

        Args:
            value (bool): Value to set.
        """
        self._ready_to_scale_tensor = value

    @property
    def flattened_scaling_metas(self):
        """Decoration function to access _flattened_scaling_metas variable."""
        return self._flattened_scaling_metas

    @flattened_scaling_metas.setter
    def flattened_scaling_metas(self, value):
        self._flattened_scaling_metas = value

    @staticmethod
    def _check_in_mem(tensor1, tensor2):
        """Check if tensor1 is in tensor2's memory. Raise RuntimeError if not.

        Args:
            tensor1 (torch.Tensor): Tensor to check.
            tensor2 (torch.Tensor): Tensor to check.
        """
        data_ptr1 = tensor1.data_ptr()
        data_ptr2 = tensor2.data_ptr()
        size1 = Dtypes.dtype_to_size[tensor1.dtype] * tensor1.numel()
        size2 = Dtypes.dtype_to_size[tensor2.dtype] * tensor2.numel()

        if not (data_ptr2 <= data_ptr1 and data_ptr1 + size1 <= data_ptr2 + size2):
            raise RuntimeError(f'Memory ({data_ptr1}, {data_ptr1 + size1}) not in ({data_ptr2}, {data_ptr2 + size2})')

    @staticmethod
    def _flat_tensors(tensors):
        """Flatten tensors into a single tensor.

        Args:
            tensors (Iterable[torch.Tensor]): dense tensors to flatten..

        Returns:
            torch.Tensor: flattened tensor.
        """
        flat = _flatten_dense_tensors(tensors)
        for p, q in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
            p.data = q.data
        return flat

    @classmethod
    def _flatten_scaling_metas(cls, metas):
        """Flatten scaling metas into a single dict.

        Args:
            metas (Iterable[ScalingMeta]): scaling metas to flatten.

        Return:
            dict: flattened scaling metas.
        """
        n = len(metas)
        window_size = metas[0].window_size
        if not all(map(lambda e: e.window_size == window_size, metas)):
            raise RuntimeError('All metas must have the same window_size.')

        qtype = metas[0].qtype

        if not all(map(lambda e: e.qtype == qtype, metas)):
            raise RuntimeError('All metas must have the same qtype.')

        # scale and amax
        scales = cls._flat_tensors([m.scale for m in metas])
        amaxs = cls._flat_tensors([m.amax for m in metas])
        amax_counters = cls._flat_tensors([m.amax_counter for m in metas])
        # scales: (n,)
        # amaxs: (n, window_size)
        for meta in metas:
            meta.locked = True
        return dict(
            qtype=qtype,
            scales=scales,
            amaxs=amaxs.view(n, window_size),
            amax_counters=amax_counters,
        )

    def check_metas_in_flat(self, scaling_metas):
        """Check if scaling meta is in the flattened memory. Raise RuntimeError if not.

        Args:
            scaling_metas (dict[str, ScalingMeta]): scaling metas to check.
        """
        if self._flattened_scaling_metas is None:
            return
        for k, v in scaling_metas.items():
            metas = self._flattened_scaling_metas[k]
            ModelState._check_in_mem(v.scale, metas['scales'])
            ModelState._check_in_mem(v.amax, metas['amaxs'])
            ModelState._check_in_mem(v.amax_counter, metas['amax_counters'])

    def register_scaling_metas(self, model):
        """Register scaling metas of the model to model state.

        Args:
            model (torch.nn.Module): model to register.
        """
        for k, m in model.named_modules():
            if hasattr(m, 'scaling_metas'):
                if k in self._local_scaling_metas:
                    i = 2
                    while True:
                        new_k = f'{k}_{i}'
                        if new_k not in self._local_scaling_metas:
                            break
                        i += 1
                    k = new_k
                self._local_scaling_metas[k] = m.scaling_metas

        # metas is list of dict[str, ScalingMeta]
        metas = list(self._local_scaling_metas.values())
        if len(metas) == 0:
            return

        # keys: ['input', 'wgrad', 'ograd']
        keys = metas[0].keys()
        values = dict()
        for key in keys:
            values[key] = ModelState._flatten_scaling_metas([m[key] for m in metas])

        self._flattened_scaling_metas = values

        # check metas memory
        for meta in metas:
            self.check_metas_in_flat(meta)


model_state = ModelState()
