# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP ScalingMeta."""

import copy
import torch

from msamp.common.dtype import Floating, Dtypes


class ScalingMeta:
    """The meta data for scaling tensor."""
    in_time_scaling: bool = True

    def __init__(self, qtype, scale=None, scale_inv=None, amax=None, window_size=1, group=None):
        """Constructor.

        Args:
            qtype (Dtypes.QType): Type of the scaling tensor.
            scale (torch.Tensor, optional): Scaling tensor, defaults to None.
            scale_inv (torch.Tensor, optional): The reciprocal of scaling tensor, defaults to None.
            amax (torch.Tensor, optional): Absolute maximum tensor, defaults to None.
            window_size (int, optional): Window size, defaults to 1.
            group (torch.distributed.ProcessGroup, optional): Distributed group, defaults to None.
        """
        self.qtype = qtype
        self.scale = scale if scale is not None else torch.ones((), device='cuda')
        self.scale_inv = scale_inv if scale_inv is not None else torch.ones((), device='cuda')
        self.amax = amax if amax is not None else torch.zeros((window_size, ), device='cuda')
        self.amax_counter = torch.zeros((), dtype=torch.int32)
        self.window_size = window_size
        self.group = group
        # lock flag to avoid the reference of the meta changed.
        self.locked = False

    @staticmethod
    @torch.jit.script
    def compute_scaling_factor(amax, scale, fp_max: float, margin: int):
        """A function to compute scaling factor.

        Args:
            amax (torch.Tensor): Absolute maximum tensor.
            scale (torch.Tensor): Scale tensor.
            fp_max (float): The maximum value of float point.
            margin (int): Margin value.

        Returns:
            return new scaling tensor.
        """
        exp = torch.floor(torch.log2(fp_max / amax)) - margin
        sf = torch.round(torch.pow(2, torch.abs(exp)))
        sf = torch.where(amax > 0.0, sf, scale)
        sf = torch.where(torch.isfinite(amax), sf, scale)
        sf = torch.where(exp < 0, 1 / sf, sf)
        return sf

    def is_warmup(self):
        """Check if in warm-up stage.

        Returns:
            bool: if amax counter less than windows size return True, otherwise return False.
        """
        return self.amax_counter < self.window_size

    def is_in_time_scaling(self):
        """Check if in time scaling.

        Returns:
            bool: if windows size equals 1 or in warm up stage return True, otherwise return False.
        """
        return ScalingMeta.in_time_scaling and (self.window_size == 1 or self.is_warmup())

    @staticmethod
    def in_time_scaling_context(enabled):
        """A context manager to set in_time_scaling flag.

        Args:
            bool: in_time_scaling flag.

        Returns:
            InTimeScalingContext: A context manager to set in_time_scaling flag.
        """
        class InTimeScalingContext:
            def __init__(self, enabled):
                self.enabled = enabled
                self.in_time_scaling = ScalingMeta.in_time_scaling

            def __enter__(self):
                self.in_time_scaling = ScalingMeta.in_time_scaling
                ScalingMeta.in_time_scaling = self.enabled

            def __exit__(self, exc_type, exc_val, exc_tb):
                ScalingMeta.in_time_scaling = self.in_time_scaling

        return InTimeScalingContext(enabled=enabled)

    def reset_scaling_factor(self, qtype=None):
        """Reset scaling factor.

        Args:
            qtype (Dtypes.QType, optional): float point type, defaults to None.
        """
        if qtype is None:
            qtype = self.qtype

        if qtype in [Dtypes.kfloat32, Dtypes.kbfloat16]:
            self.scale.fill_(1)
        else:
            fp_max = Floating.qfp_max[qtype]
            sf = ScalingMeta.compute_scaling_factor(self.amax[0], self.scale, fp_max, 0)
            self.scale.copy_(sf)

    def copy_(self, src):
        """Copies the members from src into self and returns self.

        Args:
            src (ScalingMeta): Soruce ScalingMeta instance.
        """
        self.qtype = src.qtype
        self.scale.copy_(src.scale)
        self.scale_inv.copy_(src.scale_inv)
        self.amax.copy_(src.amax)
        self.amax_counter.copy_(src.amax_counter)
        self.window_size = src.window_size

    def clone(self):
        """Returns a copy of this object."""
        group = self.group
        self.group = None
        obj = copy.deepcopy(self)
        self.group = obj.group = group
        return obj

    def cuda(self):
        """Returns a copy of this object in CUDA memory."""
        self.scale = self.scale.cuda()
        self.scale_inv = self.scale_inv.cuda()
        self.amax = self.amax.cuda()
        return self

    @property
    def is_cuda(self):
        """Check whether the tensor is stored on nvidia GPU.

        Return:
            bool: if the Tensor is stored on the GPU return True , otherwise return False.
        """
        return self.scale.is_cuda

    def __repr__(self):
        """Return a printable representation of this object.

        Return:
            string: Printable representation.
        """
        return f'ScalingMeta(qtype={self.qtype}, '\
               f'scale={self.scale.data:g}, scale_inv={self.scale_inv.data:g}, '\
               f'amax={self.amax.max():g}, window_size={self.window_size})'
