# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Utils with MS-AMP support."""

from deepspeed.runtime.utils import CheckOverflow

from msamp.common.tensor import ScalingTensor


class MSAMPCheckOverflow(CheckOverflow):
    """CheckOverflow with MS-AMP support."""
    @staticmethod
    def _has_inf_or_nan(x, i):
        """Check if the input tensor has inf or nan values.

        Args:
            x (torch.Tensor): Input tensor.
            i (int): Index of the tensor.

        Returns:
            bool: True if the input tensor has inf or nan values.
        """
        if isinstance(x, ScalingTensor):
            return x.has_inf_or_nan()
        return CheckOverflow._has_inf_or_nan(x, i)
