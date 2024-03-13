# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP floating module."""

import torch
import numpy as np

from msamp.common.utils import Device, GPUType
from msamp.common.dtype import Dtypes


class Floating:
    """Computes and provides the maximum floating point value."""
    fp_maxs: dict = {}
    qfp_max: dict = {}

    @staticmethod
    def _get_fp_max(exp, man, inf_existed=True):
        """Computes the maximum floating point value by exponent and mantissa.

        Args:
            exp (int): Number of exponent bits.
            man (int): Number of mantissa bits.
            inf_existed(bool): Whether represent infinite when exponent bits are all one.

        Return:
            value (float): The float point value.
        """
        if exp == 4 and man == 3 and Device.get_gpu_type() == GPUType.AMD:
            return 240.0
        e_bias = np.power(2., exp - 1) - 1
        if inf_existed:
            max_value_exp = (np.power(2.0, exp) - 1) - e_bias - man - 1
            max_value_man = np.power(2.0, man + 1) - 1
        else:
            max_value_exp = (np.power(2.0, exp) - 1) - e_bias - man
            max_value_man = np.power(2.0, man + 1) - 2
        return float(np.power(2.0, max_value_exp) * max_value_man)


Floating.fp_maxs = {
    torch.fp8e4m3: Floating._get_fp_max(exp=4, man=3, inf_existed=False),    # type: ignore
    torch.fp8e5m2: Floating._get_fp_max(exp=5, man=2),    # type: ignore
    torch.float16: Floating._get_fp_max(exp=5, man=10),
    torch.bfloat16: Floating._get_fp_max(exp=8, man=7),
    torch.float32: Floating._get_fp_max(exp=8, man=23),
}

Floating.qfp_max = {
    Dtypes.kfloat8_e4m3: Floating._get_fp_max(exp=4, man=3, inf_existed=False),
    Dtypes.kfloat8_e5m2: Floating._get_fp_max(exp=5, man=2),
    Dtypes.kfloat16: Floating._get_fp_max(exp=5, man=10),    # E5M10
    Dtypes.kbfloat16: Floating._get_fp_max(exp=8, man=7),    # E8M7
    Dtypes.kfloat32: Floating._get_fp_max(exp=8, man=23),    # E8M23
}
