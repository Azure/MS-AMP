# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed ZeRO Utils with MS-AMP support."""

from deepspeed.runtime.zero.utils import ZERO_SUPPORTED_OPTIMIZERS
import msamp.optim

OPT_NAMES = ['LBAdamW', 'LBAdam', 'LBLion']
for name in OPT_NAMES:
    ZERO_SUPPORTED_OPTIMIZERS.append(getattr(msamp.optim, name))
