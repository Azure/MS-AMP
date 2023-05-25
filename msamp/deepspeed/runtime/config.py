# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Config with MS-AMP support."""

from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS

MSAMP_ADAM_OPTIMIZER = 'msamp_adam'
MSAMP_ADAMW_OPTIMIZER = 'msamp_adamw'

DEEPSPEED_OPTIMIZERS.extend([
    MSAMP_ADAM_OPTIMIZER,
    MSAMP_ADAMW_OPTIMIZER,
])
