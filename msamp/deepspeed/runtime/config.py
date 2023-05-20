# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Config with MS-AMP support."""

# flake8: noqa: F403
from deepspeed.runtime.config import *

MSAMP_ADAM_OPTIMIZER = 'msamp_adam'
MSAMP_ADAMW_OPTIMIZER = 'msamp_adamw'

# flake8: noqa: F405
DEEPSPEED_OPTIMIZERS.extend([
    MSAMP_ADAM_OPTIMIZER,
    MSAMP_ADAMW_OPTIMIZER,
])
