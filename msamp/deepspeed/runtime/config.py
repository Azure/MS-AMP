# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extend the new optimizer types."""

from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS

FP8_ADAM_OPTIMIZER = 'fp8_adam'
FP8_ADAMW_OPTIMIZER = 'fp8_adamw'

DEEPSPEED_OPTIMIZERS.extend([
    FP8_ADAM_OPTIMIZER,
    FP8_ADAMW_OPTIMIZER,
])
