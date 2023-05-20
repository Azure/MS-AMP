from deepspeed.runtime.config import *

MSAMP_ADAM_OPTIMIZER = 'mfp8_adam'
MSAMP_ADAMW_OPTIMIZER = 'mfp8_adamw'

DEEPSPEED_OPTIMIZERS.extend([
    MSAMP_ADAM_OPTIMIZER,
    MSAMP_ADAMW_OPTIMIZER,
])
