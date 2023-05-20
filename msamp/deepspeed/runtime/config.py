from deepspeed.runtime.config import *

MSAMP_ADAM_OPTIMIZER = 'msamp_adam'
MSAMP_ADAMW_OPTIMIZER = 'msamp_adamw'

DEEPSPEED_OPTIMIZERS.extend([
    MSAMP_ADAM_OPTIMIZER,
    MSAMP_ADAMW_OPTIMIZER,
])
