---
id: usage
---

# Use MS-AMP

## Basic usage

Enabling MS-AMP is very simple when traning model w/o any distributed parallel technologies, you only need to add one line of code `msamp.initialize(model, optimizer, opt_level)` after defining model and optimizer.

Example:

```python
import msamp

# Declare model and optimizer as usual, with default (FP32) precision
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Allow MS-AMP to perform casts as required by the opt_level
model, optimizer = msamp.initialize(model, optimizer, opt_level="O2")
...
```

## Usage in DeepSpeed

MS-AMP supports FP8 for distributed parallel training and has the capability of integrating with advanced distributed traning frameworks. We have integrated MS-AMP with several popular distributed training frameworks such as DeepSpeed, Megatron-DeepSpeed and Megatron-LM to demonstrate this capability.

For enabling MS-AMP in DeepSpeed, add one line of code `from msamp import deepspeed` at the beginging and a "msamp" section in DeepSpeed config file:

```json
"msamp": {
  "enabled": true,
  "opt_level": "O1|O2|O3",
  "use_te": false
}
```

"O3" is designed for FP8 in ZeRO optimizer, so please make sure ZeRO is enabled when using "O3".
"use_te" is designed for Transformer Engine, if you have already used Transformer Engine in your model, don't forget to set "use_te" to true.

## Usage in FSDP

When using FSDP, enabling MS-AMP is very easy, just use `FsdpReplacer.replace` and `FP8FullyShardedDataParallel` to initialize model and `FSDPAdam` to initialize optimizer.

Example:

```python
# Your model
model = ...

# Initialize model
from msamp.fsdp import FsdpReplacer
from msamp.fsdp import FP8FullyShardedDataParallel
my_auto_wrap_policy = ...
model = FsdpReplacer.replace(model)
model = FP8FullyShardedDataParallel(model, use_orig_params=True, auto_wrap_policy=my_auto_wrap_policy)

# Initialize optimizer
from msamp.optim import FSDPAdam
optimizer = FSDPAdam(model.parameters(), lr=3e-04)
```

Please note that currently we only support `use_orig_params=True`.

## Usage in Megatron-DeepSpeed and Megatron-LM

For integrating MS-AMP with Megatron-DeepSpeed and Megatron-LM, you need to make some code changes. We provide a patch as a reference for the integration. Here is the instruction of integrating MS-AMP with Megatron-DeepSpeed/Megatron-LM and how to run [gpt-3](https://github.com/Azure/MS-AMP-Examples/tree/main/gpt3) with MS-AMP.

Runnable, simple examples demonstrating good practices can be found [here](https://azure.github.io//MS-AMP/docs/getting-started/run-msamp).
For more comprehensive examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).
