---
id: usage
---

# Use MS-AMP

## Basic usage

Enabling MS-AMP is very simple when traning model w/ or w/o data parallelism on a single node, you only need to add one line of code `msamp.initialize(model, optimizer, opt_level)` after defining model and optimizer.

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

## Usage in distributed parallel training

MS-AMP supports FP8 for distributed parallel training and has the capability of integrating with advanced distributed traning frameworks. We have integrated MS-AMP with several popular distributed training frameworks such as DeepSpeed, Megatron-DeepSpeed and Megatron-LM to demonstrate this capability.

For enabling MS-AMP when using ZeRO in DeepSpeed, add one line of code `import msamp` and a "msamp" section in DeepSpeed config file:

```json
"msamp": {
  "enabled": true,
  "opt_level": "O3"
}
```

For applying MS-AMP to Megatron-DeepSpeed and Megatron-LM, you need to do very little code change for applying it. Here is the instruction of applying MS-AMP for running [gpt-3](https://github.com/Azure/MS-AMP-Examples/tree/main/gpt3) in both Megatron-DeepSpeed and Megatron-LM.

Runnable, simple examples demonstrating good practices can be found [here](https://azure.github.io//MS-AMP/docs/getting-started/run-msamp).
For more comprehensive examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).
