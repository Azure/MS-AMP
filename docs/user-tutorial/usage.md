---
id: usage
---

# Use MS-AMP

Enabling MS-AMP is very simple when traning model on single GPU, you only need to add one line of code `msamp.initialize(model, optimizer, opt_level)` after defining model and optimizer.

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

For applying MS-AMP to DeepSpeed ZeRO, add a "msamp" section in deepspeed config file:

```json
"msamp": {
  "enabled": true,
  "opt_level": "O3"
}
```

Runnable, comprehensive examples demonstrating good practices can be found [here](https://azure.github.io//MS-AMP/docs/getting-started/run-msamp).
For more examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).
