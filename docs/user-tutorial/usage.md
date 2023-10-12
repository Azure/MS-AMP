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

For distributed training job, you need to add `optimizer.all_reduce_grads(model)` after backward to reduce gradients in process group.

Example:

```python
scaler = torch.cuda.amp.GradScaler()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss(output, target)
    scaler.scale(loss).backward()
    optimizer.all_reduce_grads(model)
    scaler.step(optimizer)
```

For applying MS-AMP to DeepSpeed ZeRO, add a "msamp" section in deepspeed config file:

```json
"msamp": {
  "enabled": true,
  "opt_level": "O3"
}
```

Runnable, comprehensive examples demonstrating good practices can be found [here](./examples).
For more examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).
