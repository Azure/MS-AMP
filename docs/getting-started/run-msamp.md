---
id: run-msamp
---

# Run Examples

After installing MS-AMP, you can run several simple examples using MS-AMP. Please note that before running these commands, you need to change work directory to [examples](https://github.com/Azure/MS-AMP/tree/main/examples).

## MNIST

### 1. Run mnist using single GPU

```bash
python mnist.py --enable-msamp --opt-level=O2
```

### 2. Run mnist using multi GPUs in single node

```bash
torchrun --nproc_per_node=8 mnist_ddp.py --enable-msamp --opt-level=O2
```

### 3. Run mnist using FSDP

```bash
python mnist_fsdp.py --msamp
```

## CIFAR10

### 1. Run cifar10 using deepspeed

```bash
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json
```

### 2. Run cifar10 using deepspeed with msamp enabled

```bash
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config_msamp.json
```

### 3. Run cifar10 using deepspeed-ZeRO with msamp enabled

```bash
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config_zero_msamp.json
```

### 4. Run cifar10 using deepspeed-ZeRO + TE with msamp enabled

```bash
deepspeed cifar10_deepspeed_te.py --deepspeed --deepspeed_config ds_config_zero_te_msamp.json
```

For more comprehensive examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).
