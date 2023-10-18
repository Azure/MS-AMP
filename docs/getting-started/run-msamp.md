---
id: run-msamp
---

# Run examples
After installing MS-AMP, you can run several simple examples using MS-AMP. Please note that before running these commands, you need to change work directory to [examples](https://github.com/Azure/MS-AMP/tree/main/examples).

## MNIST
### 1. Run mnist using single GPU

```bash
python mnist.py --enable-msamp --opt-level=O2
```

### 2. Run mnist using multi GPUS in single node

```bash
torchrun --nproc_per_node=$GPUS mnist_ddp.py --enable-msamp --opt-level=O2
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

For more comprehensive examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).