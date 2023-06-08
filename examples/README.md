This directory contains examples illustrating how to enable MS-AMP in training script.

# MNIST

## 1. Run mnist using single GPU

```bash
python mnist.py --enable-msamp --opt-level=O2
```

## 2. Run mnist using multi GPUS in single node

```bash
torchrun --nproc_per_node=$GPUS mnist_ddp.py --enable-msamp --opt-level=O2
```

# CIFAR10

## 1. Run cifar10 using deepspeed

```bash
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json
```

## 2. Run cifar10 using deepspeed with msamp enabled

```bash
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config_msamp.json
```

## 3. Run cifar10 using deepspeed-ZeRO with msamp enabled

```
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config_zero_msamp.json
```
