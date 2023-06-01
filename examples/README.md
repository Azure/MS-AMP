This directory contains examples illustrating how to enable MS-AMP in training script.

# MNIST

## 1. Run mnist using single GPU

```bash
python mnist.py --enable-msamp --opt-level=O2
```

## 2. Run mnist using multi GPUS in single node

```bash
python -m torch.distributed.launch --nproc_per_node=$GPUS mnist_ddp.py --enable-msamp --opt-level=O2
```
