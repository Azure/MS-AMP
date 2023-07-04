# MS-AMP: Microsoft Automatic Mixed Precision

MS-AMP is an automatic mixed precision package for deep learning developed by Microsoft.

Features:

- Support O1 optimization: Apply FP8 to weights and weight gradients and support FP8 in communication.
- Support O2 optimization: Support FP8 for two optimizers(Adam and AdamW).
- Provide three training examples using FP8: Swin-Transformer, DeiT and RoBERTa.

MS-AMP has the following benefit comparing with Transformer Engine:

- Support the new FP8 feature that is introduced by latest accelerators (e.g. H100).
- Speed up math-intensive operations, such as linear layers, by using Tensor Cores.
- Speed up memory-limited operations by accessing one byte compared to half or single-precision.
- Reduce memory requirements for training models, enabling larger models or larger minibatches.
- Speed up communication for distributed model by transmitting lower precision gradients.

## Get started

### Prerequisites

- Latest version of Linux, you're highly encouraged to use Ubuntu 18.04 or later.
- Nvidia GPU(e.g. V100/A100/H100) and compatible drivers should be installed correctly.
  Driver version can be checked by running `nvidia-smi`.
- Python version 3.7 or later (which can be checked by running `python3 --version`).
- Pip version 18.0 or later (which can be checked by running `python3 -m pip --version`).
- CUDA version 11 or later (which can be checked by running `nvcc --version`).
- PyTorch version 1.13 or later (which can be checked by running `python -c "import torch; print(torch.__version__)"`).

We strongly recommend using [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). For example, to start PyTorch 1.13 container, run the following command:

```
sudo docker run -it -d --name=msamp --privileged --net=host --ipc=host --gpus=all nvcr.io/nvidia/pytorch:22.09-py3 bash
sudo docker exec -it msamp bash
```

### Install MS-AMP

You can clone the source from GitHub.

```bash
git clone https://github.com/Azure/MS-AMP.git
cd MS-AMP
git submodule update --init --recursive
```

If you want to train model with multiple GPU, you need to install specific nccl to support FP8.

```bash
cd third_party/nccl

# V100
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
# A100
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
# H100
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

apt-get update
apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build
dpkg -i build/pkg/deb/libnccl2_*.deb

cd -
```

Then, you can install MS-AMP from source.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install .
make postinstall
```

Before using MS-AMP, you need to preload msampfp8 library and it's depdencies:

```bash
NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so # Change as needed
export LD_PRELOAD="/usr/local/lib/libmsampfp8.so:${NCCL_LIBRARY}:${LD_PRELOAD}"
```

After that, you can verify the installation by running:

```bash
python3 -c "import msamp; print(msamp.__version__)"
```

### Usage

Enabling MS-AMP is very simple when traning model on single GPU, you only need to add one line of code `msamp.initialize(model, optimizer, opt_level)` after defining model and optimizer.

Example:

```python
import msamp

# Declare model and optimizer as usual, with default (FP32) precision
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Allow MS-AMP to perform casts as required by the opt_level
model, optimizer = msamp.initialize(model, optimizer, opt_level="O1")
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

A runnable, comprehensive MNIST example demonstrating good practices can be found [here](./examples). For more examples, please go to [MS-AMP-Examples](https://github.com/Azure/MS-AMP-Examples).

### Optimization Level

Currently MS-AMP supports two optimization levels: O1 and O2. Try both, and see what gives the best speedup and accuracy for your model.

- O1: We found that directly transitioning weight gradients from FP32 to FP8 in the Transformer Engine leads to a decrease in accuracy. However, this issue is resolved in O1 through the implementation of FP8 for weight gradients and AllReduce communication. This optimization also has the added benefits of saving GPU memory and reducing communication bandwidth.

- O2: From O1 to O2, our main focus is on enabling the use of low-bit data formats for auxiliary tensors in the Adam/AdamW optimizer without any loss in accuracy. Specifically, we are able to maintain accuracy by representing the first-order optimizer state in FP8 and the second-order state in FP16. This optimization has the potential to save up to 62.5% of GPU memory for the optimizer when the model size is particularly large.

Here are details of different MS-AMP optimization levels:
| Optimization Level  | Computation(GEMM) | Comm  | Weight | Weight Gradient | Optimizer States |
| ------------------- | -----------       | ----- | ------ | --------------- | ---------------- |
| FP16 AMP            | FP16              | FP32  | FP32   | FP32            | FP32+FP32        |
| Nvidia TE           | FP8               | FP32  | FP32   | FP32            | FP32+FP32        |
| MS-AMP O1           | FP8               | FP8   | FP16   | FP8             | FP32+FP32        |
| MS-AMP O2           | FP8               | FP8   | FP16   | FP8             | FP8+FP16         |

## Performance

### Accuracy: no loss of accuracy

We evaluated the training loss and validation performance of three typical models, Swin-Transformer, DeiT and RoBERTa, using both MS-AMP O2 and FP16 AMP. Our observations showed that the models trained with MS-AMP O2 mode achieved comparable performance to those trained using FP16 AMP. This demonstrates the effectiveness of the Mixed FP8 O2 mode in MS-AMP.

Here are the results for Swin-T, DeiT-S and RoBERTa-B:

![image](./docs/assets/performance.png)

### Memory

MS-AMP preserves 32-bit accuracy while using only a fraction of the memory footprint on a range of tasks, including the DeiT model and Swin Transformer for ImageNet classification. For example, comparing with FP16 AMP, MS-AMP with O2 mode can achieve 44% memory saving for Swin-1.0B and 26% memory saving for ViT-1.2B. The proportion of memory saved will be more obvious for larger models.

Here are the results for Swin-1.0B and ViT-1.2B.

![Image](./docs/assets/gpu-memory.png)

For detailed setting and results, please go to [MS-AMP-Example](https://github.com/Azure/MS-AMP-Examples).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [CLA](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
