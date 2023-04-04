# MS-AMP: Microsoft Automatic Mixed Precision

MS-AMP is an automatic mixed precision package for deep learning developed by Microsoft. 

Features:
- Support the new FP8 feature that is introduced by Nvidia H100.
- Speeds up math-intensive operations, such as linear layers, by using Tensor Cores.
- Speeds up memory-limited operations by accessing one byte compared to half or single-precision.  
- Reduces memory requirements for training models, enabling larger models or larger minibatches. 
- Speeds up communication for distributed model by transmitting lower precision gradients. 

## Get started

### Prerequisites
- Latest version of Linux, you're highly encouraged to use Ubuntu 18.04 or later.
- H100 accelerator and compatible GPU drivers should be installed correctly. Driver version can be checked by running `nvidia-smi`. 
- Python version 3.7 or later (which can be checked by running `python3 --version`).
- Pip version 18.0 or later (which can be checked by running `python3 -m pip --version`).
- CUDA version 11 or later (which can be checked by running `nvcc --version`).
- PyTorch version 1.13 or later (which can be checked by running `python -c "import torch; print(torch.__version__)"`).

### Install nccl to support fp8
You need to install specific nccl to supports fp8. You can install it from source.
```bash
git clone https://github.com/yzygitzh/nccl.git
cd nccl
git checkout ziyyang/fp8-support
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
sudo make install
```

### Install MS-AMP
You can clone the source code from github and build it.
```bash
git clone https://github.com/Azure/MS-AMP.git
cd MS-AMP
python -m pip install .
make postinstall
```

After that, you can verify the installation by running:
```bash
python3 -c "import msamp; print(msamp.__version__)"
```

### Usage
Enabling MS-AMP is very simple when traning model on 1 GPU, you only need to add one line of code "msamp.initialize(model, optimizer, opt_level)" after defining model and optimizer.  
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

For distributed training job, you need to add optimizer.all_reduce_grads(model) after backward to reduce gradients in process group.  
Example:
```python
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss(output, target)
    loss.backward()
    optimizer.all_reduce_grads(model)
    optimizer.step()
```
A runnable, comprehensive Mnist example demonstrating good practices can be found [here](https://github.com/Azure/MS-AMP/tree/main/examples).  

Recognized optimizers are torch.optim.Adam and torch.optim.AdamW.  
Recognized opt_levels are "O1" and "O2". Try both, and see what gives the best speedup and accuracy for your model.
- O1: We found that directly transitioning weight gradients from FP32 to FP8 in the Transformer Engine leads to a decrease in accuracy. However, this issue is resolved in O1 through the implementation of FP8 for weight gradients and AllReduce communication. This optimization also has the added benefits of saving GPU memory and reducing communication bandwidth.

- O2: From O1 to O2, our main focus is on enabling the use of low-bit data formats for auxiliary tensors in the Adam/AdamW optimizer without any loss in accuracy. Specifically, we are able to maintain accuracy by representing the first-order optimizer state in FP8 and the second-order state in FP16. This optimization has the potential to save up to 62.5% of GPU memory for the optimizer when the model size is particularly large.  

Here here details of different MS-AMP optimization levels:  
| Optimization Levvel | Computation(GEMM) | Comm. | Weight | Weight Gradient | Optimizer States |
| ------------------- | -----------       | ----- | ------ | --------------- | ---------------- |
| Nvidia TE           | FP8               | FP32  | FP16   | FP8             | FP32+FP32 
| MS-AMP O1           | FP8               | FP8   | FP16   | FP8             | FP32+FP32        |
| MS-AMP O2           | FP8               | FP8   | FP16   | FP8             | FP8+FP16         | 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

