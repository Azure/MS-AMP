---
id: optimization-level
---

Currently MS-AMP supports three optimization levels: O1 and O2 and O3. The three levels gradually incorporate 8-bit collective communcation, optimizer and distributed parallel training in an incremental manner. Users can directly set O1/O2 using `msamp.initialize` and set O3 in config file when using DeepSpeed.

- O1: We found that directly transitioning weight gradients from FP32 to FP8 in the Transformer Engine leads to a decrease in accuracy. However, this issue is resolved in O1 through the implementation of FP8 for weight gradients and AllReduce communication. This optimization also has the added benefits of saving GPU memory and reducing communication bandwidth.

- O2: From O1 to O2, our main focus is on enabling the use of low-bit data formats for auxiliary tensors in the Adam/AdamW optimizer without any loss in accuracy. Specifically, we are able to maintain accuracy by representing the first-order optimizer state in FP8 and the second-order state in FP16. This optimization has the potential to save up to 62.5% of GPU memory for the optimizer when the model size is particularly large.

- O3: This optimization level is specifically designed for FP8 support in distributed parallel training for large scale models. These frequently-used strategies include data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism and ZeRO optimizer. ZeRO separates model weights into regular weights and master weights, with the former used for network forward/backward on each GPU, and the latter used for model updating in the optimizer. This separation allows us to use 8-bit data precision for regular weights and weight broadcasting, which reduces GPU memory and bandwidth usage even further.

Here are details of different MS-AMP optimization levels:
| Optimization Level  | Computation(GEMM) | Comm  | Weight | Master Weight  | Weight Gradient | Optimizer States |
| ------------------- | -----------       | ----- | ------ | -------------  | --------------- | ---------------- |
| FP16 AMP            | FP16              | FP32  | FP32   | N/A            | FP32            | FP32+FP32        |
| Nvidia TE           | FP8               | FP32  | FP32   | N/A            | FP32            | FP32+FP32        |
| MS-AMP O1           | FP8               | FP8   | FP16   | N/A            | FP8             | FP32+FP32        |
| MS-AMP O2           | FP8               | FP8   | FP16   | N/A            | FP8             | FP8+FP16         |
| MS-AMP O3           | FP8               | FP8   | FP8    | FP16           | FP8             | FP8+FP16         |
