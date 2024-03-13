# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Device module."""

from enum import Enum

import torch

class GPUType(Enum):
    NVIDIA=1
    AMD=2
    UNKNOW=3

class Device:
    """Device class for different hardwares."""
    @staticmethod
    def get_gpu_type():
        """Get the GPU type."""
        if torch.cuda.device_count() > 0:
            device_name = torch.cuda.get_device_name(0).upper()
            if "NVIDIA" in device_name:
                return GPUType.NVIDIA
            elif "AMD" in device_name:
                return GPUType.AMD
        return GPUType.UNKNOW
