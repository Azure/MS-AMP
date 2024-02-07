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
    def is_fp8_supported():
        """Check whether the device support FP8 or not.

        Return:
            boolean: return True if the device support FP8 precision.
        """
        gpu_name = torch.cuda.get_device_name().lower()
        if 'h100' in gpu_name:
            return True

        return False

    @staticmethod
    def get_gpu_type():
        """Get the GPU type."""
        if torch.cuda.device_count() > 0:
            device_name = torch.cuda.get_device_name(0)
            if "NVIDIA" in device_name:
                return GPUType.NVIDIA
            elif "AMD" in device_name:
                return GPUType.AMD
        return GPUType.UNKNOW
