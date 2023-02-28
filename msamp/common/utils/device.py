# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Device module."""

import torch


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
