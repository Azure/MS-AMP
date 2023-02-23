# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP dist module."""

import torch.distributed as dist


class DistUtil:
    """Distribution utility class."""
    @staticmethod
    def _is_dist_avail_and_initialized():
        """Check if distributed package is available and initialized.

        Return:
            bool: True if distributed package is available and initialized, otherwide False.
        """
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    @classmethod
    def get_world_size(cls):
        """Get the number of processes in the current process group.

        Return:
            int: return 1 if distributed package is not available or initialized,
                 otherwise the number of processes in current process group.
        """
        if cls._is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1

    @classmethod
    def get_rank(cls):
        """Get the rank of current process in process group.

        Return:
            int: return 0 if distributed package is not available or initialized,
                 otherwise the rank of current processe in current process group.
        """
        if not cls._is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    @classmethod
    def is_main_process(cls):
        """Check if current process is main process in process group.

        Return:
            bool: return True if the rank of current process in process group is zero, otherwise return False.
        """
        return cls.get_rank() == 0
