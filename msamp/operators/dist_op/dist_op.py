# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""fp8_op module."""

import os
import ctypes

import torch.distributed as dist
from msamp.common.dtype import Dtypes


class DistOp:
    """MSAMP FP8 library wrapper class."""
    lib_path = '/usr/local/lib/libmsampdist.so'
    lib = None

    @classmethod
    def disable_fp8(cls):
        """Disable fp8. It means uint8/int8 will not be treated as fp8 in ncclAllReduce."""
        cls.lib.disable_fp8()

    @classmethod
    def enable_fp8(cls, qtype):
        """Enable fp8. It means uint8/int8 will be treated as e4m3/e5m2 fp8 in ncclAllReduce."""
        if not Dtypes.is_fp8_qtype(qtype):
            raise RuntimeError(f'qtype {qtype} is not supported in enable_fp8.')
        if qtype == Dtypes.kfloat8_e4m3:
            cls.lib.enable_fp8_e4m3()
        elif qtype == Dtypes.kfloat8_e5m2:
            cls.lib.enable_fp8_e5m2()

    @classmethod
    def all_reduce(cls, qtype, tensor, op, group=None, async_op=False):
        """All reduce tensor.
        
        Args:
            qtype (Qtype): qtype of the tensor.
            tensor (Tensor): tensor to be reduced.
            op (ReduceOp): reduce operation.
            async_op (bool): whether to wait for the operation to finish.
        """
        if not Dtypes.is_fp8_qtype(qtype):
            return dist.all_reduce(tensor, op, group, async_op)
        
        cls.enable_fp8(qtype)
        ret = dist.all_reduce(tensor, op, group, async_op)
        cls.disable_fp8()
        return ret

    @classmethod
    def load_dist_lib(cls):
        """Load msamp fp8 lib."""
        if not os.path.exists(cls.lib_path):
            raise RuntimeError(f'Cannot find {cls.lib_path}, please build msamp dist lib first.')
        try:
            cls.lib = ctypes.cdll.LoadLibrary(cls.lib_path)
        except Exception as e:
            raise RuntimeError(f'Cannot load {cls.lib_path}, exception: {e}')


DistOp.load_dist_lib()
