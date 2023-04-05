# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DistOp module."""

from torch.distributed import ReduceOp as TorchReduceOp
from msamp.common.utils import DistUtil
from msamp.common.dtype import Dtypes
import msamp_dist_op


class DistOp:
    """Distributed operators to support FP8 collective operations."""
    _comm = None
    _qtype_to_nccltype = {
        Dtypes.kfloat16: 6,
        Dtypes.kfloat32: 7,
        Dtypes.kbfloat16: 9,
        Dtypes.kfloat8_e4m3: 10,
        Dtypes.kfloat8_e5m2: 11,
    }
    _torch_reduce_op_to_nccl_reduce_op = {
        TorchReduceOp.SUM: 0,    # ncclSum,
        TorchReduceOp.PRODUCT: 1,    # ncclProd,
        TorchReduceOp.MIN: 2,    # ncclMin,
        TorchReduceOp.MAX: 3,    # ncclMax,
        TorchReduceOp.AVG: 4,    # ncclAvg,
    }

    @classmethod
    def _get_global_comm(cls):
        """Get the communicator.

        Return:
            ncclCommPtr: The communicator ptr.
        """
        if cls._comm is None:
            rank = DistUtil.get_rank()
            world_size = DistUtil.get_world_size()
            nccl_uid = [None]
            if rank == 0:
                nccl_uid[0] = msamp_dist_op.get_nccl_uid()
            DistUtil.broadcast_object_list(nccl_uid, src=0)
            cls._comm = msamp_dist_op.get_communicator(nccl_uid[0], rank, world_size)
        return cls._comm

    @classmethod
    def _get_nccl_reduce_op(cls, op):
        """Get the nccl reduce op.

        Args:
            op (int): one of the values from torch.distributed.ReduceOp enum.

        Return:
            int: The nccl reduce op.
        """
        if op not in cls._torch_reduce_op_to_nccl_reduce_op:
            raise ValueError("Unsupported reduce op: {}".format(op))
        return cls._torch_reduce_op_to_nccl_reduce_op[op]

    @classmethod
    def reduce(cls, tensor, dst, qtype, op):
        """Function of reduce.

        Args:
            tensor (torch.Tensor): tensor to reduce.
            dst (int): the destination rank to reduce.
            qtype (Dtypes.QType): the data type.
            op (int): one of the values from torch.distributed.ReduceOp enum.
        """
        msamp_dist_op.reduce(
            tensor, tensor, dst, cls._get_nccl_reduce_op(op), cls._get_global_comm(), cls._qtype_to_nccltype[qtype]
        )

    @classmethod
    def all_reduce(cls, tensor, qtype, op):
        """Function of allreduce .

        Args:
            tensor (torch.Tensor): tensor to reduce.
            qtype (Dtypes.QType): the data type.
            op (int): one of the values from torch.distributed.ReduceOp enum.
        """
        msamp_dist_op.all_reduce(
            [tensor], [tensor], cls._get_nccl_reduce_op(op), [cls._get_global_comm()], cls._qtype_to_nccltype[qtype]
        )
