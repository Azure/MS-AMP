# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP dtypes module."""

from dataclasses import dataclass
import torch

setattr(torch, 'fp8e4m3', torch.uint8)
setattr(torch, 'fp8e5m2', torch.int8)


@dataclass
class QType:
    """Qtype class includes name and value."""
    name: str
    value: int

    def __int__(self):
        """Built-in int function."""
        return self.value

    def __hash__(self):
        """Built-in hash function."""
        return self.value


class Dtypes:
    """Dtypes class which defines dtype and qtype related static variables and functions."""
    kbyte = QType(name='kByte', value=0)
    kint32 = QType(name='kInt32', value=1)
    kfloat32 = QType(name='kFloat32', value=2)
    kfloat16 = QType(name='kFloat16', value=3)
    kbfloat16 = QType(name='kBFloat16', value=4)
    kfloat8_e4m3 = QType(name='kFloat8E4M3', value=5)
    kfloat8_e5m2 = QType(name='kFloat8E5M2', value=6)

    dtype_to_qtype = {
        torch.float16: kfloat16,
        torch.bfloat16: kbfloat16,
        torch.float32: kfloat32,
        torch.fp8e4m3: kfloat8_e4m3,    # type: ignore
        torch.fp8e5m2: kfloat8_e5m2    # type: ignore
    }
    qtype_to_dtype = dict((v, k) for k, v in dtype_to_qtype.items())

    dtype_to_size = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.long: 8,
        torch.int64: 8,
        torch.int32: 4
    }

    @classmethod
    def is_fp8_qtype(cls, qtype):
        """Check whether the qtype is fp8.

        Args:
            qtype (Qtype): qtype to check.

        Return:
            flag (bool): whether qtype is fp8.
        """
        return qtype in [cls.kfloat8_e4m3, cls.kfloat8_e5m2]

    @classmethod
    def get_dtype_from_qtype(cls, qtype):
        """Get dtype from qtype.

        Args:
            qtype (Qtype): qtype to get dtype.

        Return:
            dtype (torch.dtype): dtype of the qtype.
        """
        return torch.uint8 if cls.is_fp8_qtype(qtype) else cls.qtype_to_dtype[qtype]
