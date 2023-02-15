"""MS-AMP dtypes module"""

from dataclasses import dataclass
import torch
from superbench.common.utils import logger


@dataclass
class QType:
    """Qtype class includes name and value."""
    name: str
    value: int

    def __int__(self):
        return self.value

    def __hash__(self):
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
    }
    qtype_to_dtype = dict((v, k) for k, v in dtype_to_qtype.items())
    qtype_to_dtype[kfloat8_e4m3] = torch.uint8
    qtype_to_dtype[kfloat8_e5m2] = torch.uint8

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

    dtype_to_names = {
        torch.uint8: ['e4m3'],
        torch.int8: ['e5m2'],
        torch.float16: ['float16', 'fp16', 'half'],    # E5M10
        torch.bfloat16: ['bfloat16', 'bf16'],    # E8M7
        torch.float32: ['float32', 'fp32'],    # E8M23
        torch.float64: ['float64', 'fp64'],    # E11M52
    }

    name_to_dtype = dict()
    for dt, names in dtype_to_names.items():
        for name in names:
            if name in name_to_dtype:
                logger.warning(f'{name} already exists in name_to_dtype')
                continue
            name_to_dtype[name] = dt

    @classmethod
    def get_fp_dtype(cls, name):
        """Get floating point dtype by name.

        Args:
            name (str): dtype name such e4m3, e5m2.

        Return:
            dtype (torch.dtype): The data type in pytorch.
        """
        return cls.name_to_dtype[name]

    @classmethod
    def is_fp8_qtype(cls, qtype):
        """Check whether the qtype is fp8

        Args:
            qtype (Qtype): qtype to check.

        Return:
            flag (bool): whether qtype is fp8.
        """
        return qtype in {cls.kfloat8_e4m3, cls.kfloat8_e5m2}
