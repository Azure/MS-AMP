import torch
from dataclasses import dataclass


@dataclass
class DType:
    name: str
    value: int
    def __int__(self):
        return self.value
    def __hash__(self):
        return self.value


kByte       = DType(name='kByte', value=0)
kInt32      = DType(name='kInt32', value=1)
kFloat32    = DType(name='kFloat32', value=2)
kFloat16    = DType(name='kFloat16', value=3)
kBFloat16   = DType(name='kBFloat16', value=4)
kFloat8E4M3 = DType(name='kFloat8E4M3', value=5)
kFloat8E5M2 = DType(name='kFloat8E5M2', value=6)


QFP_MAXS = {
    kFloat8E4M3: 448,
    kFloat8E5M2: 57344,
    kFloat16: 65504,  # E5M10
}


DType2QType = {
    torch.float16: kFloat16,
    torch.bfloat16: kBFloat16,
    torch.float32: kFloat32,
}


QType2DType = dict((v, k) for k, v in DType2QType.items())
QType2DType[kFloat8E4M3] = torch.uint8
QType2DType[kFloat8E5M2] = torch.uint8


def is_fp8_qtype(qtype):
    return qtype in {kFloat8E4M3, kFloat8E5M2}


DType2Size = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
              torch.uint8: 1, torch.int8: 1, torch.long: 2, torch.int64: 2, torch.int32: 4}
QType2Size = dict((k, DType2Size[v]) for k, v in QType2DType.items())