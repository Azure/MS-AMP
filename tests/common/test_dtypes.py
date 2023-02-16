import torch
from msamp.common.dtype.dtypes import Dtypes


def test_is_fp8_qtype():
    """Test is_fp8_qtype function in Dtypes."""
    assert Dtypes.is_fp8_qtype(Dtypes.kfloat8_e4m3)
    assert Dtypes.is_fp8_qtype(Dtypes.kfloat8_e5m2)
    assert not Dtypes.is_fp8_qtype(Dtypes.kfloat16)


def test_qtype_to_dtype():
    """Test qtype_to_dtype in Dtypes."""
    assert len(Dtypes.qtype_to_dtype) == 5
    assert Dtypes.qtype_to_dtype[Dtypes.kfloat8_e4m3] == torch.fp8e4m3
    assert Dtypes.qtype_to_dtype[Dtypes.kfloat8_e5m2] == torch.fp8e5m2
