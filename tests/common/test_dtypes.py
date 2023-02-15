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
    assert Dtypes.qtype_to_dtype[Dtypes.kfloat8_e4m3] == torch.uint8
    assert Dtypes.qtype_to_dtype[Dtypes.kfloat8_e5m2] == torch.uint8


def test_get_fp_dtype():
    """Test get_fp_dtype in Dtypes"""
    assert len(Dtypes.name_to_dtype) == 11
    assert Dtypes.get_fp_dtype("e4m3") == torch.uint8
    assert Dtypes.get_fp_dtype("e5m2") == torch.int8
    assert Dtypes.get_fp_dtype("half") == torch.float16
    assert Dtypes.get_fp_dtype("bf16") == torch.bfloat16
    assert Dtypes.get_fp_dtype("fp32") == torch.float32
    assert Dtypes.get_fp_dtype("fp64") == torch.float64
