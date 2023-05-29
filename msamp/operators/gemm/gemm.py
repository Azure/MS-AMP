# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GEMM module."""

import torch
import torch.nn.functional as F

from msamp.common.dtype import Dtypes
from msamp.common.utils import Device
from msamp.common.utils import TransformerEngineWrapper as tew
from msamp.common.tensor import ScalingTensor


class Gemm:
    """GEMM class to support FP8 precision."""
    _cublas_workspace = None
    _empty_tensor = torch.Tensor()
    _te_base = 16

    @staticmethod
    def _round2times(value, base):
        """Calculate the round value base on base.

        Args:
            value (int): the value to round.
            base (int): the base used to round.

        Return:
            int: value after rounding.
        """
        return (value + base - 1) // base * base

    @staticmethod
    def _get_cublas_workspace_size_bytes():
        """Get the workspace size.

        Return:
            int: Return 32 MiB if using hopper, 4 MiB for all other architectures.
        """
        if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
            return 33_554_432
        return 4_194_304

    @classmethod
    def _get_workspace(cls) -> torch.Tensor:
        """Returns workspace for cublas.

        Return:
            torch.Tensor: workspace for cublas operations.
        """
        if cls._cublas_workspace is None:
            cls._cublas_workspace = torch.empty(cls._get_cublas_workspace_size_bytes(), dtype=torch.int8, device='cuda')
        return cls._cublas_workspace

    @classmethod
    @torch.no_grad()
    def fp8_gemm(
        cls,
        mat_a: ScalingTensor,
        mat_b: ScalingTensor,
        out_qtype,
        bias=None,
        accumulate=False,
        out=None,    # out is not padded
        use_split_accumulator=False
    ):
        """FP8 GEMM operator.

        Args:
            mat_a (ScalingTensor): the left operand of gemm.
            mat_b (ScalingTensor): the right operand of gemm.
            out_qtype (msamp.QType): the output type.
            bias (torch.Tensor): the bias of gemm.
            accumulate (boolean): Whether accumulate the result to out tensor or not.
            out (torch.Tensor): the output tensor.
            use_split_accumulator (boolean): control the computation precision.

        Return:
            torch.Tensor: the result of gemm in out_qtype.
        """
        assert isinstance(mat_a, ScalingTensor)
        assert isinstance(mat_b, ScalingTensor)
        assert mat_a.is_cuda and mat_a.is_contiguous()
        assert mat_b.is_cuda and mat_b.is_contiguous()
        workspace = cls._get_workspace()

        # TN for pytorch shape: (M, K) @ (N, K) = (N, M)
        M, K = mat_a.shape
        N = len(mat_b)
        aM = cls._round2times(M, cls._te_base)
        aK = cls._round2times(K, cls._te_base)
        aN = cls._round2times(N, cls._te_base)
        pM, pK, pN = aM - M, aK - K, aN - N

        a_meta = mat_a.meta
        b_meta = mat_b.meta

        if pM > 0 or pK > 0:
            mat_a = mat_a.pad((0, pK, 0, pM))
        if pN > 0 or pK > 0:
            mat_b = mat_b.pad((0, pK, 0, pN))

        src_out = out
        out_dtype = Dtypes.get_dtype_from_qtype(out_qtype)
        if out is None:
            out = torch.empty(
                mat_b.shape[0],
                mat_a.shape[0],
                dtype=out_dtype,
                device='cuda',
            )
        else:
            if pM > 0 or pN > 0:
                out = F.pad(out, (0, pM, 0, pN))
            assert out.is_cuda and out.is_contiguous()

        out_scale = torch.ones_like(a_meta.scale)
        out_max = torch.ones_like(a_meta.amax)
        bias = (bias if bias is not None else cls._empty_tensor)

        # here out is padded, and src_out is the original one.
        if Device.is_fp8_supported():
            """
            void te_gemm(at::Tensor A,
             at::Tensor A_scale_inverse,
             transformer_engine::DType A_type,
             bool transa,
             at::Tensor B,
             at::Tensor B_scale_inverse,
             transformer_engine::DType B_type,
             bool transb,
             at::Tensor D,
             at::Tensor D_scale,
             transformer_engine::DType D_type,
             at::Tensor D_amax,
             at::Tensor bias,
             transformer_engine::DType bias_type,
             at::Tensor pre_gelu_out,
             bool grad,
             at::Tensor workspace,
             size_t workspaceSize,
             bool accumulate,
             bool use_split_accumulator
            """
            tew.te_gemm(
                mat_a.value,
                a_meta.scale_inv,
                a_meta.qtype,
                True,    # transa
                mat_b.value,
                b_meta.scale_inv,
                b_meta.qtype,
                False,    # transb
                out,
                out_scale,    # scale
                out_qtype,
                out_max,
                bias,
                Dtypes.dtype_to_qtype[bias.dtype],
                cls._empty_tensor,
                False,    # grad
                workspace,
                workspace.shape[0],
                accumulate,
                use_split_accumulator,
            )
        else:
            # do gemm on device that doesn't supported fp8.
            mat_a, mat_b = mat_a.to(out_dtype), mat_b.to(out_dtype)
            tew.te_gemm(
                mat_a,
                cls._empty_tensor,
                out_qtype,
                True,
                mat_b,
                cls._empty_tensor,
                out_qtype,
                False,
                out,
                out_qtype,
                bias if bias is not None else cls._empty_tensor,
                cls._empty_tensor,
                False,    # grad
                workspace,
                workspace.shape[0],
                accumulate,
                False,
            )

        if pN > 0 or pM > 0:
            out = out[:N, :M]
            if src_out is not None:
                src_out.data.copy_(out)
                out = src_out
            else:
                out = out.contiguous()
        return out
