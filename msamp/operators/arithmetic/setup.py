# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module."""

from setuptools import setup
from torch.utils import cpp_extension

ext_t = cpp_extension.CUDAExtension
ext_fnames = ['arithmetic.cu']
define_macros = []
nvcc_flags = [
    '-O3', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '--expt-relaxed-constexpr',
    '--expt-extended-lambda', '--use_fast_math'
]

extra_compile_args = dict(cxx=['-fopenmp', '-O3'], nvcc=nvcc_flags)

define_macros.append(('WITH_CUDA', None))

setup(
    name='msamp_arithmetic',
    version='0.0.1',
    ext_modules=[
        ext_t('msamp_arithmetic', ext_fnames, define_macros=define_macros, extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
