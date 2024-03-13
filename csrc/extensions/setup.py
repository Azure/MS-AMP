# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module."""

from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import IS_HIP_EXTENSION

ext_t = cpp_extension.CUDAExtension

ext_fnames = ['operators/arithmetic.cu', 'optim/adamw.cu', "pybind.cpp"]

define_macros = []

if IS_HIP_EXTENSION:
    cxx_flags = ["-O3"]
    nvcc_flags = []

    cpp_extension.COMMON_HIPCC_FLAGS.remove('-D__HIP_NO_HALF_OPERATORS__=1')
    cpp_extension.COMMON_HIPCC_FLAGS.remove('-D__HIP_NO_HALF_CONVERSIONS__=1')
else:
    cxx_flags = ["-O3"]
    nvcc_flags = [
        '-O3', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '--expt-relaxed-constexpr',
        '--expt-extended-lambda', '--use_fast_math'
    ]
    
    define_macros.append(('WITH_CUDA', None))

extra_compile_args = dict(cxx=cxx_flags, nvcc=nvcc_flags)

setup(
    name='msamp_extension',
    version='0.0.1',
    ext_modules=[
        ext_t('msamp_extension', ext_fnames, define_macros=define_macros, extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
