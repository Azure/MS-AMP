# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = dict(cxx=['-fopenmp', '-O3'])

setup(
    name='msamp_dist_op',
    version='0.0.1',
    ext_modules=[CUDAExtension(
        'msamp_dist_op',
        ['dist.cpp'],
        extra_compile_args=extra_compile_args,
    )],
    cmdclass={'build_ext': BuildExtension}
)
