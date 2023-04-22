# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module."""

from setuptools import setup
from pkg_resources import parse_version

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

define_macros = [
    ('TORCH_VERSION_MAJOR', parse_version(torch.__version__).major),
    ('TORCH_VERSION_MINOR', parse_version(torch.__version__).minor),
]
extra_compile_args = dict(cxx=['-fopenmp', '-O3'])

setup(
    name='msamp_dist_op',
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            'msamp_dist_op',
            ['dist.cpp'],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=['nccl'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
