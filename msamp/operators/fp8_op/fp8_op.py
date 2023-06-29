# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fp8Op module."""

import ctypes

class Fp8Op:
    lib = ctypes.cdll.LoadLibrary('libmsampfp8.so')

    @classmethod
    def disable_fp8(self):
        """Disable fp8. It means uint8/int8 will not be treated as fp8 in ncclAllReduce."""
        Fp8Op.lib.disable_fp8()

    @classmethod
    def enable_fp8_e4m3(self):
        """Enable fp8. It means uint8 will be treated as e4m3-fp8 in ncclAllReduce."""
        Fp8Op.lib.enable_fp8_e4m3()

    @classmethod
    def enable_fp8_e5m2(self):
        """Enable fp8. It means uint8 will be treated as e5m2-fp8 in ncclAllReduce."""
        Fp8Op.lib.enable_fp8_e5m2()
