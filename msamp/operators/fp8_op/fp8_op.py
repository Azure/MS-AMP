# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""fp8_op module."""

import os
import ctypes


class FP8Op:
    """MSAMP FP8 library wrapper class."""
    lib_path = "/usr/local/lib/libmsampfp8.so"
    lib = None

    @classmethod
    def disable_fp8(cls):
        """Disable fp8. It means uint8/int8 will not be treated as fp8 in ncclAllReduce."""
        cls.lib.disable_fp8()

    @classmethod
    def enable_fp8_e4m3(cls):
        """Enable fp8. It means uint8 will be treated as e4m3-fp8 in ncclAllReduce."""
        cls.lib.enable_fp8_e4m3()

    @classmethod
    def enable_fp8_e5m2(cls):
        """Enable fp8. It means uint8 will be treated as e5m2-fp8 in ncclAllReduce."""
        cls.lib.enable_fp8_e5m2()

    @classmethod
    def load_fp8_lib(cls):
        """Load msamp fp8 lib."""
        if not os.path.exists(cls.lib_path):
            raise RuntimeError(f'Cannot find {cls.lib_path}, please build msamp fp8 lib first.')
        try:
            cls.lib = ctypes.cdll.LoadLibrary(cls.lib_path)
        except Exception as e:
            raise RuntimeError(f'Cannot load {cls.lib_path}, exception: {e}')

FP8Op.load_fp8_lib()
