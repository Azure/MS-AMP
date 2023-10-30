# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP te package."""

from msamp.te import extension
from msamp.te import modules
from msamp.te.replacer import TeReplacer

del extension
del modules

__all__ = ['TeReplacer']
