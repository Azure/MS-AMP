# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP dtypes package."""

from msamp.common.dtype.dtypes import Dtypes, QType
from msamp.common.dtype.floating import Floating

__all__ = ['Dtypes', 'QType', 'Floating']
