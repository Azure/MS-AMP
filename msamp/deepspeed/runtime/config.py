# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Config with MS-AMP support."""

from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.zero.config import ZeroStageEnum

MSAMP = 'msamp'
MSAMP_ENABLED = 'enabled'
MSAMP_ENABLED_DEFAULT = False
MSAMP_OPTLEVEL = 'opt_level'
MSAMP_OPTLEVEL_DEFAULT = 'O1'
MSAMP_USE_TE = 'use_te'
MSAMP_USE_TE_DEFAULT = False
FP8 = 'fp8'


class MSAMPDeepSpeedConfig(DeepSpeedConfig):
    """DeepSpeed Config with MS-AMP support."""
    def _initialize_params(self, param_dict):
        """Initialize the parameters from the parameter dictionary.

        Args:
            param_dict (dict): The parameter dictionary.
        """
        super()._initialize_params(param_dict)
        self.msamp_enabled = get_msamp_enabled(param_dict)
        self.msamp_optlevel = get_msamp_optlevel(param_dict)
        self.msamp_use_te = get_msamp_use_te(param_dict)

    def _do_error_check(self):
        """Do error checking on the parameters."""
        super()._do_error_check()
        if not self.msamp_enabled:
            return

        assert self.msamp_optlevel in ['O1', 'O2', 'O3'],  \
            f'Invalid MS-AMP opt_level: {self.msamp_optlevel}, only O1, O2, O3 are supported.'
        if self.msamp_optlevel == 'O3':
            assert self.zero_enabled and \
                self.zero_optimization_stage in [ZeroStageEnum.optimizer_states, ZeroStageEnum.gradients], \
                'MS-AMP O3 requires ZeRO with optimizer_states or gradients partitioning.'


def get_msamp_enabled(param_dict):
    """Get the MS-AMP enabled flag from the parameter dictionary.

    Args:
        param_dict (dict): The parameter dictionary.

    Returns:
        bool: The MS-AMP enabled flag.
    """
    if MSAMP in param_dict.keys():
        return get_scalar_param(param_dict[MSAMP], MSAMP_ENABLED, MSAMP_ENABLED_DEFAULT)
    return False


def get_msamp_optlevel(param_dict):
    """Get the MS-AMP opt_level from the parameter dictionary.

    Args:
        param_dict (dict): The parameter dictionary.

    Returns:
        str: The MS-AMP opt_level.
    """
    if MSAMP in param_dict.keys():
        return get_scalar_param(param_dict[MSAMP], MSAMP_OPTLEVEL, MSAMP_OPTLEVEL_DEFAULT)
    return None


def get_msamp_use_te(param_dict):
    """Get the MS-AMP use_te from the parameter dictionary.

    Args:
        param_dict (dict): The parameter dictionary.

    Returns:
        str: The MS-AMP use_te.
    """
    if MSAMP in param_dict.keys():
        return get_scalar_param(param_dict[MSAMP], MSAMP_USE_TE, MSAMP_USE_TE_DEFAULT)
    return None
