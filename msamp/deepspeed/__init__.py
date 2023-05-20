import inspect
from deepspeed import *
from deepspeed import _LRScheduler, _parse_version
from deepspeed.accelerator import get_accelerator
from .runtime.engine import MSAMPDeepSpeedEngine as DeepSpeedEngine

# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch


# Re-create deepspeed.initialize
exec(compile(inspect.getsource(initialize), '<string>', mode='exec'))
