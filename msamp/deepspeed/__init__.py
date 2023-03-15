# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of DeepSpeed."""

import torch
from deepspeed import version, _parse_version, git_hash, git_branch, Optional, Union
from deepspeed import Optimizer, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from deepspeed import _LRScheduler, log_dist, zero, PipelineModule, PipelineEngine
from .runtime.engine import FP8DeepSpeedEngine

# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch


def initialize(
    args=None,
    model: torch.nn.Module = None,
    optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    training_data: Optional[torch.utils.data.Dataset] = None,
    lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
    mpu=None,
    dist_init_required: Optional[bool] = None,
    collate_fn=None,
    config=None,
    config_params=None
):
    """Initialize the DeepSpeed Engine.

    Args:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.
        model: Required: nn.module class before apply any wrappers
        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.
        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.
        training_data: Optional: Dataset of type torch.utils.data.Dataset
        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and
            returns a Scheduler object. The scheduler object should define a get_lr(), step(), state_dict(),
            and load_state_dict() methods.
        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()
        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.
        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.
        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``
        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.
        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.
        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.
        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist(
        'DeepSpeed info: version={}, git-hash={}, git-branch={}'.format(__version__, __git_hash__, __git_branch__),
        ranks=[0]
    )

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, 'deepspeed.initialize requires a model'

    if not isinstance(model, PipelineModule):
        engine = FP8DeepSpeedEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config=config,
            config_params=config_params
        )
    else:
        assert mpu is None, 'mpu must be None with pipeline parallelism'
        engine = PipelineEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=model.mpu(),
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config=config,
            config_params=config_params
        )

    return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    return tuple(return_items)
