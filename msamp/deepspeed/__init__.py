# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSpeed Extension with MS-AMP support."""

from typing import Optional, Union

import torch
from torch.optim import Optimizer
from deepspeed import *    # noqa
from deepspeed import __version__, __git_hash__, __git_branch__, log_dist, logger,  get_accelerator, \
                      DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable, \
                      DeepSpeedHybridEngine, _LRScheduler, zero, PipelineModule

from msamp.deepspeed.runtime.config import MSAMPDeepSpeedConfig
from msamp.deepspeed.runtime.engine import MSAMPDeepSpeedEngine
from msamp.deepspeed.runtime.pipe.engine import MSAMPPipelineEngine


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

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer
            and returns a Scheduler object. The scheduler object should define a get_lr(), step(),
            state_dict(), and load_state_dict() methods

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

    global dist
    from deepspeed import comm as dist
    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend, dist_init_required=dist_init_required)

    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, 'deepscale_config') and args.deepscale_config is not None:
        logger.warning('************ --deepscale_config is deprecated, please use --deepspeed_config ************')
        if hasattr(args, 'deepspeed_config'):
            assert (
                args.deepspeed_config is None
            ), 'Not sure how to proceed, we were given both a deepscale_config and deepspeed_config'
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config is not None:
        assert config is None, 'Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments ' \
             ' and deepspeed.initialize() function call'
        config = args.deepspeed_config
    assert config is not None, 'DeepSpeed requires --deepspeed_config to specify configuration file'

    if not isinstance(model, PipelineModule):
        config_class = MSAMPDeepSpeedConfig(config, mpu)
        if config_class.hybrid_engine.enabled:
            engine = DeepSpeedHybridEngine(
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
                config_class=config_class
            )
        else:
            engine = MSAMPDeepSpeedEngine(
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
                config_class=config_class
            )
    else:
        assert mpu is None, 'mpu must be None with pipeline parallelism'
        mpu = model.mpu()
        config_class = MSAMPDeepSpeedConfig(config, mpu)
        engine = MSAMPPipelineEngine(
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
            config_class=config_class
        )

    return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    return tuple(return_items)


__all__ = ['initialize']
