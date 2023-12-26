import collections
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
import torch.distributed.fsdp._exec_order_utils as exec_order_utils
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state,
    _is_fsdp_flattened,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp._wrap_utils import _get_fully_sharded_module_to_states
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.flat_param import (
    _HandlesKey,
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
)
from torch.distributed.fsdp.wrap import _FSDPPolicy
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils.hooks import RemovableHandle

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False

PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)
FSDP_SYNCED = "_fsdp_synced"
# Specification of process groups for hybrid sharding strategies.
HybridShardProcessGroupType = Tuple[dist.ProcessGroup, dist.ProcessGroup]
# Overall specification of process group.
ProcessGroupType = Optional[Union[dist.ProcessGroup, HybridShardProcessGroupType]]


# TODO (awgu): Refactor this later
SHARDING_STRATEGY_MAP = {
    ShardingStrategy.NO_SHARD: HandleShardingStrategy.NO_SHARD,
    ShardingStrategy.FULL_SHARD: HandleShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP: HandleShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.HYBRID_SHARD: HandleShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2: HandleShardingStrategy._HYBRID_SHARD_ZERO2,
}

HYBRID_SHARDING_STRATEGIES = {
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
}


# NOTE: Since non-self attributes cannot be type annotated, several attributes
# on `state` are defined first as local variables before being assigned.


@no_type_check
def _init_process_group_state(
    state: _FSDPState,
    process_group: ProcessGroupType,
    sharding_strategy: ShardingStrategy,
    policy: Optional[_FSDPPolicy],
) -> _FSDPState:
    if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
        if process_group is None and policy is None:
            # Raise an error here, since this is manual wrapping with no process group
            # passed in, there is no way to ensure all wrapped FSDP instances use the same
            # process groups.
            raise ValueError(
                f"Manual wrapping with {sharding_strategy} requires explicit specification of process group."
            )
        else:
            state = _init_process_group_state_for_hybrid_shard(state, process_group)
            assert (
                state.process_group is not None
            ), "Expected to populate state.process_group for hybrid shard"
            assert (
                state._inter_node_pg is not None
            ), "Expected to populate state._inter_node_pg for hybrid shard"
            assert (
                state._inter_node_state is not None
            ), "Expected to populate state._inter_node_state for hybrid shad."
    else:
        state.process_group = (
            process_group if process_group is not None else _get_default_group()
        )

    state.rank = state.process_group.rank()
    state.world_size = state.process_group.size()

    return state


@no_type_check
def _init_process_group_state_for_hybrid_shard(
    state: _FSDPState, process_group
) -> _FSDPState:
    if process_group is None:
        default_group = _get_default_group()
        intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
            default_group
        )
        # we shard across intra-node
        state.process_group = intra_node_group
        # save _inter_node_pg to allreduce across.
        state._inter_node_pg = inter_node_group
    else:
        # Check type and assign state.process_group and state._inter_node_pg.
        if _is_valid_hybrid_shard_pg_type(process_group):
            # Assuming that user passed in as intra node group and inter node group
            # as documented.
            state.process_group, state._inter_node_pg = process_group
        else:
            raise ValueError(
                "Expected process_group to be passed in as either None or "
                f"Tuple[dist.ProcessGroup, dist.ProcessGroup] but got {type(process_group)}"
            )
    # Create state for allreduce
    state._inter_node_state = _get_default_comm_hook_state(
        process_group=state._inter_node_pg,
    )
    return state


@no_type_check
def _is_valid_hybrid_shard_pg_type(process_group: Any) -> bool:
    return (
        isinstance(process_group, tuple)
        and len(process_group) == 2
        and all(isinstance(pg, dist.ProcessGroup) for pg in process_group)
    )


@no_type_check
def _init_intra_node_process_group() -> dist.ProcessGroup:
    """
    Returns a process group across the current node.
    For example, given each row is a distinct node:
    0 1 2 3 4 5 6 7 8
    9 10 11 12 13 14 15
    This API would return an intra-node subgroup across
    [0, 7] or [8, 15] depending on the process's rank.
    For example, rank 3 would get [0, 7].
    """
    intra_node_subgroup, _ = dist.new_subgroups()
    return intra_node_subgroup


@no_type_check
def _init_inter_node_process_group(
    global_process_group: dist.ProcessGroup,
) -> dist.ProcessGroup:
    """
    Returns an inter-node process group where each contained rank has
    the same local rank. For example, given each column is a distinct node:
    0 1 2 3 4 5 6 7 8
    9 10 11 12 13 14 15
    This API would return inter-node process group {0, 8}, {1, 9}, {2, 10}, and so forth
    depending on the process's rank. For example, rank 1 would get {1, 9}, rank 5
    would get {5, 13}.
    """
    # the inter-node pg that is returned
    inter_node_pg = None
    sharding_backend = dist.get_backend(global_process_group)
    world_size = dist.get_world_size(global_process_group)
    # Assuming fully homogeneous setup
    num_devices = torch.cuda.device_count()
    num_nodes = world_size // num_devices
    my_local_rank = dist.get_rank(global_process_group) % num_devices
    for local_rank in range(num_devices):
        ranks_for_inter_group = [
            local_rank + (i * num_devices) for i in range(num_nodes)
        ]
        # every rank always needs to call dist.new_group
        grp = dist.new_group(ranks=ranks_for_inter_group, backend=sharding_backend)
        if local_rank == my_local_rank:
            print(f"{local_rank} created process group for {ranks_for_inter_group}")
            inter_node_pg = grp

    assert (
        inter_node_pg is not None
    ), f"{my_local_rank} expected to assign inter-node pg, but did not"
    return inter_node_pg


def _init_intra_and_inter_node_groups(
    global_process_group: dist.ProcessGroup,
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Initializes intra and inter-node process groups and returns the ones corresponding
    to this process's rank.
    This function can be used to initialize process groups for ``HYBRID_SHARD`` or
    ``_HYBRID_SHARD_ZERO2`` in FSDP.
    This function assumes each node has an equal number of CUDA-enabled devices.
    Returns:
        Tuple[dist.ProcessGroup, dist.ProcessGroup]: Intra and inter-node process group.
    """
    return (
        _init_intra_node_process_group(),
        _init_inter_node_process_group(global_process_group),
    )


@no_type_check
def _init_ignored_module_states(
    state: _FSDPState,
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]],
    ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
) -> _FSDPState:
    assert (
        ignored_modules is None or ignored_parameters is None
    ), "Can not pass `ignored_modules` and `ignored_parameters` at the same time. \
        Please either pass `ignored_modules` or `ignored_parameters`."
    state._ignored_modules = _get_ignored_modules(module, ignored_modules)
    state._ignored_params = _get_ignored_params(
        module,
        state._ignored_modules,
        ignored_parameters,
    )

    # TODO: FSDP's contract for buffers is not well-defined. They are
    # implicitly ignored for most functionality since they are not sharded;
    # however, FSDP still imposes some semantics on buffers (e.g. buffer mixed
    # precision). We should formalize this contract and decide if we need to
    # compute and store `_ignored_buffers`.
    return state


@no_type_check
def _init_buffer_state(
    state: _FSDPState,
    module: nn.Module,
) -> _FSDPState:
    state._buffer_names = _get_buffer_names(module)
    # Save a mapping from clean fully-qualified buffer name (starting from
    # `module`) to its original dtype for restoring that dtype during model
    # checkpointing when buffer mixed precision is enabled. The names should
    # be clean since the casting happens in a `summon_full_params()` context.
    _buffer_name_to_orig_dtype: Dict[str, torch.dtype] = {}
    for buffer_name, buffer in module.named_buffers():
        buffer_name = clean_tensor_name(buffer_name)
        _buffer_name_to_orig_dtype[buffer_name] = buffer.dtype
    state._buffer_name_to_orig_dtype = _buffer_name_to_orig_dtype
    return state


@no_type_check
def _init_core_state(
    state: _FSDPState,
    sharding_strategy: Optional[ShardingStrategy],
    mixed_precision: Optional[MixedPrecision],
    cpu_offload: Optional[CPUOffload],
    limit_all_gathers: bool,
    use_orig_params: bool,
    backward_prefetch_limit: int,
    forward_prefetch_limit: int,
) -> _FSDPState:
    # We clamp the strategy to `NO_SHARD` for world size of 1 since they are
    # currently functionally equivalent. This may change if/when we integrate
    # FSDP with MoE.
    if state.world_size == 1:
        if sharding_strategy != ShardingStrategy.NO_SHARD:
            warnings.warn(
                "FSDP is switching to use `NO_SHARD` instead of "
                f"{sharding_strategy or ShardingStrategy.FULL_SHARD} since "
                "the world size is 1."
            )
        sharding_strategy = ShardingStrategy.NO_SHARD
    state.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
    state.mixed_precision = mixed_precision or MixedPrecision()
    state.cpu_offload = cpu_offload or CPUOffload()
    state.limit_all_gathers = limit_all_gathers
    state._use_orig_params = use_orig_params
    state.training_state = TrainingState.IDLE
    state._is_root = None
    _streams: Dict[str, torch.cuda.Stream] = {}
    state._streams = _streams
    _stream_to_name: Dict[torch.cuda.Stream, str] = {}
    state._stream_to_name = _stream_to_name
    state._free_event_queue = _FreeEventQueue()
    state._debug_level = dist.get_debug_level()
    state._exec_order_data = exec_order_utils._ExecOrderData(
        state._debug_level,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    # Mapping from fully sharded module to the handles it is responsible to
    # unshard and reshard (see [Note: Fully Sharded Module])
    _fully_sharded_module_to_handles: Dict[
        nn.Module, List[FlatParamHandle]
    ] = collections.defaultdict(list)
    state._fully_sharded_module_to_handles = _fully_sharded_module_to_handles
    # Invariant: `state.params` contains exactly the `FlatParameter`s of the
    # handles in `state._handles`
    _handles: List[FlatParamHandle] = []
    state._handles = _handles
    params: List[FlatParameter] = []
    state.params = params
    return state


@no_type_check
def _init_runtime_state(
    state: _FSDPState,
) -> _FSDPState:
    _root_pre_forward_handles: List[RemovableHandle] = []
    state._root_pre_forward_handles = _root_pre_forward_handles
    _pre_forward_handles: List[RemovableHandle] = []
    state._pre_forward_handles = _pre_forward_handles
    _post_forward_handles: List[RemovableHandle] = []
    state._post_forward_handles = _post_forward_handles
    state._sync_gradients = True
    state._communication_hook = _get_default_comm_hook(state.sharding_strategy)
    state._communication_hook_state = _get_default_comm_hook_state(state.process_group)
    state._hook_registered = False
    # Used to prevent running the pre-backward hook multiple times
    _ran_pre_backward_hook: Dict[_HandlesKey, bool] = {}
    state._ran_pre_backward_hook = _ran_pre_backward_hook
    return state


@no_type_check
def _init_prefetching_state(
    state: _FSDPState,
    backward_prefetch: BackwardPrefetch,
    forward_prefetch: bool,
) -> _FSDPState:
    state.backward_prefetch = backward_prefetch
    state.forward_prefetch = forward_prefetch
    _handles_prefetched: Dict[_HandlesKey, bool] = {}
    state._handles_prefetched = _handles_prefetched
    # Used for guarding against mistargeted backward prefetches
    _needs_pre_backward_unshard: Dict[_HandlesKey, bool] = {}
    state._needs_pre_backward_unshard = _needs_pre_backward_unshard
    # Used for guarding against mistargeted forward prefetches
    _needs_pre_forward_unshard: Dict[_HandlesKey, bool] = {}
    state._needs_pre_forward_unshard = _needs_pre_forward_unshard
    # The data structures use tuples of handles to generalize over the case
    # where a module's forward involves multiple handles.
    return state


def _init_state_dict_state(state: _FSDPState) -> _FSDPState:
    state._state_dict_type = StateDictType.FULL_STATE_DICT
    state_dict_config: StateDictConfig = FullStateDictConfig()
    state._optim_state_dict_config = FullOptimStateDictConfig()
    state._state_dict_config = state_dict_config
    unshard_params_ctx: Dict[nn.Module, Generator] = {}
    state._unshard_params_ctx = unshard_params_ctx
    return state


@no_type_check
def _init_param_handle_from_module(
    state: _FSDPState,
    fully_sharded_module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
    module_wrapper_cls: Type,
) -> _FSDPState:
    """
    Initializes a ``FlatParamHandle`` from a module ``fully_sharded_module``.
    This is the module wrapper code path.
    """
    _check_single_device_module(fully_sharded_module, state._ignored_params)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    is_meta_module, is_torchdistX_deferred_init = _need_to_materialize_module(
        fully_sharded_module, state._ignored_params
    )
    # Materialize the module if needed
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        _materialize_with_param_init_fn(fully_sharded_module, param_init_fn)
    elif is_meta_module:
        _materialize_meta_module(fully_sharded_module, device_id)
    elif is_torchdistX_deferred_init:
        deferred_init.materialize_module(
            fully_sharded_module,
            check_fn=lambda k: not isinstance(k, module_wrapper_cls),
        )
    # TODO: Investigate refactoring `_move_module_to_device()` to
    # `_move_states_to_device()` to avoid the `device_id` + CPU offload hack
    _move_module_to_device(
        fully_sharded_module, state._ignored_params, device_from_device_id
    )
    state.compute_device = _get_compute_device(
        fully_sharded_module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
    )
    managed_params = list(_get_orig_params(fully_sharded_module, state._ignored_params))
    if sync_module_states:
        _sync_module_params_and_buffers(
            fully_sharded_module, managed_params, state.process_group
        )
    _init_param_handle_from_params(state, managed_params, fully_sharded_module)
    return state


@no_type_check
def _init_param_handles_from_module(
    state: _FSDPState,
    root_module: nn.Module,
    policy: _FSDPPolicy,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
) -> _FSDPState:
    """
    Initializes all ``FlatParamHandle`` s from a module ``root_module``. This
    is the non-module-wrapper code path. ``root_module`` is guaranteed to be
    a fully sharded module, and some of its submodules may be as well,
    depending on ``policy``. See [Note: Fully Sharded Module].
    """
    fully_sharded_module_to_states = _get_fully_sharded_module_to_states(
        root_module,
        policy,
        state._ignored_modules,
        state._ignored_params,
    )
    _check_single_device_module(root_module, state._ignored_params)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    # Initialize and shard `FlatParamHandle`s one by one following reverse
    # depth-first order (i.e. reverse `.modules()` order), which represents a
    # reverse topological sort order. This avoids increasing peak GPU memory
    # usage when the unsharded model exists on CPU or meta device.
    # NOTE: This order differs from that followed by the wrapper path when
    # using auto wrapping, which also represents a valid reverse toplogical
    # sort order, but the difference does not matter.
    materialized_module = False
    for fully_sharded_module, (params, buffers) in reversed(
        fully_sharded_module_to_states.items()
    ):
        # Materialize the module if needed
        is_meta_module, is_torchdistX_deferred_init = _need_to_materialize_module(
            fully_sharded_module, state._ignored_params
        )
        if is_meta_module or is_torchdistX_deferred_init:
            materialized_module = True
            # Save the parameter and buffer names to reacquire references after
            # after materialization since their variables may change
            param_names, buffer_names = _get_state_names_for_states(
                fully_sharded_module, params, buffers
            )
        if (
            is_meta_module or is_torchdistX_deferred_init
        ) and param_init_fn is not None:
            _materialize_with_param_init_fn(fully_sharded_module, param_init_fn)
        elif is_meta_module:
            _materialize_meta_module(fully_sharded_module, device_id)
        elif is_torchdistX_deferred_init:
            deferred_init.materialize_module(
                root_module,
                check_fn=lambda _: True,
            )
        if materialized_module:
            # Reacquire references using the pre-computed state names
            params = [
                fully_sharded_module.get_parameter(param_name)
                for param_name in param_names
            ]
            buffers = [
                fully_sharded_module.get_buffer(buffer_name)
                for buffer_name in buffer_names
            ]
        _move_states_to_device(params, buffers, device_from_device_id)
        if not hasattr(state, "compute_device"):  # only need to set once
            state.compute_device = _get_compute_device(
                fully_sharded_module,
                state._ignored_params,
                device_from_device_id,
                state.rank,
            )
        if sync_module_states:
            _sync_module_states(params, buffers, state.process_group)
        _init_param_handle_from_params(state, params, fully_sharded_module)
    # Reverse `_handles` to preserve depth-first `.modules()` order for
    # consistency with the wrapper path (namely, so that `_get_fsdp_handles()`
    # returns the same ordering for both paths).
    state._handles.reverse()
    return state


@no_type_check
def _init_param_handle_from_params(
    state: _FSDPState,
    params: List[nn.Parameter],
    fully_sharded_module: nn.Module,
):
    if len(params) == 0:
        return
        
    handle = FlatParamHandle(
        params,
        fully_sharded_module,
        state.compute_device,
        SHARDING_STRATEGY_MAP[state.sharding_strategy],
        state.cpu_offload.offload_params,
        state.mixed_precision.param_dtype,
        state.mixed_precision.reduce_dtype,
        state.mixed_precision.keep_low_precision_grads,
        state.process_group,
        state._use_orig_params,
    )

    # TODO: Can simplify call `shard()` in the `FlatParamHandle` ctor
    handle.shard()

    assert handle not in state._handles
    state.params.append(handle.flat_param)
    state._handles.append(handle)
    state._fully_sharded_module_to_handles[handle._fully_sharded_module].append(handle)
    num_fully_sharded_module_handles = len(
        state._fully_sharded_module_to_handles[handle._fully_sharded_module]
    )
    assert num_fully_sharded_module_handles == 1, (
        "The current design assumes a module manages at most one "
        f"`FlatParamHandle` but got {num_fully_sharded_module_handles}"
    )
    cpu_device = torch.device("cpu")
    if state.cpu_offload.offload_params and handle.flat_param.device != cpu_device:
        handle.flat_param_to(cpu_device)


def _get_state_names_for_states(
    module: nn.Module,
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
) -> Tuple[List[str], List[str]]:
    """
    Returns the parameter and buffer names of the given ``params`` and
    ``buffers``, where the names are prefixed starting from ``module``. This
    function assumes that the parameters and buffers are in the module tree.
    """
    param_names: List[str] = []
    buffer_names: List[str] = []
    param_to_param_name = {
        param: param_name for param_name, param in module.named_parameters()
    }
    buffer_to_buffer_name = {
        buffer: buffer_name for buffer_name, buffer in module.named_buffers()
    }
    for param in params:
        assert (
            param in param_to_param_name
        ), f"Parameter not in the module tree:\n{module}\n{param}"
        param_names.append(param_to_param_name[param])
    for buffer in buffers:
        assert (
            buffer in buffer_to_buffer_name
        ), f"Buffer not in the module tree:\n{module}\n{buffer}"
        buffer_names.append(buffer_to_buffer_name[buffer])
    return param_names, buffer_names


def _get_ignored_modules(
    root_module: nn.Module,
    _ignored_modules: Optional[Iterable[torch.nn.Module]],
) -> Set[nn.Module]:
    """
    Checks that ``_ignored_modules`` is an iterable of ``nn.Module`` s without
    any FSDP instances, and returns the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.

    ``_ignored_modules`` represents the argument passed by the user to FSDP.
    """
    msg_prefix = "`ignored_modules` should be an iterable of `torch.nn.Module`s "
    try:
        ignored_root_modules = (
            set(_ignored_modules) if _ignored_modules is not None else set()
        )
    except TypeError as e:
        raise TypeError(msg_prefix + f"but got {type(_ignored_modules)}") from e
    for module in ignored_root_modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(msg_prefix + f"but got an iterable with {type(module)}")
        if isinstance(module, fsdp_file.FullyShardedDataParallel):
            # TODO: We may relax this by taking the FSDP instance's wrapped
            # module to provide more flexibility to the user.
            raise ValueError("`ignored_modules` should not include FSDP modules")
    # Treat modules that cannot compose with `fully_shard` as ignored modules,
    # meaning that their subtrees are ignored
    for module in root_module.modules():
        if not traversal_utils._composable(module):
            ignored_root_modules.add(module)
    # NOTE: Even if `ignored_root_modules` is empty, do not return early so
    # that this FSDP instance can get any ignored modules from its children.

    # Include child modules and exclude nested FSDP modules themselves
    ignored_modules = {
        child
        for module in ignored_root_modules
        for child in module.modules()
        if not isinstance(child, fsdp_file.FullyShardedDataParallel)
    }
    if root_module in ignored_modules:
        warnings.warn(
            "Trying to ignore the top-level module passed into the FSDP "
            "constructor itself will result in all parameters being "
            f"ignored and is not well-supported: {module}"
        )
    # Include nested FSDP modules' ignored modules
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            assert hasattr(optional_fsdp_state, "_ignored_modules")
            ignored_modules.update(optional_fsdp_state._ignored_modules)
    return ignored_modules


def _get_ignored_params(
    root_module: torch.nn.Module,
    ignored_modules: Set[torch.nn.Module],
    ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
) -> Set[torch.nn.Parameter]:
    """
    Returns the parameters of the modules in ``ignored_modules`` and
    the parameters in ``ignored_parameters``, excluding any :class:`FlatParameter` s.
    """
    all_ignored_params: Set[torch.nn.Parameter] = set()

    params_in_ignored_modules = {
        p for m in ignored_modules for p in m.parameters() if not _is_fsdp_flattened(p)
    }

    all_ignored_params.update(params_in_ignored_modules)

    if ignored_parameters is not None:
        params_in_ignored_parameters = {
            p for p in ignored_parameters if not _is_fsdp_flattened(p)
        }
        all_ignored_params.update(params_in_ignored_parameters)

        # Include nested FSDP modules' ignored parameters
        for submodule in root_module.modules():
            optional_fsdp_state = _get_module_fsdp_state(submodule)
            if optional_fsdp_state is not None:
                assert hasattr(optional_fsdp_state, "_ignored_params")
                all_ignored_params.update(optional_fsdp_state._ignored_params)

    return all_ignored_params


def _get_buffer_names(root_module: nn.Module) -> Set[str]:
    """
    Returns the fully prefixed names of all buffers in the module hierarchy
    rooted at ``root_module`` as a class:`set`.
    """
    return {
        clean_tensor_name(buffer_name) for buffer_name, _ in root_module.named_buffers()
    }


def _check_single_device_module(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> None:
    """
    Raises an error if ``module`` has original parameters on multiple devices,
    ignoring the parameters in ``ignored_params``. Thus, after this method, the
    module must be either fully on the CPU or fully on a non-CPU device.
    """
    devices = {param.device for param in _get_orig_params(module, ignored_params)}
    if len(devices) > 1:
        raise RuntimeError(
            f"FSDP only supports single device modules but got params on {devices}"
        )


def _get_device_from_device_id(
    device_id: Optional[Union[int, torch.device]],
    rank: int,
) -> Optional[torch.device]:
    """
    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    if device_id is None:
        return None
    device = (
        device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    )
    if device == torch.device("cuda"):
        warnings.warn(
            f"FSDP got the argument `device_id` {device_id} on rank "
            f"{rank}, which does not have an explicit index. "
            f"FSDP will use the current device {torch.cuda.current_device()}. "
            "If this is incorrect, please explicitly call `torch.cuda.set_device()` "
            "before FSDP initialization or pass in the explicit device "
            "index as the `device_id` argument."
        )
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _need_to_materialize_module(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Tuple[bool, bool]:
    """
    Returns if ``module`` has parameters on meta device and if ``module`` is
    using torchdistX deferred initialization. At most of the returned bools can
    be ``True``. If either is ``True``, then ``module`` needs to be
    materialized.
    """
    managed_params = _get_orig_params(module, ignored_params)
    is_meta_module = any(param.is_meta for param in managed_params)
    is_torchdistX_deferred_init = (
        not is_meta_module
        and _TORCHDISTX_AVAIL
        and any(fake.is_fake(param) for param in managed_params)
    )
    return is_meta_module, is_torchdistX_deferred_init


def _materialize_with_param_init_fn(
    module: nn.Module,
    param_init_fn,
) -> None:
    if not callable(param_init_fn):
        raise ValueError(
            f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
        )
    param_init_fn(module)


def _materialize_meta_module(
    module: nn.Module,
    device_from_device_id: Optional[torch.device],
):
    # Run default meta device initialization
    materialization_device = device_from_device_id or torch.device(
        torch.cuda.current_device()
    )
    module.to_empty(device=materialization_device)
    try:
        with torch.no_grad():
            module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
        warnings.warn(
            "Unable to call `reset_parameters()` for module on meta "
            f"device with error {str(e)}. Please ensure your "
            "module implements a `reset_parameters()` method."
        )
        raise e


def _move_module_to_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    Moves ``module`` depending on ``device_from_device_id`` and its current
    device. This includes moving ignored modules' parameters.

    - If ``device_from_device_id`` is not ``None``, then this moves
    ``module`` to the device.
    - If ``device_from_device_id`` is ``None``, then this does not move
    ``module`` but warns the user if it is on CPU.

    Precondition: ``_check_single_device_module()``.
    """
    param = next(_get_orig_params(module, ignored_params), None)
    if param is None:
        return  # no original parameters to manage
    cpu_device = torch.device("cpu")

    if device_from_device_id is not None:
        if param.device == cpu_device:
            # NOTE: This includes moving ignored modules' parameters.
            module = module.to(device_from_device_id)
            # TODO: This is a temporary fix to move already-constructed
            # `FlatParameter`s back to CPU if needed. This is needed to
            # make CPU offload work with `device_id`.
            for submodule in module.modules():
                if (
                    isinstance(submodule, fsdp_file.FullyShardedDataParallel)
                    and submodule.cpu_offload.offload_params
                ):
                    for handle in submodule._handles:
                        handle.flat_param_to(torch.device("cpu"))
    elif param.device == cpu_device:
        _warn_cpu_init()


def _move_states_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    Precondition: ``_check_single_device_module()``.
    """
    if len(params) == 0 and len(buffers) == 0:
        return
    if len(params) > 0:
        current_device = params[0].device
    elif len(buffers) > 0:
        current_device = buffers[0].device
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        # Move the parameters and buffers like the `.data` code path in
        # `nn.Module._apply()`, which underlies `nn.Module.to()`
        for param in params:
            with torch.no_grad():
                param.data = param.to(device_from_device_id)
                if param.grad is not None:
                    param.grad.data = param.grad.to(device_from_device_id)
        for buffer in buffers:
            buffer.data = buffer.to(device_from_device_id)
    elif current_device == cpu_device:
        _warn_cpu_init()


def _warn_cpu_init():
    warnings.warn(
        "The passed-in `module` is on CPU and will thus have FSDP's sharding "
        "initialization run on CPU, which may be slower than on GPU. We "
        "recommend passing in the `device_id` argument for FSDP to move "
        "`module` to GPU for the sharding initialization. `module` must also "
        "be on GPU device to work with the `sync_module_states=True` flag "
        "since that requires GPU communication."
    )


def _get_compute_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
    rank: int,
) -> torch.device:
    """
    Determines and returns this FSDP instance's compute device. If the module
    is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA GPU device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """
    # If the module is on GPU already, then that GPU device has priority
    # over the current device
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type == "cuda":
        compute_device = param.device
    else:
        compute_device = torch.device("cuda", torch.cuda.current_device())
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(
            f"Inconsistent compute device and `device_id` on rank {rank}: "
            f"{compute_device} vs {device_from_device_id}"
        )
    return compute_device


# TODO: See how to deprecate!
def _sync_module_params_and_buffers(
    module: nn.Module,
    params: List[nn.Parameter],
    process_group: dist.ProcessGroup,
) -> None:
    """
    Synchronizes module states (i.e. parameters ``params`` and all
    not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

    Precondition: ``sync_module_states == True`` and ``self.process_group`` has
    been set.
    """
    _check_params_for_sync_module_states(params)
    module_states: List[torch.Tensor] = []
    for buffer in module.buffers():
        # Avoid re-synchronizing buffers in case of nested wrapping
        if not getattr(buffer, FSDP_SYNCED, False):
            setattr(buffer, FSDP_SYNCED, True)
            module_states.append(buffer.detach())
    module_states.extend(param.detach() for param in params)
    _sync_params_and_buffers(
        process_group,
        module_states,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )


def _sync_module_states(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    process_group: dist.ProcessGroup,
) -> None:
    _check_params_for_sync_module_states(params)
    # Assumes that each call to this method passes in disjoint `params` and
    # and `buffers` across calls, so there is no chance of re-synchronizing
    params_and_buffers = [param.detach() for param in params] + [
        buffer.detach() for buffer in buffers
    ]
    _sync_params_and_buffers(
        process_group,
        params_and_buffers,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )


def _check_params_for_sync_module_states(
    params: List[nn.Parameter],
) -> None:
    if params and any(param.device == torch.device("cpu") for param in params):
        raise ValueError(
            "The module has CPU parameters when `sync_module_states=True`, "
            "which only works when all parameters are on GPU. Please specify "
            "the `device_id` argument or move the module to GPU before passing "
            "into FSDP."
        )


def _get_orig_params(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Iterator[nn.Parameter]:
    """
    Returns an iterator over the original parameters in ``module``, ignoring
    the parameters in ``ignored_params``, any ``FlatParameter`` s (which may be
    present due to nested FSDP wrapping), and any original parameters already
    flattened (only relevant when ``use_orig_params=True``).
    """
    param_gen = module.parameters()
    try:
        while True:
            param = next(param_gen)
            if param not in ignored_params and not _is_fsdp_flattened(param):
                yield param
    except StopIteration:
        pass


def _check_orig_params_flattened(
    fsdp_module,
    ignored_params: Set[nn.Parameter],
) -> None:
    """
    Checks that all original parameters have been flattened and hence made
    invisible to ``named_parameters()`` for the module hierarchy rooted at
    ``fsdp_module``. This should be called as a sanity check after flattening
    the wrapped module's parameters.
    """
    for param_name, param in fsdp_module.named_parameters():
        if param not in ignored_params and not _is_fsdp_flattened(param):
            raise RuntimeError(
                f"Found an unflattened parameter: {param_name}; "
                f"{param.size()} {param.__class__}"
            )


def _get_default_comm_hook(sharding_strategy: ShardingStrategy):
    return (
        default_hooks.allreduce_hook
        if sharding_strategy == ShardingStrategy.NO_SHARD
        else default_hooks.reduce_scatter_hook
    )


def _get_default_comm_hook_state(
    process_group: dist.ProcessGroup,
) -> default_hooks.DefaultState:
    return default_hooks.DefaultState(process_group=process_group)
