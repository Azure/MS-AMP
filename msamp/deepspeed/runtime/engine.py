import deepspeed
from deepspeed.runtime.engine import *

from .fp8.fused_optimizer import FP8_Optimizer
from .config import MSAMP_ADAM_OPTIMIZER, MSAMP_ADAMW_OPTIMIZER
from .constants import FP8

import msamp
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.optim.optimizer import LBOptimizer
from msamp.common.dtype import Dtypes


def split_half_float_double_sparse(tensors):
    supported_types = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor", "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor",
        "msamp.common.tensor.tensor.ScalingTensor",
        SparseTensor.type()
    ]

    for t in tensors:
        assert t.type() in supported_types, f"attempting to reduce an unsupported grad type: {t.type()}"

    buckets = []
    for i, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


deepspeed.runtime.engine.split_half_float_double_sparse = split_half_float_double_sparse


_origin_DeepSpeedEngine = deepspeed.runtime.engine.DeepSpeedEngine


class MSAMPDeepSpeedEngine(_origin_DeepSpeedEngine):
    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res

        if self.fp16_enabled():
            return torch.float16

        return torch.float32

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is not None:
            if isinstance(client_optimizer, tuple(self._supported_optims())):
                client_optimizer.param_groups[:] = [
                    pg for pg in client_optimizer.param_groups if len(pg["params"]) != 0
                ]
                log_dist(
                    "Removing param_group that has no 'params' in the client Optimizer",
                    ranks=[0])

                basic_optimizer = client_optimizer
                log_dist('Using client Optimizer as basic optimizer', ranks=[0])
            else:
                basic_optimizer = client_optimizer(model_parameters)
                log_dist('Using client callable to create basic optimizer', ranks=[0])
        else:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            log_dist(
                f"Using DeepSpeed Optimizer param name {self.optimizer_name()} as basic optimizer",
                ranks=[0])

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        log_dist("DeepSpeed Basic Optimizer = {}".format(
            basic_optimizer.__class__.__name__),
                 ranks=[0])

        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)
        use_fp8 = False
        if isinstance(basic_optimizer, LBOptimizer):
            use_fp8 = True

        if optimizer_wrapper == ZERO_OPTIMIZATION or (use_fp8 and optimizer_wrapper == BFLOAT16):
            self.optimizer = self._configure_zero_optimizer(basic_optimizer, use_fp8=use_fp8)
        elif use_fp8:
            self.optimizer = self._configure_fp8_optimizer(basic_optimizer, optimizer_wrapper)
        elif optimizer_wrapper == AMP:
            amp_params = self.amp_params()
            log_dist(f"Initializing AMP with these params: {amp_params}", ranks=[0])
            model, self.optimizer = amp.initialize(
                self.module, basic_optimizer, **amp_params
            )
            self._set_client_model(model)
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif optimizer_wrapper == FP16:
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        elif optimizer_wrapper == BFLOAT16:
            self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        log_dist("DeepSpeed Final Optimizer = {}".format(self.optimizer_name()),
                 ranks=[0])

        self.compression_scheduler = self._configure_compression_scheduler()
        self.quantizer = self._configure_quantization()

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if optimizer_parameters is None:
            optimizer_parameters = {}
        # print(optimizer_parameters.keys())
        if "max_grad_norm" in optimizer_parameters.keys():
            raise ValueError(
                "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
            )

        if self.optimizer_name() in [ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

            # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
            effective_adam_w_mode = self.optimizer_name(
            ) == ADAMW_OPTIMIZER or adam_w_mode

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = torch.optim.Adam(model_parameters,
                                                 **optimizer_parameters)
                else:
                    optimizer = torch.optim.AdamW(model_parameters,
                                                  **optimizer_parameters)
            else:
                if self.zero_use_cpu_optimizer():
                    if self.optimizer_name() == ADAGRAD_OPTIMIZER:
                        from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                        optimizer = DeepSpeedCPUAdagrad(model_parameters,
                                                        **optimizer_parameters)
                    else:
                        from deepspeed.ops.adam import DeepSpeedCPUAdam
                        optimizer = DeepSpeedCPUAdam(model_parameters,
                                                     **optimizer_parameters,
                                                     adamw_mode=effective_adam_w_mode)
                else:
                    from deepspeed.ops.adam import FusedAdam

                    optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )

        elif self.optimizer_name() in [MSAMP_ADAM_OPTIMIZER, MSAMP_ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)
            effective_adam_w_mode = self.optimizer_name(
            ) == MSAMP_ADAMW_OPTIMIZER or adam_w_mode
            from msamp.optim import LBAdam as MSAMP_Adam, LBAdamW as MSAMP_AdamW, DSAdam

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = MSAMP_Adam(model_parameters,
                                          **optimizer_parameters)
                else:
                    optimizer = MSAMP_AdamW(model_parameters,
                                           **optimizer_parameters)
            else:
                if self.zero_use_cpu_optimizer():
                    raise NotImplementedError('Not implemented on ZeRO CPU Optimizer')
                else:
                    optimizer = DSAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )

        elif self.optimizer_name() == LAMB_OPTIMIZER:
            from deepspeed.ops.lamb import FusedLamb

            optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

            optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(
                    f"Currently the convergence of 1-bit Adam is only verified under FP16"
                )
        elif self.optimizer_name() == ZERO_ONE_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "0/1 Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

            optimizer = ZeroOneAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(
                    f'Currently the convergence of 0/1 Adam is only verified under FP16')
        elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

            optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(
                    f"Currently the convergence of 1-bit Lamb is only verified under FP16"
                )
        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
        return optimizer

    def _configure_fp8_optimizer(self, optimizer, optimizer_wrapper):
        initial_dynamic_scale = self.initial_dynamic_scale()
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()
        if APEX_INSTALLED:
            fused_opts = (apex.optimizers.FusedAdam, FusedAdam)
        else:
            fused_opts = FusedAdam

        if optimizer_wrapper == FP16 and self.dynamic_loss_scale():
            log_dist("Creating fp8 optimizer with dynamic loss scale", ranks=[0])
            timers = self.timers if self.wall_clock_breakdown() else None
            optimizer = FP8_Optimizer(
                optimizer,
                deepspeed=self,
                dynamic_loss_scale=True,
                initial_dynamic_scale=initial_dynamic_scale,
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                timers=timers,
                has_moe_layers=self.has_moe_layers,
            )
        else:
            log_dist(
                "Creating fp8 optimizer with static loss scale: {}".format(
                    self.loss_scale()),
                ranks=[0],
            )
            loss_scale = self.loss_scale()
            if loss_scale == 0:
                loss_scale = 1
            optimizer = FP8_Optimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=loss_scale,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                has_moe_layers=self.has_moe_layers,
            )

        return optimizer

    def _configure_zero_optimizer(self, optimizer, use_fp8=False):
        zero_stage = self.zero_optimization_stage()
        timers = self.timers if self.wall_clock_breakdown() else None

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        if self.zero_legacy_stage1():
            raise Exception(
                "The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. Please downgrade to a version less than 0.5.9 if you need to use this deprecated version of ZeRO."
            )

        if zero_stage <= ZeroStageEnum.gradients:
            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), "zero stage {} requires an optimizer".format(zero_stage)

            log_dist('Creating fp16 ZeRO stage {} optimizer'.format(zero_stage),
                     ranks=[0])
            # Overlap and contiguous grads are meaningless in stage 1 and are ignored
            if zero_stage == ZeroStageEnum.optimizer_states:
                overlap_comm = False
                round_robin_gradients = False

            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning(
                        "Pipeline parallelism does not support overlapped communication, will be disabled."
                    )
                    overlap_comm = False
            if use_fp8:
                raise NotImplementedError('fp8 ZeRO is not supported yet')
            else:
                zero_t = DeepSpeedZeroOptimizer
            optimizer = zero_t(
                optimizer,
                self.param_names,
                timers=timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.data_parallel_group,
                expert_parallel_group=self.expert_parallel_group
                if self.has_moe_layers else None,
                expert_data_parallel_group=self.expert_data_parallel_group
                if self.has_moe_layers else None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                cpu_offload=self.zero_cpu_offload(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZeroStageEnum.gradients,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=self.has_moe_layers,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(
                ),
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint())

        elif zero_stage == ZeroStageEnum.weights:
            assert not self.has_moe_layers, "MoE not supported with Stage 3"
            if isinstance(optimizer, DummyOptim):
                log_dist("Creating ZeRO Offload", ranks=[0])
                optimizer = DeepSpeedZeRoOffload(
                    self.module,
                    timers=timers,
                    ds_config=self.config,
                    overlap_comm=self.zero_overlap_comm(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    offload_param_config=self.zero_offload_param(),
                    mpu=self.mpu)
            else:
                log_dist('Creating fp16 ZeRO stage {} optimizer'.format(zero_stage),
                         ranks=[0])
                from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
                optimizer = DeepSpeedZeroOptimizer_Stage3(
                    self.module,
                    optimizer,
                    timers=timers,
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.data_parallel_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    sub_group_size=self.zero_sub_group_size(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps(),
                    aio_config=self.aio_config(),
                    communication_data_type=self.communication_data_type)

        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

        return optimizer

    @instrument_w_nvtx
    def backward(self,
                 loss,
                 allreduce_gradients=True,
                 release_loss=False,
                 retain_graph=False,
                 scale_wrt_gas=True):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
            retain_graph: bool, default: false
                forward on user defined choice of retain_graph
        """

        see_memory_usage("Engine before backward", force=self.memory_breakdown())

        if self.scale_wrt_gas is not None:
            scale_wrt_gas = self.scale_wrt_gas

        if not allreduce_gradients:
            logger.warning(
                f"Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed"
            )

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1 and scale_wrt_gas:
            loss = self._scale_loss_by_gas(loss.float())

        # Log training Loss
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(
                        f"Train/Samples/train_loss",
                        loss.mean().item() * self.gradient_accumulation_steps(),
                        self.global_samples,
                    )]
                    self.monitor.write_events(self.summary_events)

        self._start_timers(self.engine_timers.backward_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use backward"

        self._start_timers(self.engine_timers.backward_inner_timers)

        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary(
            )
            self.optimizer.backward(loss, retain_graph=retain_graph)
        elif isinstance(self.optimizer, FP8_Optimizer):
            self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.amp_enabled():
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss,
                                self.optimizer,
                                delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        elif self.fp16_enabled():
            if self.eigenvalue_enabled():
                self.optimizer.backward(loss, create_graph=True, retain_graph=True)
            else:
                self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.bfloat16_enabled():
            self.optimizer.backward(loss)
        else:
            if self.eigenvalue_enabled():
                loss.backward(create_graph=True, retain_graph=True)
            else:
                loss.backward(retain_graph=retain_graph)

        self._stop_timers(self.engine_timers.backward_inner_timers)

        self._start_timers(self.engine_timers.backward_reduce_timers)

        if allreduce_gradients and self.enable_backward_allreduce:
            # Traditional code path that allreduces the module parameter grads
            self.allreduce_gradients()

        self._stop_timers(self.engine_timers.backward_reduce_timers)

        self._stop_timers(self.engine_timers.backward_timers)

        if release_loss:
            # loss.data = None
            pass

        see_memory_usage("Engine after backward", force=self.memory_breakdown())

        return loss

    def fp8_allreduce_bucket(self, bucket, dp_group):
        # TODO: assert dp_group == global_dp_group
        from msamp.common.tensor import TensorDist
        # in msamp, a.data = b.data. It is not a copy!
        if self.gradient_average:
            TensorDist.all_reduce_avg(bucket)
        else:
            TensorDist.all_reduce_sum(bucket)

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        from msamp.nn import model_state
        model_state.ready_to_all_reduce_grads = False
        return super().allreduce_gradients(bucket_size)

    def allreduce_and_copy(self, small_bucket, dp_group):
        if len(small_bucket) == 0:
            return
        if isinstance(small_bucket[0], ScalingTensor):
            # ScalingTensor all reduce
            self.fp8_allreduce_bucket(small_bucket, dp_group)
            return
        allreduced = self.allreduce_bucket(small_bucket, dp_group)
        for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def _get_gradients_for_reduction(self):
        non_expert_grads = []
        expert_grads = {}
        if self.has_moe_layers:
            for key in self.expert_data_parallel_group.keys():
                expert_grads[key] = []

        for param_name, param in self.module.named_parameters():
            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                if isinstance(param, ScalingTensor):
                    param.grad = ScalingTensor(torch.zeros(param.size(),
                                               dtype=torch.uint8,
                                               device=param.device), ScalingMeta(Dtypes.kfloat_8e4m3))
                else:
                    param.grad = torch.zeros(param.size(),
                                             dtype=param.dtype,
                                             device=param.device)

            grad_data = param.grad.data
            if param_name in self.sparse_tensor_module_names or grad_data.is_sparse:
                # Call param.grad without data to avoid problem with setting of updated grads
                grad_data = SparseTensor(param.grad)

            if is_moe_param(param):
                expert_grads[param.group_name].append(grad_data)
            else:
                non_expert_grads.append(grad_data)

        return non_expert_grads, expert_grads
