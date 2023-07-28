# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP PipelineEngine."""

from deepspeed.utils import instrument_w_nvtx
from deepspeed.runtime.pipe.engine import PipelineEngine, schedule
from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum

from msamp.deepspeed.runtime.engine import MSAMPDeepSpeedEngine


class MSAMPPipelineEngine(MSAMPDeepSpeedEngine, PipelineEngine):
    """Pipeline engine supports pipeline+ZeRO-2+BF16."""
    def _exec_reduce_grads(self):
        """Reduce gradients across pipeline stages."""
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.bfloat16_enabled():
                if self.zero_optimization_stage() == ZeroStageEnum.disabled:
                    self._bf16_reduce_grads()
                elif self.zero_optimization_stage() == ZeroStageEnum.optimizer_states:
                    self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
                else:
                    raise NotImplementedError('PP+BF16 only work for ZeRO Stage 1')
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        """Allreduce gradients across pipeline stages."""
        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        # ZeRO stage >= 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage(
            ) == ZeroStageEnum.optimizer_states and hasattr(self.optimizer, 'reduce_gradients'):
                self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    PipelineEngine._INSTRUCTION_MAP.update({schedule.ReduceGrads: _exec_reduce_grads})
