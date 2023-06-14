# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP PipelineEngine."""

from deepspeed.utils import instrument_w_nvtx
from deepspeed.runtime.pipe.engine import PipelineEngine, schedule
from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum

from msamp.nn import model_state
from msamp.deepspeed.runtime.engine import MSAMPDeepSpeedEngine


class MSAMPPipelineEngine(MSAMPDeepSpeedEngine, PipelineEngine):
    """Pipeline engine supports pipeline+ZeRO-2+BF16."""
    def _exec_reduce_grads(self):
        """Reduce gradients across pipeline stages."""
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.bfloat16_enabled():
                if self.zero_optimization_stage() < ZeroStageEnum().gradients:
                    self._bf16_reduce_grads()
                elif self.zero_optimization_stage() == ZeroStageEnum().gradients:
                    self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
                else:
                    raise NotImplementedError("PP+BF16 only work for ZeRO Stage 2")
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        """Allreduce gradients across pipeline stages."""
        model_state.ready_to_all_reduce_grads = False
        super().allreduce_gradients(bucket_size=bucket_size)

    PipelineEngine._INSTRUCTION_MAP.update({schedule.ReduceGrads: _exec_reduce_grads})
