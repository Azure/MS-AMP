# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP adamw module."""

import math
from typing import Dict, List, Union

import torch
from torch import Tensor
import torch.distributed as dist

from msamp.optim import LBAdamWBase
from msamp.common.tensor import ScalingMeta, ScalingTensor
from msamp.common.dtype import Floating, Dtypes
import msamp_adamw


class LBAdamW(LBAdamWBase):
    r"""Implements AdamW algorithm with cuda.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        maximize: bool = False,
        exp_avg_dtype=torch.uint8,
        exp_avg_sq_dtype=torch.float16,
        tensor_scale=True,
    ):
        """Constructor. See class docstring for details."""
        self.tensor_scale = tensor_scale
        super().__init__(
            params,
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            maximize=maximize,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype
        )

    def adamw_fn(   # noqa: C901
        self,
        params: List[Union[Tensor, ScalingTensor]],
        grads: List[Union[Tensor, ScalingTensor]],
        exp_avgs: List[Dict[str, Union[Tensor, ScalingTensor]]],
        exp_avg_sqs: List[Dict[str, Union[Tensor, ScalingTensor]]],
        max_exp_avg_sqs: List[Union[Tensor, ScalingTensor]],
        state_steps: List[int],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool,
    ):
        """Functional API that performs AdamW algorithm computation.

        Args:
            params (List[Union[Tensor, ScalingTensor]]): list of parameters.
            grads (List[Union[Tensor, ScalingTensor]]): list of gradients.
            exp_avgs (List[Dict[str, Union[Tensor, ScalingTensor]]]): list of exponential moving average of gradient.
            exp_avg_sqs (List[Dict[str, Union[Tensor, ScalingTensor]]]): list of exponential moving average of
                squared gradient.
            max_exp_avg_sqs (List[Union[Tensor, ScalingTensor]]): list of maximum exponential moving average of
                squared gradient.
            state_steps (List[int]): list of steps, currently not supported.
            amsgrad (bool): whether to use the AMSGrad variant of this algorithm, currently only support True.
            beta1 (float): coefficient used for computing running averages of gradient.
            beta2 (float): coefficient used for computing running averages of squared gradient.
            lr (float): learning rate.
            weight_decay (float): weight decay coefficient.
            eps (float): term added to the denominator to improve numerical stability.
            maximize (bool): maximize the params based on the objective, instead of minimizing.
        """
        if amsgrad:
            raise ValueError('Only amsgrad=False is supported for now.')

        if len(exp_avgs) > 0 and exp_avgs[0].dtype != torch.float32:
            _exp_avg_amaxs = [exp_avg.meta.amax for exp_avg in exp_avgs]
            _exp_avg_sq_amaxs = [exp_avg_sq.meta.amax for exp_avg_sq in exp_avg_sqs]
            _exp_avg_inv_factors = [1.0 / exp_avg.meta.scale for exp_avg in exp_avgs]
            _exp_avg_sq_inv_factors = [1.0 / exp_avg_sq.meta.scale for exp_avg_sq in exp_avg_sqs]
            torch._foreach_zero_(_exp_avg_amaxs)
            torch._foreach_zero_(_exp_avg_sq_amaxs)

            for i, param in enumerate(params):
                param, grad = param.float(), grads[i].float() if not maximize else -grads[i].float()

                # Perform step weight decay
                if weight_decay != 0:
                    if self.use_adam:
                        grad = grad.add(param, alpha=weight_decay)
                    else:
                        param.mul_(1 - lr * weight_decay)
                assert param.is_contiguous()
                assert grad.is_contiguous()

                msamp_adamw.adamw_fp8_stage1_compute(
                    param, grad, exp_avgs[i].value, _exp_avg_inv_factors[i], _exp_avg_amaxs[i], beta1,
                    exp_avg_sqs[i].value, _exp_avg_sq_inv_factors[i], _exp_avg_sq_amaxs[i], beta2, eps, state_steps[i],
                    lr, self.bias_correction
                )
                if isinstance(params[i], ScalingTensor):
                    params[i].copy_(param.cast(params[i].qtype, meta=params[i].meta))

            if self.tensor_scale:
                amaxs, sq_amaxs = torch.cat(_exp_avg_amaxs), torch.cat(_exp_avg_sq_amaxs)
                ones = amaxs.new_ones((1, ))
                _new_exp_avg_factors = ScalingMeta.compute_scaling_factor(
                    amaxs, ones, Floating.fp_maxs[exp_avgs[0].dtype], 0
                ).tolist()
                _new_exp_avg_sq_factors = ScalingMeta.compute_scaling_factor(
                    sq_amaxs, ones, Floating.fp_maxs[exp_avg_sqs[0].dtype], 0
                ).tolist()

            for i, param in enumerate(params):
                grad = grads[i].float() if not maximize else -grads[i].float()
                exp_avgs[i].meta.scale = _new_exp_avg_factors[i] if self.tensor_scale else 1.0
                exp_avg_sqs[i].meta.scale = _new_exp_avg_sq_factors[i] if self.tensor_scale else 1.0
                # update state
                msamp_adamw.adamw_fp8_stage2_compute(
                    grad, exp_avgs[i].value, _exp_avg_inv_factors[i], exp_avgs[i].meta.scale, beta1,
                    exp_avg_sqs[i].value, _exp_avg_sq_inv_factors[i], exp_avg_sqs[i].meta.scale, beta2, state_steps[i],
                    self.bias_correction
                )
        else:
            # float
            for i, param in enumerate(params):
                param, grad = param.float(), grads[i].float() if not maximize else -grads[i].float()
                exp_avg_value, exp_avg_sq_value = exp_avgs[i], exp_avg_sqs[i]

                # Perform step weight decay
                if weight_decay != 0:
                    if self.use_adam:
                        grad = grad.add(param, alpha=weight_decay)
                    else:
                        param.mul_(1 - lr * weight_decay)

                if self.bias_correction:
                    bias_correction1 = 1 - beta1**state_steps[i]
                    bias_correction2 = 1 - beta2**state_steps[i]
                else:
                    bias_correction1 = bias_correction2 = 1.0

                step_size = lr / bias_correction1

                # Decay the first and second moment running average coefficient
                # exp_avg = beta1 * exp_avg  + (1 - beta1) * grad
                exp_avg_value.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_sq_value.mul_(beta2).add_(torch.square(grad), alpha=1 - beta2)
                # sqrt(exp_avg_sq / bias_correction2) + eps
                denom = (exp_avg_sq_value.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # param = param - step_size * (exp_avg / denom)
                # param.addcdiv_(exp_avg, denom, value=-step_size)
                param.add_(exp_avg_value / denom, alpha=-step_size)

                if isinstance(params[i], ScalingTensor):
                    params[i].copy_(param.cast(params[i].qtype, meta=params[i].meta))


class FSDPAdamW(LBAdamWBase):
    """Implements AdamW algorithm for FSDP."""
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        maximize: bool = False,
        exp_avg_dtype=torch.uint8,
        exp_avg_sq_dtype=torch.float16,
        tensor_scale=True,
    ):
        """Constructor. See LBAdamW class docstring for details."""
        self.tensor_scale = tensor_scale
        super().__init__(
            params,
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            maximize=maximize,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype
        )

        self.original_params = []
        self.master_weights = []

        for group in self.param_groups:
            params = []
            for param in group['params']:
                if param is None:
                    continue

                self.original_params.append(param)
                if hasattr(param, '_meta') and param._meta is not None and param.numel() > 0:
                    dtype = Dtypes.qtype_to_dtype[param._meta.qtype]
                    param = ScalingTensor(param.view(dtype), param._meta)
                    master_weight = param.cast(Dtypes.kfloat16)
                    master_weight.requires_grad = True
                    self.master_weights.append(master_weight)
                    params.append(master_weight)
                else:
                    self.master_weights.append(None)
                    params.append(param)

            group['params'] = params

    def zero_grad(self, set_to_none=False):
        """Zero gradients."""
        for param in self.original_params:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

    def step(self):
        """Performs a single optimization step."""
        # Set gradient of master weight.
        for i, param in enumerate(self.original_params):
            if self.master_weights[i] is not None:
                grad_meta = param._grad_meta
                dtype = Dtypes.qtype_to_dtype[grad_meta.qtype]
                self.master_weights[i].grad = ScalingTensor(param.grad.view(dtype), grad_meta)
                param.grad = None

        # call step() to update master weight
        super().step()

        # Copy master weight to weight
        for i, param in enumerate(self.original_params):
            if hasattr(param, '_meta') and param._meta is not None:
                hp_data = None
                if param.numel() == 0:
                    param._meta.amax[0].zero_()
                else:
                    hp_data = self.master_weights[i].float()
                    param._meta.amax[0] = hp_data.abs().max()

                dist.all_reduce(param._meta.amax[0], op=dist.ReduceOp.MAX)
                param._meta.reset_scaling_factor()
                if param.numel() > 0:
                    with ScalingMeta.in_time_scaling_context(False):
                        data = hp_data.cast(param._meta.qtype, param._meta, False) \
                                .value.view(torch.float32)
                    param.data.copy_(data)
                else:
                    param._meta.scale_inv.data.copy_(torch.reciprocal(param._meta.scale))
