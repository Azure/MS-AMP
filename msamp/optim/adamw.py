# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP adamw module."""

import math
from typing import List, Union

import torch
from torch import Tensor

from msamp.optim import LBAdamWBase
from msamp.common.tensor import ScalingMeta, ScalingTensor
from msamp.common.dtype import Floating
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

    def _get_state_tensor(self, state, dtype):
        """Get state dict from tensor and dtype.

        Convert state to dtype since the caller in LBAdamWBase always pass state as a float32 tensor.

        Args:
            state (Tensor): state tensor.
            dtype (torch.dtype): dtype of the state tensor.

        Return:
            dict: state dict contains state tensor and other scaling meta data if not float32.
        """
        state = state.to(dtype=dtype)

        if dtype != torch.float32:
            return dict(state=state, factor=1.0, amax=state.new_zeros((1, ), dtype=torch.float32))
        return dict(state=state)

    def adamw_fn(
        self, params: List[Union[Tensor, ScalingTensor]], grads: List[Union[Tensor, ScalingTensor]],
        exp_avgs: List[Union[Tensor, ScalingTensor]], exp_avg_sqs: List[Union[Tensor, ScalingTensor]],
        max_exp_avg_sqs: List[Union[Tensor, ScalingTensor]], state_steps: List[int], *, amsgrad: bool, beta1: float,
        beta2: float, lr: float, weight_decay: float, eps: float, maximize: bool
    ):
        """Functional API that performs AdamW algorithm computation.

        Args:
            params (List[Union[Tensor, ScalingTensor]]): list of parameters.
            grads (List[Union[Tensor, ScalingTensor]]): list of gradients.
            exp_avgs (List[Union[Tensor, ScalingTensor]]): list of exponential moving average of gradient.
            exp_avg_sqs (List[Union[Tensor, ScalingTensor]]): list of exponential moving average of squared gradient.
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

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # Perform stepweight decay
            # FP32/16 Tensor * float

            grad = grad.float()
            param = param.float()

            if weight_decay != 0:
                if self.use_adam:
                    grad = grad.add(param, alpha=weight_decay)
                else:
                    param.mul_(1 - lr * weight_decay)

            assert param.is_contiguous()
            assert grad.is_contiguous()

            exp_avg_value = exp_avg['state']
            exp_avg_sq_value = exp_avg_sq['state']

            if exp_avg_value.dtype != torch.float32:
                exp_avg_factor, exp_avg_amax = exp_avg['factor'], exp_avg['amax']
                exp_avg_sq_factor, exp_avg_sq_amax = exp_avg_sq['factor'], exp_avg_sq['amax']
                exp_avg_inv_factor = 1.0 / exp_avg_factor
                exp_avg_sq_inv_factor = 1.0 / exp_avg_sq_factor

                # update params
                exp_avg_amax.zero_()
                exp_avg_sq_amax.zero_()

                msamp_adamw.adamw_fp8_stage1_compute(
                    param, grad, exp_avg_value, exp_avg_inv_factor, exp_avg_amax, beta1, exp_avg_sq_value,
                    exp_avg_sq_inv_factor, exp_avg_sq_amax, beta2, eps, step, lr, self.bias_correction
                )

                # update scaling factor
                one = exp_avg_amax.new_ones((1, ))

                if self.tensor_scale:
                    fp_max = Floating.fp_maxs[exp_avg_value.dtype]
                    exp_avg['factor'] = ScalingMeta.compute_scaling_factor(exp_avg_amax, one, fp_max, 0).item()
                    fp_max = Floating.fp_maxs[exp_avg_sq_value.dtype]
                    exp_avg_sq['factor'] = ScalingMeta.compute_scaling_factor(exp_avg_sq_amax, one, fp_max, 0).item()
                else:
                    exp_avg['factor'] = 1.0
                    exp_avg_sq['factor'] = 1.0

                # update state
                msamp_adamw.adamw_fp8_stage2_compute(
                    grad, exp_avg_value, exp_avg_inv_factor, exp_avg['factor'], beta1, exp_avg_sq_value,
                    exp_avg_sq_inv_factor, exp_avg_sq['factor'], beta2, step, self.bias_correction
                )
            else:
                # float
                if self.bias_correction:
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
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


class LBAdam(LBAdamW):
    """Implements Adam algorithm with weight decay fix."""
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize: bool = False,
        *args,
        **kwargs
    ):
        """Constructor. See LBAdamW class docstring for details."""
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            *args,
            **kwargs
        )
        self.use_adam = True
