# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP adamw module."""

from typing import Dict, List, Union

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

        _grads = [grad.float() if not maximize else -grad.float() for grad in grads]
        _params = [param.float() for param in params]
        if weight_decay != 0:
            if self.use_adam:
                torch._foreach_add_(_grads, _params, alpha=weight_decay)
            else:
                torch._foreach_mul_(_params, 1 - lr * weight_decay)

        if len(exp_avgs) > 0 and exp_avgs[0]['state'].dtype != torch.float32:
            _exp_avg_amaxs = [exp_avg['amax'] for exp_avg in exp_avgs]
            _exp_avg_sq_amaxs = [exp_avg_sq['amax'] for exp_avg_sq in exp_avg_sqs]
            _exp_avg_inv_factors = [1.0 / exp_avg['factor'] for exp_avg in exp_avgs]
            _exp_avg_sq_inv_factors = [1.0 / exp_avg_sq['factor'] for exp_avg_sq in exp_avg_sqs]
            torch._foreach_zero_(_exp_avg_amaxs)
            torch._foreach_zero_(_exp_avg_sq_amaxs)

            for i in range(len(_params)):
                assert _params[i].is_contiguous()
                assert _grads[i].is_contiguous()
                msamp_adamw.adamw_fp8_stage1_compute(
                    _params[i], _grads[i], exp_avgs[i]['state'], _exp_avg_inv_factors[i], _exp_avg_amaxs[i], beta1,
                    exp_avg_sqs[i]['state'], _exp_avg_sq_inv_factors[i], _exp_avg_sq_amaxs[i], beta2, eps,
                    state_steps[i], lr, self.bias_correction
                )
            if self.tensor_scale:
                amaxs, sq_amaxs = torch.cat(_exp_avg_amaxs), torch.cat(_exp_avg_sq_amaxs)
                ones = amaxs.new_ones((1, ))
                _new_exp_avg_factors = ScalingMeta.compute_scaling_factor(
                    amaxs, ones, Floating.fp_maxs[exp_avgs[0]['state'].dtype], 0
                ).tolist()
                _new_exp_avg_sq_factors = ScalingMeta.compute_scaling_factor(
                    sq_amaxs, ones, Floating.fp_maxs[exp_avg_sqs[0]['state'].dtype], 0
                ).tolist()
            for i in range(len(_params)):
                exp_avgs[i]['factor'] = _new_exp_avg_factors[i] if self.tensor_scale else 1.0
                exp_avg_sqs[i]['factor'] = _new_exp_avg_sq_factors[i] if self.tensor_scale else 1.0
                # update state
                msamp_adamw.adamw_fp8_stage2_compute(
                    _grads[i], exp_avgs[i]['state'], _exp_avg_inv_factors[i], exp_avgs[i]['factor'], beta1,
                    exp_avg_sqs[i]['state'], _exp_avg_sq_inv_factors[i], exp_avg_sqs[i]['factor'], beta2,
                    state_steps[i], self.bias_correction
                )
                if isinstance(params[i], ScalingTensor):
                    params[i].copy_(_params[i].cast(params[i].qtype, meta=params[i].meta))
        else:
            # Refer to _multi_tensor_adamw in torch.optim.adamw
            # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/optim/adamw.py#L445

            # Decay the first and second moment running average coefficient
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, _grads, alpha=1 - beta1)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, _grads, _grads, 1 - beta2)

            if self.bias_correction:
                bias_correction1 = torch._foreach_pow(beta1, state_steps)
                bias_correction2 = torch._foreach_pow(beta2, state_steps)
                torch._foreach_sub_(bias_correction1, 1)
                torch._foreach_sub_(bias_correction2, 1)
                torch._foreach_neg_(bias_correction1)
                torch._foreach_neg_(bias_correction2)
            else:
                bias_correction1 = bias_correction2 = [torch.ones((1, )) for _ in range(len(state_steps))]
            step_size = torch._foreach_div(bias_correction1, lr)
            torch._foreach_reciprocal_(step_size)
            torch._foreach_neg_(step_size)

            # sqrt(exp_avg_sq / bias_correction2) + eps
            bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

            torch._foreach_addcdiv_(_params, exp_avgs, denom, step_size)
