# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP pure-python AdamW optimizer."""

import math
from typing import List, Union

import torch
from torch import Tensor

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.optim import LBOptimizer


class LBAdamWBase(LBOptimizer):
    r"""Implements AdamW algorithm.

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
        exp_avg_dtype(torch.dtype, optional): dtype for the first moment (default: torch.uint8)
        exp_avg_sq_dtype(torch.dtype, optional): dtype for the second moment (default: torch.float16)

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
    ):
        """Constructor. See class docstring for details."""
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)

        self.bias_correction = bias_correction
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.use_adam = False

        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set state of the optimizer.

        Args:
            state (dict): state of the optimizer.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    def _get_state_tensor(self, tensor, dtype):
        """Get the state tensor of the given dtype.

        Args:
            tensor (torch.Tensor): the orignal tensor.
            dtype (torch.dtype): the dtype of the state tensor.

        Returns:
            ScalingTensor: the state tensor.
        """
        if dtype != torch.float32:
            if dtype not in [torch.uint8, torch.int8, torch.float16]:
                raise TypeError('Unsupported dtype: {}'.format(dtype))
            qtype = Dtypes.dtype_to_qtype[dtype]
            state = tensor.cast(qtype)
        return state

    def _update_lp_tensor(self, lp, hp):
        """Update the low precision tensor with the value high precision tensor.

        Args:
            lp (torch.Tensor or ScalingTensor): the low precision tensor.
            hp (torch.Tensor): the high precision tensor.
        """
        # hp -> lp
        if id(lp) == id(hp):
            return
        if isinstance(lp, ScalingTensor):
            lp.copy_(hp.cast(lp.qtype))
        else:
            lp.copy_(hp)

    @torch.no_grad()
    def lb_step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                params_with_grad.append(p)
                grads.append(grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = self._get_state_tensor(
                        torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32),
                        self.exp_avg_dtype
                    )

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = self._get_state_tensor(
                        torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32),
                        self.exp_avg_sq_dtype
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = self._get_state_tensor(
                            torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32),
                            self.state_dtype
                        )

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            self.adamw_fn(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
            )

        return loss

    def adamw_fn(
        self, params: List[Union[Tensor, ScalingTensor]], grads: List[Union[Tensor, ScalingTensor]],
        exp_avgs: List[Union[Tensor, ScalingTensor]], exp_avg_sqs: List[Union[Tensor, ScalingTensor]],
        max_exp_avg_sqs: List[Union[Tensor, ScalingTensor]], state_steps: List[int], *, amsgrad: bool, beta1: float,
        beta2: float, lr: float, weight_decay: float, eps: float, maximize: bool
    ):
        """Functional API that performs AdamW algorithm computation."""
        if amsgrad:
            raise ValueError('Only amsgrad=False is supported for now.')

        for i, param_lp in enumerate(params):
            grad_lp = grads[i] if not maximize else -grads[i]
            exp_avg_lp = exp_avgs[i]
            exp_avg_sq_lp = exp_avg_sqs[i]
            step = state_steps[i]

            grad = grad_lp.float()
            param = param_lp.float()

            if weight_decay != 0:
                if self.use_adam:
                    grad = grad.add(param, alpha=weight_decay)
                else:
                    param.mul_(1 - lr * weight_decay)

            assert param.is_contiguous()
            assert grad.is_contiguous()

            exp_avg = exp_avg_lp.float()
            exp_avg_sq = exp_avg_sq_lp.float()

            if self.bias_correction:
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
            else:
                bias_correction1 = bias_correction2 = 1.0

            step_size = lr / bias_correction1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

            # Parameter
            self._update_lp_tensor(param_lp, param)
            # momentum 1
            self._update_lp_tensor(exp_avg_lp, exp_avg)
            # momentum 2
            self._update_lp_tensor(exp_avg_sq_lp, exp_avg_sq)
