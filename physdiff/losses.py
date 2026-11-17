"""Training objective: lead-normalised task losses + two-group uncertainty weighting, and PIGA gradient routing.

训练目标：按预报时效归一化的任务损失 + 两组不确定性加权，以及 PIGA 梯度路由。
"""
import math

import torch
from torch import nn

from .models.layers import PIGAModule

__all__ = ['UncertaintyLoss', 'routed_backward']


class UncertaintyLoss(nn.Module):
    """Uncertainty-weighted sum of the diffusion (v-MSE) and reconstruction (track + wind + pressure) losses.

    Per-lead task errors are divided by ``task_scales`` (the anchor's training MAE at the start of each phase), and
    the two groups are balanced with learnable log-variances (Kendall et al., 2018) after the ``anchor`` phase.

    扩散损失（v-MSE）与重建损失（路径 + 风速 + 气压）的不确定性加权和。

    各时效的任务误差除以 ``task_scales``（每个阶段开始时 anchor 的训练 MAE）；``anchor`` 阶段之后，
    两组损失用可学习的对数方差平衡（Kendall et al., 2018）。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.log_var_diffusion = nn.Parameter(torch.tensor(0.))
        self.log_var_reconstruction = nn.Parameter(torch.tensor(0.))
        self.register_buffer('task_scales', torch.ones(4, 3))

    def per_lead(self, prediction, target, references):
        """Absolute errors (B, 4 leads, 3 tasks): local-Cartesian track distance, wind, pressure.

        绝对误差 (B, 4 个时效, 3 个任务)：局部笛卡尔坐标下的路径距离、风速、气压。
        """
        std = prediction.new_tensor(self.config['normalization']['coord_std'])
        difference = (prediction[..., :2] - target[..., :2]) * std
        lat = (target[..., 0] * std[0] + references[:, None, 0]) * (math.pi / 180)
        d = torch.stack([difference[..., 0], difference[..., 1] * lat.cos()], -1)
        return torch.stack([torch.linalg.vector_norm(d, dim=-1) / std[0],
                            (prediction[..., 2] - target[..., 2]).abs(),
                            (prediction[..., 3] - target[..., 3]).abs()], -1)

    def forward(self, prediction, target, references, diffusion, phase):
        scales = self.task_scales.clamp_min(1e-3)
        physical = self.per_lead(prediction, target, references) / scales
        if phase == 'alignment':
            physical = physical * diffusion['reconstruction_weight']
            free = self.per_lead(diffusion['free_running_prediction'], target, references) / scales
            physical = physical + self.config['story']['alignment_weight'] * free
        coord, wind, pressure = physical.mean((0, 1)).unbind()
        reconstruction = coord + wind + pressure
        score = diffusion['value']
        if phase == 'anchor':
            total = reconstruction + score
        else:
            d = self.log_var_diffusion.clamp(-6, 6)
            r = self.log_var_reconstruction.clamp(-6, 6)
            total = .5 * (-d).exp() * score + .5 * (-r).exp() * reconstruction + .5 * (d + r)
        return dict(total_loss=total, diffusion_loss=score, coord_loss=coord, msw_loss=wind, mlsp_loss=pressure)


def routed_backward(model: nn.Module, losses: dict) -> None:
    """Back-propagate ``total_loss``, but let each PIGA task projection receive only its own task gradient.

    Diffusion gradients reach every projection; the reconstruction gradient of task *k* reaches only
    ``<k>_mapping``. All other parameters receive the ordinary gradient of ``total_loss``.

    反向传播 ``total_loss``，但每个 PIGA 任务投影只接收本任务的梯度。

    扩散梯度传到所有投影；任务 *k* 的重建梯度只传到 ``<k>_mapping``；其余参数接收 ``total_loss`` 的普通梯度。
    """
    tasks = dict(coord_loss='coord_mapping', msw_loss='msw_mapping', mlsp_loss='mslp_mapping')
    names = ['diffusion_loss', *tasks]
    groups = {name: [] for name in tasks}
    for module in model.modules():
        if isinstance(module, PIGAModule):
            for task, projection in tasks.items():
                groups[task].extend(p for p in getattr(module, projection).parameters() if p.requires_grad)
    coefficients = dict(zip(names, torch.autograd.grad(losses['total_loss'], [losses[n] for n in names],
                                                       retain_graph=True)))
    routed = []
    for task, parameters in groups.items():
        allowed = coefficients['diffusion_loss'] * losses['diffusion_loss'] + coefficients[task] * losses[task]
        gradients = torch.autograd.grad(allowed, parameters, retain_graph=True, allow_unused=True)
        routed.extend(zip(parameters, gradients))
    losses['total_loss'].backward()
    for parameter, gradient in routed:
        parameter.grad = gradient
