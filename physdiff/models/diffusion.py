"""Latent residual codec and the cosine v-prediction DDIM schedule.

潜空间残差编解码器与余弦 v-prediction DDIM 噪声调度。
"""
import math

import torch
from torch import nn

__all__ = ['ResidualCodec', 'Schedule']


class ResidualCodec(nn.Module):
    """Invertible map between forecast residuals (B, 4 leads, 4 vars) and the diffusion latent space.

    ``encode``: subtract the median, apply ``asinh(x / s)`` with a per-element scale fitted by maximum likelihood,
    centre, and whiten per lead with a block-diagonal Cholesky factor (track | wind | pressure).

    预报残差 (B, 4 个时效, 4 个变量) 与扩散潜空间之间的可逆映射。

    ``encode``：减去中位数，以最大似然拟合的逐元素尺度做 ``asinh(x / s)``，中心化，再按时效用分块对角 Cholesky 因子（路径 | 风速 | 气压）
    白化。
    """

    def __init__(self, method: str = 'asinh'):
        super().__init__()
        self.method = method
        self.register_buffer('location', torch.zeros(4, 4))
        self.register_buffer('asinh_scale', torch.ones(4, 4))
        self.register_buffer('latent_center', torch.zeros(4, 4))
        self.register_buffer('cholesky', torch.eye(4).repeat(4, 1, 1))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def _store(self, location, scale, center, matrix):
        self.location.copy_(location)
        self.asinh_scale.copy_(scale)
        self.latent_center.copy_(center)
        self.cholesky.copy_(matrix)
        self.fitted.fill_(True)

    def fit(self, residual: torch.Tensor) -> None:
        """Fit the codec on training residuals of shape (N, 4, 4).

        在形状为 (N, 4, 4) 的训练残差上拟合编解码器。
        """
        y = residual.detach().cpu().double()
        location = y.median(0).values if self.method != 'identity' else torch.zeros_like(y[0])
        shifted = y - location
        scale = torch.ones_like(location)
        if self.method == 'asinh':
            log_scale = nn.Parameter(torch.zeros_like(location))
            optimizer = torch.optim.Adam([log_scale], lr=0.04)
            with torch.enable_grad():
                for _ in range(80):
                    optimizer.zero_grad(set_to_none=True)
                    s = log_scale.clamp(-2.3, 2.3).exp()
                    z = torch.asinh(shifted / s)
                    std = z.std(0).clamp_min(0.0001)
                    nll = (std.log() + s.log() + 0.5 * torch.log1p((shifted / s).square()).mean(0)).mean()
                    nll.backward()
                    optimizer.step()
            scale = log_scale.detach().clamp(-2.3, 2.3).exp()
        raw = torch.asinh(shifted / scale) if self.method == 'asinh' else shifted
        center = raw.mean(0)
        centered = raw - center
        matrix = torch.zeros(4, 4, 4, dtype=torch.float64)
        for lead in range(4):
            cov = centered[:, lead].T @ centered[:, lead] / len(y)
            # keep only the (lat, lon) block and the wind / pressure diagonal
            # 只保留 (lat, lon) 块以及风速 / 气压的对角元
            cov[0:2, 2:] = 0
            cov[2:, 0:2] = 0
            cov[2, 3] = 0
            cov[3, 2] = 0
            matrix[lead] = torch.linalg.cholesky(cov + torch.eye(4) * 1e-05)
        if self.method == 'identity':
            location.zero_()
            center.zero_()
            matrix = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
        self._store(location, scale, center, matrix)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            return x
        z = x - self.location
        if self.method == 'asinh':
            z = torch.asinh(z / self.asinh_scale)
        return torch.linalg.solve_triangular(self.cholesky, (z - self.latent_center).unsqueeze(-1), upper=False).squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            return z
        x = (self.cholesky @ z.unsqueeze(-1)).squeeze(-1) + self.latent_center
        if self.method == 'asinh':
            x = self.asinh_scale * torch.sinh(x)
        return x + self.location


class Schedule:
    """Cosine noise schedule (Nichol & Dhariwal, 2021) with v-prediction and deterministic DDIM (eta = 0).

    余弦噪声调度（Nichol & Dhariwal, 2021），采用 v-prediction 和确定性 DDIM（eta = 0）
    。
    """

    def __init__(self, config: dict):
        options = config['model']['ddpm']
        self.num_timesteps = options['num_timesteps']
        self.sampling_steps = options['sampling_steps']
        assert options['prediction_type'] == 'v_prediction'
        assert options['sampling_method'] == 'ddim' and options['ddim_eta'] == 0
        t = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, dtype=torch.float64)
        curve = torch.cos(((t / self.num_timesteps) + .008) / 1.008 * math.pi * .5).square()
        curve = curve / curve[0]
        beta = (1 - curve[1:] / curve[:-1]).clamp(1e-6, .999).float().double()
        alpha = torch.cumprod(1 - beta, 0).clamp_min(torch.finfo(beta.dtype).tiny)
        self.alphas_cumprod = alpha.float()
        self.sqrt_alphas_cumprod = alpha.sqrt().float()
        self.sqrt_one_minus_alphas_cumprod = (1 - alpha).clamp_min(1e-8).sqrt().float()

    def to(self, device) -> 'Schedule':
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    @staticmethod
    def _extract(values, t, shape):
        return values.gather(-1, t).reshape(len(t), *([1] * (len(shape) - 1)))

    def coefficients(self, t, shape):
        """Return ``(sqrt(alpha_bar_t), sqrt(1 - alpha_bar_t))`` broadcastable to ``shape``.

        返回可广播到 ``shape`` 的 ``(sqrt(alpha_bar_t), sqrt(1 - alpha_bar_t))``。
        """
        return (self._extract(self.sqrt_alphas_cumprod, t, shape),
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, shape))

    def q_sample(self, clean, t, noise):
        a, s = self.coefficients(t, clean.shape)
        return a * clean + s * noise

    def velocity_target(self, clean, t, noise):
        a, s = self.coefficients(t, clean.shape)
        return a * noise - s * clean

    def predict_original_sample(self, noisy, t, velocity):
        a, s = self.coefficients(t, noisy.shape)
        return a * noisy - s * velocity

    def ddim_timesteps(self, steps=None):
        return torch.linspace(0, self.num_timesteps - 1, steps or self.sampling_steps).round().long().flip(0).tolist()

    def step(self, velocity, noisy, t, previous):
        """One DDIM update from timestep ``t`` to ``previous`` (``-1`` returns the predicted clean sample).

        从时间步 ``t`` 到 ``previous`` 的一次 DDIM 更新（``-1`` 返回预测的干净样本）。
        """
        clean = self.predict_original_sample(noisy, t, velocity)
        if previous == -1:
            return clean
        a = self._extract(self.alphas_cumprod, t, noisy.shape)
        prior = self.alphas_cumprod[previous]
        epsilon = (noisy - a.sqrt() * clean) / (1 - a).sqrt().clamp_min(1e-8)
        result = prior.sqrt() * clean + (1 - prior).clamp_min(0).sqrt() * epsilon
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite DDIM state')
        return result
