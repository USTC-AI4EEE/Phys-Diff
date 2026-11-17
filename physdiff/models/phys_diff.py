"""Phys-Diff: anchor forecaster + latent residual diffusion with PIGA decoders.

Phys-Diff：anchor 预报器 + 基于 PIGA 解码器的潜空间残差扩散。
"""
import copy
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .diffusion import ResidualCodec, Schedule
from .encoder import ConditionEncoder, init_context_base, init_context_features
from .layers import DiffusionEmbedding, PIGATransformerDecoderLayer

__all__ = ['physical_baseline', 'TaskHeads', 'ForecastHead', 'PhysDiff']

N_LEAD = 4      # +6h, +12h, +18h, +24h / +6h、+12h、+18h、+24h 四个预报时效
N_VARS = 4      # lat, lon, max sustained wind, min sea-level pressure (all normalised) / 纬度、经度、最大持续风速、最低海平面气压（均已归一化）
NOISE_SEED_STRIDE = 104729


def physical_baseline(history: torch.Tensor) -> torch.Tensor:
    """Constant-velocity track extrapolation and persistence of intensity, (B, 4, 4) -> (B, 4, 4).

    路径按匀速外推、强度保持不变的基线预报，(B, 4, 4) -> (B, 4, 4)。
    """
    steps = torch.arange(1, N_LEAD + 1, device=history.device, dtype=history.dtype)[None, :, None]
    coords = history[:, -1:, :2] + (history[:, -1:, :2] - history[:, -2:-1, :2]) * steps
    return torch.cat([coords, history[:, -1:, 2:].expand(-1, N_LEAD, -1)], -1)


class TaskHeads(nn.Module):
    """Separate trajectory / wind / pressure output MLPs, initialised by splitting a shared MLP.

    路径 / 风速 / 气压各自独立的输出 MLP，由一个共享 MLP 拆分初始化。
    """

    def __init__(self, previous: nn.Sequential):
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, start, stop in [('trajectory', 0, 2), ('wind', 2, 3), ('pressure', 3, 4)]:
            layers = [copy.deepcopy(previous[i]) for i in range(3)]
            last = nn.Linear(previous[3].in_features, stop - start)
            with torch.no_grad():
                last.weight.copy_(previous[3].weight[start:stop])
                last.bias.copy_(previous[3].bias[start:stop])
            self.heads[name] = nn.Sequential(*layers, last)

    def forward(self, x):
        return torch.cat([self.heads[k](x) for k in ('trajectory', 'wind', 'pressure')], -1)


class ForecastHead(nn.Module):
    """PIGA Transformer decoder with one query per lead time.

    ``diffusion=False``: deterministic anchor head (2 layers) predicting a correction to the physical baseline.
    ``diffusion=True``: denoiser (3 layers) additionally conditioned on the noisy latent and the diffusion step.

    每个预报时效一个 query 的 PIGA Transformer 解码器。

    ``diffusion=False``：确定性 anchor 头（2 层），预测对物理基线的修正；``diffusion=True``：
    去噪器（3 层），额外以带噪潜变量和扩散步为条件。
    """

    def __init__(self, config: dict, diffusion: bool = False):
        super().__init__()
        d = config['model']['d_model']
        self.diffusion = diffusion
        self.lead = nn.Parameter(torch.randn(N_LEAD, d) * 0.02)
        self.anchor_proj = nn.Linear(N_VARS, d)
        self.layers = nn.ModuleList([PIGATransformerDecoderLayer(config) for _ in range(3 if diffusion else 2)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, N_VARS))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if diffusion:
            self.noisy_proj = nn.Linear(N_VARS, d)
            self.time_embedding = DiffusionEmbedding(d)

    def forward(self, memory, anchor, noisy=None, t=None):
        query = self.lead[None] + self.anchor_proj(anchor)
        if self.diffusion:
            query = query + self.noisy_proj(noisy) + self.time_embedding(t)[:, None]
        for layer in self.layers:
            query = layer(query, memory)
        return self.output(query)


class PhysDiff(nn.Module):
    """Phys-Diff tropical-cyclone forecaster.

    Forward pipeline::

        memory  = context(batch)                                  # ConditionEncoder
        anchor  = baseline(history) + anchor_scale * mean_head(memory, baseline)
        z_T     ~ N(0, I)  ->  DDIM(denoiser | condition_context(batch), anchor)  ->  z_0
        sample  = anchor + codec.decode(z_0) * residual_scale

    All tensors are in the normalised space of ``config['normalization']``; see :func:`physdiff.metrics.physical_values`.

    Training has three phases (see ``tools/train.py``): ``anchor`` -> ``base_diffusion`` -> ``alignment``.

    Phys-Diff 热带气旋预报模型。

    流程：编码条件得到 memory；anchor = 基线 + anchor_scale * mean_head；从 z_T ~ N(0, I) 经 DDIM 去噪得到 z_0；
    样本 = anchor + codec.decode(z_0) * residual_scale。所有张量都在 ``config['normalization']`` 的归一化空间中。
    训练分三个阶段（见 ``tools/train.py``）：``anchor`` -> ``base_diffusion`` -> ``alignment``。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # The construction below intentionally mirrors the order in which the released checkpoints were built:
        # it consumes the RNG identically, so ``torch.manual_seed(42); PhysDiff(cfg)`` reproduces the training
        # initialisation exactly. Do not reorder.
        # 下面的构建顺序刻意与发布 checkpoint 的构建顺序一致：随机数消耗完全相同，因此 ``torch.manual_seed(42); PhysDiff(cfg)`` 能精确复现训练初始化。
        # 请勿调整顺序。
        self.context = nn.Module()
        init_context_base(self.context, config)
        self.mean_head = ForecastHead(config)
        self.denoiser = ForecastHead(config, diffusion=True)
        self.scheduler = Schedule(config)
        self.register_buffer('anchor_scale', torch.ones(N_LEAD, N_VARS))
        self.register_buffer('residual_scale', torch.ones(N_LEAD, N_VARS))
        self.context = nn.Module()
        init_context_features(self.context, config)
        self.mean_head.output = TaskHeads(self.mean_head.output)
        self.denoiser.output = TaskHeads(self.denoiser.output)
        self.codec = ResidualCodec(config.get('story', {}).get('codec', 'asinh'))
        coarse = self.context
        self.context = ConditionEncoder(config)
        self.context.load_state_dict(coarse.state_dict(), strict=False)
        #: separate encoder for the diffusion branch, initialised from ``context`` before ``base_diffusion``
        #: 扩散分支的独立编码器，在 ``base_diffusion`` 之前由 ``context`` 初始化
        self.condition_context = copy.deepcopy(self.context)
        self.register_buffer('mean_signal_power', self.scheduler.alphas_cumprod.mean().clone())

    # -- housekeeping ---------------------------------------------------------------------------------------------
    # -- 辅助功能 --
    def _apply(self, fn, *args, **kwargs):
        # keep the (non-Module) schedule tensors on the same device as the parameters
        # 让（非 Module 的）调度张量与参数位于同一设备
        super()._apply(fn, *args, **kwargs)
        self.scheduler.to(self.anchor_scale.device)
        return self

    def set_phase(self, phase: str) -> None:
        """Freeze everything except the sub-modules trained in ``phase``.

        冻结除 ``phase`` 阶段所训练子模块之外的全部参数。
        """
        if phase not in ('anchor', 'base_diffusion', 'alignment'):
            raise ValueError(f'Unknown phase {phase!r}')
        self.requires_grad_(False)
        self.eval()
        if phase == 'anchor':
            self.context.requires_grad_(True).train()
            self.mean_head.requires_grad_(True).train()
        else:
            self.denoiser.requires_grad_(True).train()
            if phase == 'alignment':
                self.condition_context.requires_grad_(True).train()

    # -- deterministic branch -------------------------------------------------------------------------------------
    # -- 确定性分支 --
    def anchor(self, **batch):
        """Return the deterministic anchor forecast (B, 4, 4) and the encoder memory.

        返回确定性 anchor 预报 (B, 4, 4) 和编码器 memory。
        """
        memory = self.context(**batch)
        baseline = physical_baseline(batch['history'])
        return baseline + self.anchor_scale * self.mean_head(memory, baseline), memory

    def diffusion_memory(self, **batch):
        return self.condition_context(**batch)

    # -- training -------------------------------------------------------------------------------------------------
    # -- 训练 --
    def training_prediction(self, target, anchor, memory, alignment: bool = False):
        """Diffusion training step: returns the one-step reconstruction and a dict of loss terms.

        扩散训练步：返回一步重建结果和损失项字典。
        """
        clean = self.codec.encode((target - anchor) / self.residual_scale)
        t = torch.randint(0, self.scheduler.num_timesteps, (len(target),), device=target.device)
        noise = torch.randn_like(clean)
        noisy = self.scheduler.q_sample(clean, t, noise)
        velocity = self.denoiser(memory, anchor, noisy, t)
        a, _ = self.scheduler.coefficients(t, clean.shape)
        losses = {'value': F.mse_loss(velocity, self.scheduler.velocity_target(clean, t, noise))}
        prediction = anchor + self.codec.decode(self.scheduler.predict_original_sample(noisy, t, velocity)) \
            * self.residual_scale
        if alignment:
            losses['reconstruction_weight'] = a.square() / self.mean_signal_power
            count = self.config['story']['alignment_members']
            z = torch.randn(len(anchor), count // 2, N_LEAD, N_VARS, device=anchor.device, dtype=anchor.dtype)
            was_training = self.denoiser.training
            self.denoiser.eval()
            try:
                losses['free_running_prediction'] = self.sample_context(
                    anchor, memory, noise=torch.cat([z, -z], 1), steps=self.config['story']['alignment_steps']).mean(1)
            finally:
                self.denoiser.train(was_training)
        return prediction, losses

    # -- sampling -------------------------------------------------------------------------------------------------
    # -- 采样 --
    @staticmethod
    def seeded_noise(packed_ids: Sequence[int], members: int = 50, seed: int = 42, device='cpu') -> torch.Tensor:
        """Random Gaussian initial noise, reproducible per forecast window: ``seed + 104729 * packed_id``.

        随机高斯初始噪声，每个预报窗口可复现：种子为 ``seed + 104729 * packed_id``。
        """
        draws = []
        for identifier in packed_ids:
            generator = torch.Generator(device=device).manual_seed(seed + NOISE_SEED_STRIDE * int(identifier))
            draws.append(torch.randn(members, N_LEAD, N_VARS, generator=generator, device=device))
        return torch.stack(draws)

    def sample_context(self, anchor, memory, noise: Optional[torch.Tensor] = None, packed_ids=None, members: int = 50,
                       seed: int = 42, steps: Optional[int] = None):
        """DDIM sampling given precomputed anchor/memory. Returns (B, members, 4, 4).

        给定预先计算的 anchor/memory 做 DDIM 采样，返回 (B, members, 4, 4)。
        """
        if noise is None:
            if packed_ids is None:
                raise ValueError('Pass either `noise` or `packed_ids`')
            noise = self.seeded_noise(packed_ids, members, seed, anchor.device)
        if noise.ndim == 3:
            noise = noise.unsqueeze(0).expand(len(anchor), -1, -1, -1)
        count = noise.shape[1]
        z = noise.reshape(-1, N_LEAD, N_VARS).contiguous()
        a = anchor.repeat_interleave(count, 0)
        m = memory.repeat_interleave(count, 0)
        times = self.scheduler.ddim_timesteps(steps)
        for i, value in enumerate(times):
            t = torch.full((len(z),), value, dtype=torch.long, device=z.device)
            previous = times[i + 1] if i + 1 < len(times) else -1
            z = self.scheduler.step(self.denoiser(m, a, z, t), z, t, previous)
        values = (a + self.codec.decode(z) * self.residual_scale).reshape(len(anchor), count, N_LEAD, N_VARS)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite residual samples')
        return values

    @torch.no_grad()
    def sample(self, *, noise=None, packed_ids=None, members: int = 50, seed: int = 42, steps=None, **batch):
        """Generate an ensemble ``(B, members, 4, 4)`` for a batch from :meth:`physdiff.data.PreparedData.inputs`.

        为 :meth:`physdiff.data.PreparedData.inputs` 给出的批次生成集合预报 ``(B, members, 4, 4)``。
        """
        anchor, _ = self.anchor(**batch)
        memory = self.diffusion_memory(**batch)
        return self.sample_context(anchor, memory, noise=noise, packed_ids=packed_ids, members=members, seed=seed,
                                   steps=steps)
