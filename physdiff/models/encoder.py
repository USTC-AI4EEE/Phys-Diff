"""Multimodal condition encoder (IBTrACS history + ERA5 / FengWu environmental fields).

多模态条件编码器（IBTrACS 历史 + ERA5 / FengWu 环境场）。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .layers import PatchMerging, SwinTransformerBlock

__all__ = ['FINE_CHANNELS', 'init_context_base', 'init_context_features', 'FineCoreSwin', 'ConditionEncoder']

#: Indices (into the 69 ERA5/FengWu channels) of the 11 variables kept in the high-resolution core crop.
#: 高分辨率核心裁剪中保留的 11 个变量在 69 个 ERA5/FengWu 通道中的索引。
FINE_CHANNELS = [0, 1, 2, 3, 24, 26, 33, 40, 46, 53, 66]

N_CHANNELS = 69     # 4 surface + 5 variables x 13 pressure levels / 4 个地面变量 + 5 个变量 x 13 个气压层
N_HISTORY = 4       # observed steps (t-18h ... t) / 观测步（t-18h ... t）
N_LEAD = 4          # forecast steps (+6h ... +24h) / 预报步（+6h ... +24h）
N_STEPS = N_HISTORY + N_LEAD


def init_context_base(module: nn.Module, config: dict) -> None:
    """Coarse context encoder: history MLP, environmental CNN, extrema MLP and a Transformer encoder.

    Implemented as a function that populates ``module`` so that :class:`PhysDiff` can reproduce the exact
    parameter-creation order (and therefore the seeded initialisation) used to train the released checkpoints.

    粗尺度上下文编码器：历史 MLP、环境场 CNN、极值 MLP 和 Transformer 编码器。

    以函数形式填充 ``module``，使 :class:`PhysDiff` 能复现训练发布 checkpoint 时的参数创建顺序（从而复现带种子的初始化）
    。
    """
    d = config['model']['d_model']
    module.history = nn.Sequential(nn.Linear(16, d), nn.SiLU(), nn.Linear(d, d), nn.LayerNorm(d))
    module.weather = nn.Sequential(
        nn.Conv2d(N_CHANNELS, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
        nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.SiLU(),
        nn.Conv2d(96, d, 3, stride=2, padding=1), nn.GroupNorm(8, d), nn.SiLU())
    module.extrema = nn.Sequential(nn.Linear(2 * N_CHANNELS, d), nn.SiLU(), nn.Linear(d, d))
    module.time = nn.Parameter(torch.randn(N_STEPS, d) * 0.02)
    module.space = nn.Parameter(torch.randn(9, d) * 0.02)
    module.scale = nn.Parameter(torch.randn(2, d) * 0.02)
    module.source = nn.Parameter(torch.randn(2, d) * 0.02)
    layer = nn.TransformerEncoderLayer(d, 4, 3 * d, dropout=0.05, activation='gelu', batch_first=True, norm_first=True)
    module.encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(d), enable_nested_tensor=False)
    module.register_buffer('coord_std', torch.tensor(config['normalization']['coord_std']))


def init_context_features(module: nn.Module, config: dict) -> None:
    """Base encoder + GRU history branch, Swin environmental branch, geometry and physics features.

    基础编码器 + GRU 历史分支、Swin 环境分支、几何与物理特征。
    """
    init_context_base(module, config)
    module.options = config.get('story', {})
    d = config['model']['d_model']
    module.history_gru = nn.GRU(16, d // 2, 2, dropout=0.05, batch_first=True)
    module.history_gru_output = nn.Linear(d // 2, d)
    module.swin_embed = nn.Conv2d(N_CHANNELS, 64, 1)
    module.swin_blocks = nn.ModuleList([
        SwinTransformerBlock(64, (10, 10), 4, window_size=5, shift_size=s, drop=0.05, attn_drop=0.05) for s in (0, 2)])
    module.swin_output = nn.Conv2d(64, d, 1)
    module.geometry = nn.Linear(3, d)
    module.physics = nn.Sequential(nn.Linear(14, d), nn.SiLU(), nn.Linear(d, d))
    for layer in (module.history_gru_output, module.swin_output, module.geometry, module.physics[-1]):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    stats = config['weather_normalization']
    module.register_buffer('weather_mean', torch.tensor(stats['channel_means']))
    module.register_buffer('weather_std', torch.tensor(stats['channel_stds']))
    norm = config['normalization']
    module.wind_mean, module.wind_std = norm['wind_mean'], norm['wind_std']
    module.pressure_mean, module.pressure_std = norm['pressure_mean'], norm['pressure_std']


class FineCoreSwin(nn.Module):
    """Two-stage Swin encoder for the 11-channel, 40x40 storm-core crop; outputs a 3x3 feature map.

    针对 11 通道 40x40 风暴核心裁剪的两级 Swin 编码器，输出 3x3 特征图。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.patch = nn.Conv2d(len(FINE_CHANNELS), 64, 2, stride=2)
        self.position = nn.Parameter(torch.randn(1, 400, 64) * 0.02)
        self.first = nn.ModuleList([
            SwinTransformerBlock(64, (20, 20), 2, window_size=5, shift_size=s, drop=0.05, attn_drop=0.05) for s in (0, 2)])
        self.merge = PatchMerging((20, 20), 64)
        self.second = nn.ModuleList([
            SwinTransformerBlock(128, (10, 10), 4, window_size=5, shift_size=s, drop=0.05, attn_drop=0.05) for s in (0, 2)])
        self.norm = nn.LayerNorm(128)
        self.output = nn.Conv2d(128, d_model, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2) + self.position
        for layer in self.first:
            x = layer(x)
        x = self.merge(x)
        for layer in self.second:
            x = layer(x)
        x = self.norm(x).transpose(1, 2).reshape(-1, 128, 10, 10)
        return self.output(F.adaptive_avg_pool2d(x, (3, 3)))


class ConditionEncoder(nn.Module):
    """Encode one forecast window into a memory sequence of ``4 + 8*2*9 + 8 = 156`` tokens of width ``d_model``.

    Inputs (normalised, see ``docs/DATA.md``)::

        history         (B, 4, 4)               observed [lat, lon, wind, pressure], relative to ``references``
        references      (B, 2)                  reference lat/lon at forecast time t
        weather         (B, 8, 2, 69, 10, 10)   ERA5 (steps 0-3) / FengWu (steps 4-7) crops at two scales
        extrema         (B, 8, 69, 2)           per-channel extrema of the crops
        future_centers  (B, 4, 2)               FengWu-tracked vortex centres for the 4 lead times
        center_quality  (B, 4)                  1 if a vortex was detected, else 0
        fine_weather    (B, 8, 11, 40, 40)      high-resolution storm-core crop

    将一个预报窗口编码为 ``4 + 8*2*9 + 8 = 156`` 个宽度为 ``d_model`` 的记忆 token。

    输入均已归一化（见 ``docs/DATA.md``）：history 为相对 references 的观测 [lat, lon, wind, pressure]；
    references 为预报时刻 t 的参考经纬度；weather 为两种尺度的 ERA5（第 0-3 步）/ FengWu（第 4-7 步）
    裁剪；extrema 为裁剪的逐通道极值；future_centers 为 FengWu 追踪的 4 个时效涡旋中心；center_quality 表示是否检测到涡旋；
    fine_weather 为高分辨率风暴核心裁剪。
    """

    def __init__(self, config: dict):
        super().__init__()
        init_context_features(self, config)
        d = config['model']['d_model']
        self.fine_era5 = FineCoreSwin(d)
        self.fine_fengwu = FineCoreSwin(d)
        self.fine_physics = nn.Sequential(nn.Linear(8, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.fine_physics[-1].weight)
        nn.init.zeros_(self.fine_physics[-1].bias)
        grid = torch.linspace(-1, 1, 40)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        self.register_buffer('fine_ring', (x.square() + y.square() > 0.7 ** 2).float(), persistent=False)

    # -- hand-crafted physical summaries --------------------------------------------------------------------------
    # -- 手工构造的物理摘要特征 --
    def _observed(self, history):
        pressure = history[:, -1, 3] * self.pressure_std + self.pressure_mean
        wind = (history[:, -1, 2] * self.wind_std + self.wind_mean) * (1852 / 3600)   # kt -> m/s / 节 -> 米/秒
        return pressure, wind

    def fine_summary(self, history, fine):
        """8 storm-core descriptors per step (pressure deficit, wind excess, warm core, shear, ...).

        每步 8 个风暴核心描述量（气压亏损、风速超出、暖心、风切变等）。
        """
        mu = self.weather_mean[FINE_CHANNELS][None, None, :, None, None]
        std = self.weather_std[FINE_CHANNELS][None, None, :, None, None] + 1e-08
        raw = fine * std + mu
        p = raw[:, :, 3] / 100
        wind = torch.linalg.vector_norm(raw[:, :, :2], dim=2)
        observed_p, observed_w = self._observed(history)
        ring = self.fine_ring
        ring_p = (p * ring).sum((-2, -1)) / ring.sum()
        temperature = raw[:, :, 10]
        temp_ring = (temperature * ring).sum((-2, -1)) / ring.sum()
        u_shear = (raw[:, :, 6] - raw[:, :, 7]).mean((-2, -1))
        v_shear = (raw[:, :, 8] - raw[:, :, 9]).mean((-2, -1))
        p_min = p.amin((-2, -1))
        return torch.stack([
            (p_min - observed_p[:, None]) / 20,
            (p.mean((-2, -1)) - observed_p[:, None]) / 20,
            (ring_p - p_min) / 20,
            (wind.amax((-2, -1)) - observed_w[:, None]) / 30,
            wind.square().mean((-2, -1)).sqrt() / 30,
            fine[:, :, 5].mean((-2, -1)),
            (temperature[:, :, 16:24, 16:24].mean((-2, -1)) - temp_ring) / 5,
            (u_shear.square() + v_shear.square() + 1e-08).sqrt() / 30,
        ], -1)

    def physical_features(self, history, weather, extrema):
        """14 large-scale environmental descriptors per step.

        每步 14 个大尺度环境描述量。
        """
        raw = weather[:, :, 1] * (self.weather_std[None, None, :, None, None] + 1e-08) \
            + self.weather_mean[None, None, :, None, None]
        raw_mean = raw.mean((-2, -1))
        wind = torch.linalg.vector_norm(raw[:, :, :2], dim=2)
        observed_pressure, observed_wind = self._observed(history)
        pmin = (extrema[:, :, 3, 0] * (self.weather_std[3] + 1e-08) + self.weather_mean[3]) / 100
        shear_u = raw_mean[:, :, 33] - raw_mean[:, :, 40]
        shear_v = raw_mean[:, :, 46] - raw_mean[:, :, 53]
        return torch.stack([
            (pmin - observed_pressure[:, None]) / 20,
            (raw_mean[:, :, 3] / 100 - observed_pressure[:, None]) / 20,
            (wind.amax((-2, -1)) - observed_wind[:, None]) / 30,
            wind.mean((-2, -1)) / 30,
            shear_u / 20,
            shear_v / 20,
            torch.sqrt(shear_u.square() + shear_v.square() + 1e-08) / 30,
            raw_mean[:, :, 40] / 20,
            raw_mean[:, :, 53] / 20,
            weather[:, :, 1, 24].mean((-2, -1)),
            weather[:, :, 1, 26].mean((-2, -1)),
            weather[:, :, 1, 66].mean((-2, -1)),
            weather[:, :, 1, 11].mean((-2, -1)),
            weather[:, :, 1, 3].mean((-2, -1)) - weather[:, :, 0, 3].mean((-2, -1)),
        ], -1)

    # -- forward ---------------------------------------------------------------------------------------------------
    # -- 前向计算 --
    def forward(self, history, references, weather, extrema, future_centers, center_quality, fine_weather):
        if not self.options['use_fengwu']:
            # w/o FengWu: persist the last observed (ERA5) step over the forecast horizon.
            # 无 FengWu：在预报时段内沿用最后一个观测（ERA5）步。
            weather = torch.cat([weather[:, :4], weather[:, 3:4].expand(-1, 4, -1, -1, -1, -1)], 1)
            extrema = torch.cat([extrema[:, :4], extrema[:, 3:4].expand(-1, 4, -1, -1)], 1)
            future_centers = history[:, -1:, :2].expand(-1, 4, -1)
            center_quality = torch.zeros_like(center_quality)
            fine_weather = torch.cat([fine_weather[:, :4], fine_weather[:, 3:4].expand(-1, 4, -1, -1, -1)], 1)
        b, d = len(history), self.time.shape[-1]

        # storm-core branch
        # 风暴核心分支
        era = self.fine_era5(fine_weather[:, :4].reshape(-1, 11, 40, 40)).reshape(b, 4, -1, 3, 3)
        future = self.fine_fengwu(fine_weather[:, 4:].reshape(-1, 11, 40, 40)).reshape(b, 4, -1, 3, 3)
        central = torch.cat([era, future], 1)
        native = torch.stack([torch.zeros_like(central), central], 2).reshape(b * 16, -1, 3, 3)
        summary = self.fine_physics(self.fine_summary(history, fine_weather))

        # history branch (MLP + GRU on position/velocity/acceleration features)
        # 历史分支（基于位置/速度/加速度特征的 MLP + GRU）
        delta = torch.cat([torch.zeros_like(history[:, :1]), history[:, 1:] - history[:, :-1]], 1)
        acc = torch.cat([torch.zeros_like(delta[:, :1]), delta[:, 1:] - delta[:, :-1]], 1)
        radians = (history[..., :2] * self.coord_std + references[:, None]) * (math.pi / 180)
        features = torch.cat([history, delta, acc, radians.sin(), radians.cos()], -1)
        hist = self.history(features) + self.history_gru_output(self.history_gru(features)[0]) + self.time[:4]

        # environmental branch (CNN + Swin) -> 3x3 tokens per step and scale
        # 环境分支（CNN + Swin）-> 每步每个尺度 3x3 个 token
        image = weather.reshape(b * 16, N_CHANNELS, 10, 10)
        z = self.weather(image) + native
        sw = self.swin_embed(image).flatten(2).transpose(1, 2)
        for layer in self.swin_blocks:
            sw = layer(sw)
        sw = sw.transpose(1, 2).reshape(-1, 64, 10, 10)
        z = z + self.swin_output(F.adaptive_avg_pool2d(sw, (3, 3)))
        z = z.flatten(2).transpose(1, 2).reshape(b, 8, 2, 9, d)

        # step embeddings: time + data source (ERA5 / FengWu) + geometry + physics descriptors
        # 步嵌入：时间 + 数据来源（ERA5 / FengWu）+ 几何 + 物理描述量
        source = torch.cat([self.source[0].expand(4, -1), self.source[1].expand(4, -1)])
        temporal = self.time + source
        points = torch.cat([history[..., :2], future_centers], 1)
        quality = torch.cat([torch.ones_like(center_quality), center_quality], 1)
        extra = torch.zeros(b, 8, d, device=z.device, dtype=z.dtype)
        extra = extra + self.geometry(torch.cat([points, quality[..., None]], -1))
        extra = extra + self.physics(self.physical_features(history, weather, extrema))
        z = z + (temporal[None] + extra)[:, :, None, None] + self.scale[None, None, :, None] + self.space[None, None, None]
        extremes = self.extrema(extrema.flatten(-2)) + summary + temporal + extra
        return self.encoder(torch.cat([hist, z.flatten(1, 3), extremes], 1))
