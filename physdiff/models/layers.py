"""Building blocks: Swin Transformer blocks, the PIGA module and the PIGA decoder layer.

基础模块：Swin Transformer 块、PIGA 模块和 PIGA 解码层。
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    'window_partition', 'window_reverse', 'WindowAttention', 'SwinTransformerBlock', 'PatchMerging',
    'PIGAModule', 'PIGATransformerDecoderLayer', 'DiffusionEmbedding',
]


# ---------------------------------------------------------------------------
# Swin Transformer (Liu et al., 2021)
# Swin Transformer（Liu 等, 2021）
# ---------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * num_windows, window_size, window_size, C).

    将 (B, H, W, C) 划分为窗口 (B * num_windows, window_size, window_size, C)。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`.

    :func:`window_partition` 的逆操作。
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    带相对位置偏置的窗口多头自注意力。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(window_size[0]), torch.arange(window_size[1])], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        area = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(area, area, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """Swin block operating on a flattened ``(B, H*W, C)`` token sequence.

    作用于展平 ``(B, H*W, C)`` token 序列的 Swin 块。
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must be in [0, window_size)'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), act_layer(), nn.Dropout(drop), nn.Linear(hidden, dim),
                                 nn.Dropout(drop))
        self.register_buffer('attn_mask', self._shift_mask() if self.shift_size > 0 else None)

    def _shift_mask(self) -> torch.Tensor:
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        windows = self.attn(windows, mask=self.attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """2x2 patch merging: (B, H*W, C) -> (B, H/2*W/2, 2C).

    2x2 patch 合并：(B, H*W, C) -> (B, H/2*W/2, 2C)。
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


# ---------------------------------------------------------------------------
# Physics-Inspired Gated Attention (PIGA)
# 物理启发门控注意力（PIGA）
# ---------------------------------------------------------------------------
TASKS = ('coord', 'msw', 'mslp')  # trajectory, maximum sustained wind, minimum sea-level pressure / 路径、最大持续风速、最低海平面气压


class PIGAModule(nn.Module):
    """Disentangle features into trajectory / wind / pressure sub-spaces and let each task attend to the others.

    For every task ``i`` the module computes ``q_i = W_q f_i`` and ``k_i, v_i`` from the concatenation of the other
    two task features, then fuses the cross-task context with a learned sigmoid gate::

        f_i' = LayerNorm((1 - g_i) * f_i + g_i * Attn(q_i, k_i, v_i))

    The three updated sub-features are concatenated and projected back to ``d_model``.

    将特征解耦为路径 / 风速 / 气压子空间，并让每个任务关注其余两个任务。

    对每个任务 i，q_i = W_q f_i，k_i、v_i 由另外两个任务特征拼接后得到；跨任务上下文通过可学习的 sigmoid 门融合：
    f_i' = LayerNorm((1 - g_i) * f_i + g_i * Attn(q_i, k_i, v_i))。
    三个更新后的子特征拼接并投影回 ``d_model``。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.d_model = config['model']['d_model']
        self.d_sub = config['model']['piga']['d_sub']
        self.gate_mlp_dims = config['model']['piga']['gate_mlp_dims']
        # NOTE: attribute names and creation order define the checkpoint keys and seeded initialisation.
        # 注意：属性名和创建顺序决定 checkpoint 的键名和带种子的初始化。
        for task in TASKS:
            setattr(self, f'{task}_mapping', nn.Conv1d(self.d_model, self.d_sub, kernel_size=1))
        for task in TASKS:
            setattr(self, f'{task}_q_proj', nn.Linear(self.d_sub, self.d_sub))
            setattr(self, f'{task}_k_proj', nn.Linear(2 * self.d_sub, self.d_sub))
            setattr(self, f'{task}_v_proj', nn.Linear(2 * self.d_sub, self.d_sub))
        for task in TASKS:
            setattr(self, f'{task}_gate_mlp', self._build_gate_mlp())
        self.output_conv = nn.Conv1d(3 * self.d_sub, self.d_model, kernel_size=1)
        for task in TASKS:
            setattr(self, f'norm_{task}', nn.LayerNorm(self.d_sub))
        self._init_parameters()

    def _build_gate_mlp(self) -> nn.Module:
        layers, input_dim = [], 2 * self.d_sub
        for i, hidden_dim in enumerate(self.gate_mlp_dims):
            layers.append(nn.Linear(input_dim if i == 0 else self.gate_mlp_dims[i - 1], hidden_dim))
            if i < len(self.gate_mlp_dims) - 1:
                layers += [nn.ReLU(), nn.Dropout(0.1)]
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _init_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _cross_attention(query, key, value):
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        return torch.bmm(F.softmax(scores, dim=-1), value)

    def _update(self, task: str, own: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        attention = self._cross_attention(getattr(self, f'{task}_q_proj')(own),
                                          getattr(self, f'{task}_k_proj')(others),
                                          getattr(self, f'{task}_v_proj')(others))
        gate = getattr(self, f'{task}_gate_mlp')(torch.cat([own, attention], dim=-1))
        return getattr(self, f'norm_{task}')((1 - gate) * own + gate * attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) -> (B, N, d_model).

        输入输出形状均为 (B, N, d_model)。
        """
        xt = x.transpose(1, 2)
        f = {task: getattr(self, f'{task}_mapping')(xt).transpose(1, 2) for task in TASKS}
        updated = [self._update(task, f[task], torch.cat([f[o] for o in TASKS if o != task], dim=-1))
                   for task in TASKS]
        return self.output_conv(torch.cat(updated, dim=-1).transpose(1, 2)).transpose(1, 2)


class PIGATransformerDecoderLayer(nn.Module):
    """Post-norm Transformer decoder layer whose feed-forward input is first passed through :class:`PIGAModule`.

    后归一化 Transformer 解码层，其前馈输入先经过 :class:`PIGAModule`。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['decoder']['num_heads']
        self.d_ff = config['model']['decoder']['d_ff']
        self.dropout = config['model']['decoder']['dropout']
        self.causal_attention = config['model']['decoder'].get('causal_attention', True)
        self.use_piga = config['model'].get('ablation', {}).get('use_piga', True)
        self.self_attn = nn.MultiheadAttention(self.d_model, self.num_heads, dropout=self.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(self.d_model, self.num_heads, dropout=self.dropout, batch_first=True)
        self.piga = PIGAModule(config) if self.use_piga else None
        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.d_ff), nn.GELU(), nn.Dropout(self.dropout),
                                 nn.Linear(self.d_ff, self.d_model), nn.Dropout(self.dropout))
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tgt_mask is None and self.causal_attention:
            T = tgt.shape[1]
            tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        tgt = self.norm1(tgt + self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)[0])
        h = self.norm2(tgt + self.cross_attn(tgt, memory, memory, need_weights=False)[0])
        return self.norm3(h + self.ffn(self.piga(h) if self.use_piga else h))


class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion-timestep embedding followed by an MLP.

    正弦扩散时间步嵌入，后接 MLP。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.half_dim = d_model // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(torch.arange(self.half_dim, device=t.device) * -self.emb)
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
        return self.mlp(emb)
