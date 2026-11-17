"""Loader for the prepared IBTrACS + ERA5 + FengWu forecast windows (layout described in ``docs/DATA.md``).

预处理后的 IBTrACS + ERA5 + FengWu 预报窗口加载器（数据格式见 ``docs/DATA.md``）
。
"""
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .metrics import clean_basin

__all__ = ['SPLITS', 'SPLIT_SIZES', 'SPLIT_OFFSETS', 'PreparedData']

#: Chronological split: train 1980-2017, validation 2018, test 2019-2022.
#: 按时间划分：训练 1980-2017，验证 2018，测试 2019-2022。
SPLITS = ('train', 'val', 'test')
SPLIT_SIZES = dict(train=32721, val=2307, test=6699)
#: Global window index offset of each split; used to derive the per-window sampling seed.
#: 各划分的全局窗口索引偏移，用于推导每个窗口的采样随机种子。
SPLIT_OFFSETS = dict(train=0, val=32721, test=32721 + 2307)

META_KEYS = ('states', 'references', 'basin', 'sid', 'truth_coords', 'truth_winds', 'truth_pres')


class PreparedData:
    """Memory-mapped forecast windows from one or more splits, concatenated in the given order.

    Args:
        data: directory with ``<split>/states.npz``, ``<split>/weather.npy``, ``<split>/extrema.npy``.
        geometry: directory with ``<split>.npz`` holding FengWu vortex-tracker centres (required if ``use_fengwu``).
        fine: directory with ``<split>/fine_core.npy`` and ``<split>/packed_index.npy``.
        config: model config (``story.use_fengwu`` and ``normalization.coord_std`` are used).
        parts: splits to load.

    以内存映射方式读取一个或多个划分的预报窗口，按给定顺序拼接。

    参数：data 为含 states.npz / weather.npy / extrema.npy 的目录；geometry 为 FengWu 涡旋追踪中心目录（use_fengwu 时必需）
    ；fine 为含 fine_core.npy 与 packed_index.npy 的目录；config 为模型配置；parts 为要加载的划分。
    """

    def __init__(self, data: Path, geometry: Optional[Path], fine: Path, config: dict,
                 parts: Sequence[str] = ('train', 'val')):
        data, fine = Path(data), Path(fine)
        geometry = Path(geometry) if geometry is not None else None
        self.parts = []
        self.names = tuple(parts)
        self.use_fengwu = config['story']['use_fengwu']
        if self.use_fengwu and geometry is None:
            raise ValueError('--geometry is required for the main (FengWu) model')
        coord_std = np.array(config['normalization']['coord_std'], np.float32)
        for name in self.names:
            if name not in SPLITS:
                raise ValueError(f'Unknown split {name!r}')
            folder = data / name
            with np.load(folder / 'states.npz') as z:
                part = {key: z[key] for key in z.files}
            part['weather'] = np.load(folder / 'weather.npy', mmap_mode='r')
            part['extrema'] = np.load(folder / 'extrema.npy', mmap_mode='r')
            if self.use_fengwu:
                with np.load(geometry / (name + '.npz')) as z:
                    np.testing.assert_array_equal(z['packed_index'], part['packed_index'])
                    points = z['forecast_centers'].astype(np.float32) - part['references'][:, None]
                    points[..., 1] = (points[..., 1] + 180) % 360 - 180
                    part['future_centers'] = points / coord_std
                    part['center_quality'] = z['vortex_detected'].astype(np.float32)
            else:
                part['future_centers'] = np.repeat(part['states'][:, 3:4, :2], 4, axis=1)
                part['center_quality'] = np.zeros((len(part['packed_index']), 4), np.float32)
            np.testing.assert_array_equal(np.load(fine / name / 'packed_index.npy'), part['packed_index'])
            part['fine_weather'] = np.load(fine / name / 'fine_core.npy', mmap_mode='r')
            self.parts.append(part)
        self.sizes = np.array([len(p['packed_index']) for p in self.parts])
        self.offsets = np.r_[0, self.sizes.cumsum()]
        self.n = int(self.offsets[-1])
        self.meta = {key: np.concatenate([p[key] for p in self.parts]) for key in META_KEYS}
        self.meta['basin'] = clean_basin(self.meta['basin'])
        self.meta['original_packed_index'] = np.concatenate([p['packed_index'] for p in self.parts])
        self.meta['source_split'] = np.concatenate([np.repeat(n, s) for n, s in zip(self.names, self.sizes)])
        self.meta['noise_index'] = np.concatenate(
            [np.arange(SPLIT_OFFSETS[n], SPLIT_OFFSETS[n] + s) for n, s in zip(self.names, self.sizes)])

    def __len__(self) -> int:
        return self.n

    def targets(self, ids) -> np.ndarray:
        """Normalised future states (len(ids), 4, 4).

        归一化的未来状态 (len(ids), 4, 4)。
        """
        return self.meta['states'][ids, 4:]

    def inputs(self, ids, device='cuda') -> Dict[str, torch.Tensor]:
        """Model inputs for the given global indices, as keyword arguments of :meth:`PhysDiff.anchor`.

        给定全局索引的模型输入，作为 :meth:`PhysDiff.anchor` 的关键字参数。
        """
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= self.n:
            raise ValueError('Invalid dataset indices')
        result = {}
        for i, part in enumerate(self.parts):
            positions = np.flatnonzero((ids >= self.offsets[i]) & (ids < self.offsets[i + 1]))
            if not len(positions):
                continue
            rows = ids[positions] - self.offsets[i]

            def fields(array):
                if self.use_fengwu:
                    return np.array(array[rows])
                past = np.array(array[rows, :4])   # w/o FengWu: never read future fields / 无 FengWu：从不读取未来场
                return np.concatenate([past, np.repeat(past[:, 3:4], 4, axis=1)], axis=1)

            values = dict(history=part['states'][rows, :4], references=part['references'][rows],
                          weather=fields(part['weather']), extrema=fields(part['extrema']),
                          future_centers=part['future_centers'][rows], center_quality=part['center_quality'][rows],
                          fine_weather=fields(part['fine_weather']))
            positions = torch.as_tensor(positions, device=device)
            for key, value in values.items():
                value = torch.as_tensor(value, device=device)
                if key not in result:
                    result[key] = torch.empty((len(ids),) + value.shape[1:], dtype=value.dtype, device=device)
                result[key].index_copy_(0, positions, value)
        return result
