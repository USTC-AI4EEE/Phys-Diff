import json
from pathlib import Path
from typing import Tuple, Union

import torch

from .models import PhysDiff

__all__ = ['ROOT', 'VARIANTS', 'load_config', 'load_checkpoint', 'load_model', 'save_checkpoint']

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    'main': dict(config=ROOT / 'configs' / 'main.json',
                 checkpoint=ROOT / 'checkpoints' / 'main.pth',
                 members=ROOT / 'members' / 'main.npz',
                 description='Phys-Diff (ERA5 history + FengWu forecast fields)'),
    'wo_fengwu': dict(config=ROOT / 'configs' / 'wo_fengwu.json',
                      checkpoint=ROOT / 'checkpoints' / 'wo_fengwu.pth',
                      members=ROOT / 'members' / 'wo_fengwu.npz',
                      description='Phys-Diff w/o FengWu (ERA5 history only)'),
}

PathLike = Union[str, Path]


def load_config(path_or_variant: PathLike) -> dict:
    """Load a JSON config by path or by variant name (``'main'`` / ``'wo_fengwu'``).

    按路径或变体名（``'main'`` / ``'wo_fengwu'``）加载 JSON 配置。
    """
    path = VARIANTS[path_or_variant]['config'] if str(path_or_variant) in VARIANTS else Path(path_or_variant)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_checkpoint(path_or_variant: PathLike) -> dict:
    path = VARIANTS[path_or_variant]['checkpoint'] if str(path_or_variant) in VARIANTS else Path(path_or_variant)
    return torch.load(path, map_location='cpu', weights_only=True)


def load_model(path_or_variant: PathLike, device: Union[str, torch.device] = 'cpu') -> Tuple[PhysDiff, dict]:
    """Build :class:`PhysDiff` from the config stored in the checkpoint and load its (EMA) weights.

    >>> model, config = load_model('main', device='cuda')

    用 checkpoint 中保存的配置构建 :class:`PhysDiff`，并加载其（EMA）权重。
    """
    saved = load_checkpoint(path_or_variant)
    config = saved['config']
    model = PhysDiff(config)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.to(device).eval(), config


def save_checkpoint(path: PathLike, model: PhysDiff, criterion: torch.nn.Module, history: list, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(), criterion_state_dict=criterion.state_dict(),
                    config=model.config, weights='EMA', history=history, **extra), path)
