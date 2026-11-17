"""Phys-Diff: A Physics-Inspired Latent Diffusion Model for Tropical Cyclone Forecasting (ICASSP 2026).

Phys-Diff：一种物理启发的热带气旋预报潜空间扩散模型（ICASSP 2026）。
"""
from .checkpoint import VARIANTS, load_config, load_model
from .losses import UncertaintyLoss
from .metrics import metric_rows, physical_values
from .models import PhysDiff

__version__ = '1.0.0'
__all__ = ['PhysDiff', 'UncertaintyLoss', 'load_model', 'load_config', 'VARIANTS', 'metric_rows', 'physical_values']
