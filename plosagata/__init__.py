"""Plosagata — model AI murni buatan sendiri (bukan open-source ambilan)."""
from .config import CONFIGS, GPTConfig
from .model import PureGPT, count_params

__all__ = ["CONFIGS", "GPTConfig", "PureGPT", "count_params"]
