"""Plosagata — model AI murni buatan sendiri (bukan open-source ambilan)."""
from .config import CONFIGS, GPTConfig
from .guard import is_news_topic, redirect_reply
from .model import PureGPT, count_params

__all__ = ["CONFIGS", "GPTConfig", "PureGPT", "count_params",
           "is_news_topic", "redirect_reply"]
