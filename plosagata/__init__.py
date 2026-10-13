"""Plosagata — model AI murni buatan sendiri (bukan open-source ambilan)."""
from .config import CONFIGS, GPTConfig
from .guard import WAKE_WORD, SAGI_HELLO, is_news_topic, redirect_reply, strip_wake
from .model import PureGPT, count_params

__all__ = ["CONFIGS", "GPTConfig", "PureGPT", "count_params",
           "WAKE_WORD", "SAGI_HELLO", "is_news_topic", "redirect_reply",
           "strip_wake"]
