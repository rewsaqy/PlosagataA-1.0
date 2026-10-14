"""Plosagata — model AI murni buatan sendiri (bukan open-source ambilan)."""
from .config import CONFIGS, GPTConfig
from .guard import (WAKE_WORD, SAGI_HELLO, ada_kata_berita, ada_kata_offtopic,
                     is_follow_up, is_follow_up_safe, is_news_topic,
                     redirect_reply, sagi_hello, sagi_redirect, stem_id,
                     strip_wake)
from .model import PureGPT, count_params, detect_size
from .slang import SLANG, normalize_slang

__all__ = ["CONFIGS", "GPTConfig", "PureGPT", "count_params", "detect_size",
           "WAKE_WORD", "SAGI_HELLO", "ada_kata_berita", "ada_kata_offtopic",
           "is_follow_up", "is_follow_up_safe", "is_news_topic", "redirect_reply",
           "sagi_hello", "sagi_redirect", "stem_id", "strip_wake",
           "SLANG", "normalize_slang"]
