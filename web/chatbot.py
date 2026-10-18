"""Kompatibilitas lama — mesin Sagi kini tinggal di paket `sagi`.

Modul ini dulu berisi seluruh SagiEngine; sekarang hanya re-export supaya
import lama tetap jalan:
    from web.chatbot import SagiEngine   (dari web/app.py)
    from chatbot import SagiEngine       (dari scripts/eval_*.py)
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sagi import SagiEngine  # noqa: E402
from sagi.prompt import bangun_prompt, bersihkan_output  # noqa: E402,F401
from sagi.session import MAX_SESI, SESI_UMUR  # noqa: E402,F401

__all__ = ["SagiEngine", "bangun_prompt", "bersihkan_output", "MAX_SESI", "SESI_UMUR"]
