"""Paket Sagi — mesin chatbot Plosagata (dipakai web & CLI).

Modul:
  - engine  : SagiEngine (guard + topik + verdict + generasi streaming)
  - prompt  : pembangun prompt format training + pembersih output
  - session : memori sesi (topik + riwayat giliran)
  - sources : normalisasi sumber & blok sitasi jawaban
"""
from .engine import SagiEngine
from .prompt import bangun_prompt, bersihkan_output, rapikan_isi
from .session import MAX_SESI, MAKS_GILIRAN, SESI_UMUR, Sesi, SesiStore
from .sources import blok_sitasi, kumpulkan_sumber

__all__ = [
    "SagiEngine",
    "bangun_prompt", "bersihkan_output", "rapikan_isi",
    "Sesi", "SesiStore", "MAX_SESI", "SESI_UMUR", "MAKS_GILIRAN",
    "kumpulkan_sumber", "blok_sitasi",
]
