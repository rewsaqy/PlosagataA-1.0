"""Konfigurasi arsitektur Plosagata — murni buatan sendiri."""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab: int = 5000
    dim: int = 256
    layers: int = 6
    heads: int = 8
    ff: int = 1024
    max_len: int = 256
    dropout: float = 0.0
    tie_weights: bool = True  # head pinjam bobot embedding (hemat ~32M di More)


# Model resmi Plosagata (dengan varian 12M andalan untuk CPU / latihan mandiri):
CONFIGS: dict[str, GPTConfig] = {
    # Plosagata_Browser — arsitektur 25M Core andalan (Hybrid RAG & Internet Live Search)
    "Plosagata_Browser": GPTConfig(vocab=8000, dim=512, layers=8, heads=8, ff=1536, max_len=512, dropout=0.05, tie_weights=True),
    # Plosata_Rowser — alias kompatibilitas 25M
    "Plosata_Rowser": GPTConfig(vocab=8000, dim=512, layers=8, heads=8, ff=1536, max_len=512, dropout=0.05, tie_weights=True),
    # Plosagata_12M — arsitektur 12M hasil rework (latihan mandiri di CPU/GPU, hemat RAM)
    "Plosagata_12M": GPTConfig(vocab=5000, dim=384, layers=6, heads=6, ff=1536, max_len=256, dropout=0.05, tie_weights=True),
    # Plosagata_Flash — alias kompatibilitas 12M
    "Plosagata_Flash": GPTConfig(vocab=5000, dim=384, layers=6, heads=6, ff=1536, max_len=256),
    # Plosagata_Go — respon cepat (CPU/laptop, demo kilat)
    "Plosagata_Go": GPTConfig(vocab=8000, dim=768, layers=12, heads=12, ff=3072, max_len=512),
    # Plosagata_More — medium, target training utama (~912M, GPU 6GB+)
    "Plosagata_More": GPTConfig(vocab=16000, dim=2048, layers=22, heads=16, ff=5632, max_len=1024),
    # Plosagata_Next — besar (~1.55B, butuh GPU 12GB+ / cluster)
    "Plosagata_Next": GPTConfig(vocab=16000, dim=2048, layers=38, heads=16, ff=5632, max_len=1024),
}

# Jumlah parameter terverifikasi (scripts/gpt_pure.py) — tanpa bangun model.
PARAMS = {
    "Plosagata_Browser": 25379840,
    "Plosata_Rowser": 25379840,
    "Plosagata_12M": 12665856,
    "Plosagata_Flash": 12665856,
    "Plosagata_Go": 91593216,
    "Plosagata_More": 912008192,
    "Plosagata_Next": 1549927424,
}

# Nama tampilan untuk manusia
DISPLAY = {
    "Plosagata_Browser": "Plosagata_Browser (25M Core) — Hybrid RAG & Real-Time Search",
    "Plosata_Rowser": "Plosata_Rowser (25M) — Hybrid RAG & Real-Time Search",
    "Plosagata_12M": "Plosagata_12M (12M Core) — Reworked Neural Chat",
    "Plosagata_Flash": "Plosagata_Flash (12M) — super ringan",
    "Plosagata_Go": "Plosagata_Go (small) — respon cepat",
    "Plosagata_More": "Plosagata_More (B1) — medium",
    "Plosagata_Next": "Plosagata_Next (b1x38) — besar",
}
