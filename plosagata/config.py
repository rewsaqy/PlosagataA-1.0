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


# Tepat 3 model (nama resmi):
CONFIGS: dict[str, GPTConfig] = {
    # Plosagata_Go — respon cepat (CPU/laptop, demo kilat)
    "Plosagata_Go": GPTConfig(vocab=8000, dim=768, layers=12, heads=12, ff=3072, max_len=512),
    # Plosagata_More — medium, target training utama (~912M, GPU 6GB+)
    "Plosagata_More": GPTConfig(vocab=16000, dim=2048, layers=22, heads=16, ff=5632, max_len=1024),
    # Plosagata_Next — besar (~1.55B, butuh GPU 12GB+ / cluster)
    "Plosagata_Next": GPTConfig(vocab=16000, dim=2048, layers=38, heads=16, ff=5632, max_len=1024),
}

# Nama tampilan untuk manusia
DISPLAY = {
    "Plosagata_Go": "Plosagata_Go (small) — respon cepat",
    "Plosagata_More": "Plosagata_More (B1) — medium",
    "Plosagata_Next": "Plosagata_Next (b1x38) — besar",
}
