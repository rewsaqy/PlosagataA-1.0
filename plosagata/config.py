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
    tie_weights: bool = True  # head pinjam bobot embedding (hemat ~32M di b1)


CONFIGS: dict[str, GPTConfig] = {
    # latihan di CPU sini
    "tiny": GPTConfig(),
    # latihan GPU menengah (~65M dengan tie, hemat dari 98M)
    "small": GPTConfig(vocab=8000, dim=768, layers=12, heads=12, ff=3072, max_len=512),
    # target ~900M — training di device GPU lain
    "b1": GPTConfig(vocab=16000, dim=2048, layers=22, heads=16, ff=5632, max_len=1024),
    # varian dalam (~1.15B): hanya jika GPU 10GB+ (RTX 3080) + data besar
    "b1deep": GPTConfig(vocab=16000, dim=2048, layers=28, heads=16, ff=5632, max_len=1024),
    # EKSPERIMEN cluster (BUKAN untuk RTX 2060/3080):
    # 38 layer ~1.55B butuh ~10GB VRAM mode hemat; 48 layer ~1.95B butuh ~12GB+.
    "b1x38": GPTConfig(vocab=16000, dim=2048, layers=38, heads=16, ff=5632, max_len=1024),
    "b1x48": GPTConfig(vocab=16000, dim=2048, layers=48, heads=16, ff=5632, max_len=1024),
}
