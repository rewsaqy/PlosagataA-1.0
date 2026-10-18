"""Kompatibilitas lama — implementasi pindah ke paket plosagata/.

Pakai yang baru:
  from plosagata import PureGPT, CONFIGS, count_params
"""
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from plosagata import CONFIGS, PureGPT, count_params

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Plosagata_Go"
    cfg = CONFIGS[name]
    m = PureGPT(cfg)
    total = count_params(m)
    print(f"PureGPT-{name}: {total:,} ({total/1e9:.3f}B / {total/1e6:.1f}M)")
    print(asdict(cfg))
    if name in ("Plosagata_Browser", "Plosagata_12M", "Plosagata_Flash", "Plosagata_Go"):
        x = torch.randint(0, cfg.vocab, (1, 17))
        logits, _ = m(x[:, :-1], x[:, 1:])  # geser: posisi i menebak token i+1
        import torch.nn.functional as F
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), x[:, 1:].reshape(-1))
        print("forward OK:", tuple(logits.shape), "loss:", loss.item())
        print("generate OK:", tuple(m.generate(x[:, :-1], max_new=10).shape))
    else:
        print("Mode besar: konstruksi di GPU saja (CPU terlalu berat untuk forward).")
