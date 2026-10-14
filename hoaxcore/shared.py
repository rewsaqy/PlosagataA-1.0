"""Satu instance klasifier 12M untuk seluruh proses.

Dulu app.py dan SagiEngine masing-masing memuat copy sendiri (RAM 2x).
Semua pemanggil WAJIB lewat get_classifier().
"""
import os
from pathlib import Path

_CACHE = {}


def get_classifier(ckpt_dir="checkpoints", device="cpu"):
    key = (str(ckpt_dir), str(device))
    if key not in _CACHE:
        import torch
        from .model import TinyHoaxTransformer
        from .tokenizer import WordTokenizer
        ckpt = Path(os.getenv("CKPT_DIR", ckpt_dir))
        tok = WordTokenizer.load(ckpt / "vocab.json")
        model = TinyHoaxTransformer(len(tok.vocab), max_len=tok.max_len)
        for n in ["best.pt", "last.pt"]:
            p = ckpt / n
            if p.exists():
                model.load_state_dict(torch.load(p, map_location=device,
                                                 weights_only=True))
                break
        model.to(device).eval()
        _CACHE[key] = (model, tok)
    return _CACHE[key]


def predict_proba(text, ckpt_dir="checkpoints", device="cpu"):
    """Helper satu panggilan: teks -> (label, probs). Dengan slang-normalize."""
    import torch
    import torch.nn.functional as F
    from .dataset import ID2LABEL
    try:
        from plosagata.slang import normalize_slang
        text = normalize_slang(text)
    except Exception:
        pass
    model, tok = get_classifier(ckpt_dir, device)
    ids, mask = tok.encode(text)
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([ids]).to(device),
                                torch.tensor([mask]).to(device)), dim=-1)[0].tolist()
    return ID2LABEL[max(range(len(probs)), key=lambda i: probs[i])], probs
