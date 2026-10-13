"""Inferensi cepat: teks -> label + probabilitas."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import ID2LABEL
from model import TinyHoaxTransformer
from tokenizer import WordTokenizer


def load_model(ckpt_dir, device="cpu"):
    ckpt_dir = Path(ckpt_dir)
    tok = WordTokenizer.load(ckpt_dir / "vocab.json")
    model = TinyHoaxTransformer(
        vocab_size=len(tok.vocab), max_len=tok.max_len, num_classes=3
    )
    for name in ["best.pt", "last.pt"]:
        p = ckpt_dir / name
        if p.exists():
            model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            break
    model.to(device).eval()
    return model, tok


def predict(text, model, tok, device="cpu"):
    ids, mask = tok.encode(text)
    ids_t = torch.tensor([ids], dtype=torch.long).to(device)
    mask_t = torch.tensor([mask], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids_t, mask_t)
        probs = F.softmax(logits, dim=-1)[0].tolist()
    pred = max(range(len(probs)), key=lambda i: probs[i])
    return {"label": ID2LABEL[pred], "probs": probs, "logits": logits[0].tolist()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints")
    p.add_argument("--text", default="Pemerintah resmi membagikan bantuan langsung tunai, cek info di situs resmi.")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(a.ckpt, device)
    print(predict(a.text, model, tok, device))
