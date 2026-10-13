"""Eval baku klasifier 12M: confusion matrix + P/R per kelas + probe regresi.

Split SAMA seperti train.py (test_size=0.15, random_state=42, stratify)
agar angka sebanding antar run. Probe = kalimat anekdot yg pernah salah.

  python scripts/eval_verdict.py                 (pakai checkpoints/best.pt)
  python scripts/eval_verdict.py --ckpt checkpoints/last.pt
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hoaxcore import (ID2LABEL, HoaxDataset, TinyHoaxTransformer,
                      WordTokenizer, load_csv)

LABELS = ["valid", "hoaks", "meragukan"]

PROBE = [
    ("Giveaway iPhone gratis, klik link ini dan transfer dulu!", "hoaks"),
    ("Vaksin covid mengandung microchip, sebarkan sebelum dihapus!", "hoaks"),
    ("BMKG mencatat gempa magnitudo 4,8 di barat daya Sumur, tidak berpotensi tsunami.", "valid"),
    ("Jadwal posyandu RW 05 bulan ini Sabtu pekan kedua pukul 08.00.", "valid"),
    ("Katanya harga cabai naik lagi minggu ini, ada yang dengar juga?", "meragukan"),
    ("Benarkah BLT tambahan cair bulan depan?", "meragukan"),
]


def load(ckpt_dir, ckpt, device):
    ckpt_dir = Path(ckpt_dir)
    tok = WordTokenizer.load(ckpt_dir / "vocab.json")
    m = TinyHoaxTransformer(vocab_size=len(tok.vocab), max_len=tok.max_len)
    m.load_state_dict(torch.load(ckpt_dir / ckpt, map_location=device,
                                 weights_only=True))
    return m.to(device).eval(), tok


@torch.no_grad()
def predict(m, tok, texts, device, batch=32):
    ds = HoaxDataset(texts, [0] * len(texts), tok)
    ld = DataLoader(ds, batch_size=batch)
    out = []
    for ids, mask, _ in ld:
        out += m(ids.to(device), mask.to(device)).argmax(-1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contoh_dataset.csv")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts, labels = load_csv(a.data)
    y = list(labels)  # load_csv sudah mengembalikan id integer
    _, va_t, _, va_l = train_test_split(texts, y, test_size=0.15,
                                        random_state=42, stratify=y)
    m, tok = load(a.ckpt_dir, a.ckpt, device)
    pred = predict(m, tok, va_t, device)

    cm = confusion_matrix(va_l, pred, labels=[0, 1, 2])
    print(f"val: {len(va_t)} sampel | ckpt: {a.ckpt}")
    print("        pred-> valid hoaks ragu")
    for i, l in enumerate(LABELS):
        print(f"aktual {l:9s} {cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}")
    acc = sum(p == t for p, t in zip(pred, va_l)) / len(va_l)
    print(f"akurasi: {acc:.4f}")
    for i, l in enumerate(LABELS):
        tp = cm[i][i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i, :].sum(), 1)
        print(f"{l:9s} precision={prec:.3f} recall={rec:.3f}")

    print("--- probe regresi ---")
    pt = [t for t, _ in PROBE]
    pp = predict(m, tok, pt, device)
    skor = 0
    for (t, exp), p in zip(PROBE, pp):
        got = ID2LABEL[p]
        ok = "OK " if got == exp else "SALAH"
        if got == exp:
            skor += 1
        print(f"[{ok}] exp={exp:9s} got={got:9s} :: {t[:60]}")
    print(f"probe: {skor}/{len(PROBE)}")


if __name__ == "__main__":
    main()
