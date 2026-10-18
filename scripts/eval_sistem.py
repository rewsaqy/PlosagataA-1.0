"""Eval END-TO-END sistem verdict: model 12M + heuristik -> label_final.

Mengukur akurasi pipa keputusan yang BENAR-BENAR dipakai /api/check:
predict_proba() -> local_verdict() -> final_label(), pada split val yang
SAMA seperti train.py & eval_verdict.py (test_size=0.15, seed 42, stratify).

  python scripts/eval_sistem.py                     (checkpoints/best.pt)
  python scripts/eval_sistem.py --ckpt-dir /tmp/opencode/ckpt_ls
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sklearn.model_selection import train_test_split

from hoaxcore import ID2LABEL, load_csv, local_verdict
from hoaxcore.shared import predict_proba

LABELS = ["valid", "hoaks", "meragukan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contoh_dataset.csv")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-show", type=int, default=15)
    a = ap.parse_args()

    texts, labels = load_csv(a.data)
    y = list(labels)
    _, va_t, _, va_l = train_test_split(texts, y, test_size=0.15,
                                        random_state=42, stratify=y)

    n = len(va_t)
    m_acc = s_acc = 0
    n_fiksi = 0
    cm_sys = {l: {k: 0 for k in LABELS + ["fiksi"]} for l in LABELS}
    geser = {}
    salah = []
    for t, gold in zip(va_t, va_l):
        label, probs = predict_proba(t, ckpt_dir=a.ckpt_dir)
        lv = local_verdict(t, {"label": label, "probs": probs})
        lab_final = lv["label_final"]
        ml = ID2LABEL[gold]
        if label == ml:
            m_acc += 1
        if lab_final == ml:
            s_acc += 1
        cm_sys[ml][lab_final] += 1
        if lab_final == "fiksi":
            n_fiksi += 1
        if label != lab_final:
            geser[(label, lab_final)] = geser.get((label, lab_final), 0) + 1
        if lab_final != ml:
            salah.append((ml, label, lab_final, lv["skor_hoaks"], t[:70]))

    print(f"val: {n} sampel | ckpt: {a.ckpt_dir}")
    print(f"akurasi model-only : {m_acc / n:.4f}")
    print(f"akurasi SISTEM     : {s_acc / n:.4f}  (label_final yg dipakai app)")
    if n_fiksi:
        print(f"catatan: {n_fiksi} sampel val terdeteksi 'fiksi'")

    print("\n--- confusion SISTEM (baris aktual, kolom label_final) ---")
    print("          pred->  " + "  ".join(f"{l:>9s}" for l in LABELS + ["fiksi"]))
    for i, l in enumerate(LABELS):
        print(f"aktual {l:9s}  " + "  ".join(f"{cm_sys[l][k]:9d}" for k in LABELS + ["fiksi"]))
    for k in LABELS:
        tp = cm_sys[k][k]
        prec = tp / max(sum(cm_sys[l][k] for l in LABELS), 1)
        rec = tp / max(sum(cm_sys[k].values()), 1)
        print(f"{k:9s} precision={prec:.3f} recall={rec:.3f}")

    if geser:
        print("\n--- pergeseran label model -> sistem ---")
        for (dari, ke), c in sorted(geser.items(), key=lambda x: -x[1]):
            print(f"  {dari:9s} -> {ke:9s} : {c}")

    print(f"\n--- {len(salah)} mismatch sistem (act | model | final | skor | teks) ---")
    for ml, label, lab_final, skor, t in salah[: a.max_show]:
        print(f"[{ml:9s}] model={label:9s} final={lab_final:9s} skor={skor:3d} :: {t}")
    if len(salah) > a.max_show:
        print(f"... dan {len(salah) - a.max_show} lainnya")


if __name__ == "__main__":
    main()
