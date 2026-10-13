"""Ambil daftar kata HF hndrbrm/indonesia_vocabulary -> gabung ke vocab_seed.txt.

Sumber: https://huggingface.co/datasets/hndrbrm/indonesia_vocabulary
File mentah: vocabulary.json (1.93 MB, 116rb kata tunggal, kolom "text").
Tanpa dependensi baru (urllib stdlib). Satu baris seed = satu kata/frasa.

Contoh:
  python scripts/fetch_vocab.py --limit 10000   (subset uji, sebar merata)
  python scripts/fetch_vocab.py --limit 0       (full 116rb)
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

URL = ("https://huggingface.co/datasets/hndrbrm/indonesia_vocabulary"
       "/resolve/main/vocabulary.json")
CACHE = Path("data/vocab_hf.json")


def download():
    if CACHE.exists() and CACHE.stat().st_size > 1_000_000:
        print(f"pakai cache {CACHE} ({CACHE.stat().st_size / 1e6:.2f} MB)")
        return json.loads(CACHE.read_text(encoding="utf-8"))
    print(f"download {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "PlosagataA/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8")
    CACHE.write_text(raw, encoding="utf-8")
    print(f"tersimpan {CACHE} ({len(raw) / 1e6:.2f} MB)")
    return json.loads(raw)


def word_list(data):
    if isinstance(data, dict):
        for key in ("text", "words", "vocabulary", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = [row.get("text", "") for row in data]
    return [str(w).strip() for w in data if str(w).strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10000,
                    help="0 = full 116rb; N = subset N kata sebar merata")
    ap.add_argument("--seed", default="data/vocab_seed.txt")
    a = ap.parse_args()

    words = word_list(download())
    print(f"total kata HF: {len(words)}")
    if a.limit and a.limit < len(words):
        step = len(words) / a.limit
        words = [words[int(i * step)] for i in range(a.limit)]
        print(f"subset merata: {len(words)} kata")

    seed_p = Path(a.seed)
    lama = [l for l in seed_p.read_text(encoding="utf-8").splitlines()]
    ada = {l.strip().lower() for l in lama if l.strip() and not l.startswith("#")}
    baru = [w for w in words if w.lower() not in ada]
    print(f"baris seed lama: {len(lama)} | kata baru: {len(baru)}")

    with open(seed_p, "a", encoding="utf-8") as f:
        f.write("\n# --- Tambahan HF hndrbrm/indonesia_vocabulary ---\n")
        for w in baru:
            f.write(w + "\n")
    print(f"selesai -> {seed_p} (+{len(baru)} baris)")


if __name__ == "__main__":
    main()
