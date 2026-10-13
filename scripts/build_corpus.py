"""Bangun korpus teks Indonesia untuk pretrain Plosagata b1.

Sumber utama: Wikipedia Bahasa Indonesia (gratis, legal, ~ratusan juta token).
Jalankan di komputer GPU SEBELUM training (butuh internet):

  pip install datasets
  python build_corpus.py --out data/korpus.txt --limit 0

--limit N: batasi N artikel untuk tes cepat. Default 0 = semua.
Output: data/korpus.txt (artikel dipisah baris kosong).
Tambahkan juga artikel turnbackhoax.id bagian FAKTA/VALID agar model
mengenal gaya bahasa berita benar (copy manual ke file yang sama).
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/korpus.txt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets  (di komputer GPU)")

    print("Download Wikipedia Indonesia ... (sekali saja, ~1GB)")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
    except Exception:
        ds = load_dataset("wikipedia", "20231101.id", split="train", streaming=True)

    n, chars = 0, 0
    with open(a.out, "w", encoding="utf-8") as o:
        for row in ds:
            text = row["text"].strip()
            if len(text) < 500:  # buang artikel rintisan
                continue
            o.write(text[:8000] + "\n\n")
            n += 1
            chars += len(text)
            if n % 20000 == 0:
                print(f"  {n} artikel ({chars/1e6:.0f} MB) ...")
            if a.limit and n >= a.limit:
                break
    print(f"Selesai: {n} artikel, {chars/1e6:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()
