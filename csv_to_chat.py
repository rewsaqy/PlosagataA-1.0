"""Ubah dataset hoaks CSV (text,label) jadi data chat instruction (jsonl).

  python3 csv_to_chat.py --in data/contoh_dataset.csv --out data/chat_hoaks.jsonl
  cat data/chat_hoaks.jsonl data/contoh_chat.jsonl > data/train_chat.jsonl

Lalu latih di GPU (lihat train_chat_lora.py). Makin banyak contoh
( target 200-500 ), makin pintar dia bedakan hoaks.
"""
import argparse
import csv
import json

TEMPLATE = {
    "valid": "Klaim itu tampak valid. {extra} Pastikan sumbernya media resmi dan perhatikan tanggal publikasinya.",
    "hoaks": "Itu berpotensi HOAKS. {extra} Jangan disebarkan. Cek ke turnbackhoax.id atau cekfakta.com dulu.",
    "meragukan": "Itu meragukan. {extra} Jangan sebarkan sebelum terverifikasi dari sumber resmi.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/contoh_dataset.csv")
    ap.add_argument("--out", default="data/chat_hoaks.jsonl")
    a = ap.parse_args()

    n = 0
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as o:
        for row in csv.DictReader(f):
            text, label = row["text"].strip(), row["label"].strip().lower()
            if label not in TEMPLATE:
                continue
            extra = "Ciri khas hoaks terdeteksi." if label == "hoaks" else "Tidak ada ciri hoaks kuat."
            o.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "Kamu asisten validator hoaks Indonesia yang ramah."},
                    {"role": "user", "content": f"Bener gak sih berita ini: {text}"},
                    {"role": "assistant", "content": TEMPLATE[label].format(extra=extra)},
                ]
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"Ditulis {n} contoh -> {a.out}")


if __name__ == "__main__":
    main()
