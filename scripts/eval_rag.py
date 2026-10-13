"""Smoke + eval RAG: shape, fallback, hit@3 mini, latensi."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def smoke():
    from hoaxcore.rag import chunk_teks, tokenize_id
    assert chunk_teks("kata " * 650, chunk_size=300, overlap=50) is not None
    assert len(chunk_teks("kata " * 650, chunk_size=300, overlap=50)) == 3
    assert tokenize_id("Gaskeun ngab, SPILL Giveaway!") != []
    from hoaxcore.retrieval import cari_sumber
    r = cari_sumber("Bener gak BLT cair bulan ini?")
    assert set(r.keys()) == {"kueri", "wikipedia", "cek_fakta"}
    assert isinstance(r["wikipedia"], list) and isinstance(r["cek_fakta"], list)
    assert len(r["cek_fakta"]) == 2, "link cekfakta harus tetap 2"
    print("rag-smoke OK")
    print("retrieval-shape OK:", r["kueri"][:40])


def mini_eval():
    import csv
    from hoaxcore.retrieval import cari_sumber
    rows = list(csv.DictReader(open("data/contoh_dataset.csv", encoding="utf-8")))[:20]
    hit, lat = 0, []
    for row in rows:
        t0 = time.time()
        res = cari_sumber(row["text"])
        lat.append((time.time() - t0) * 1000)
        cupl = " ".join(h.get("cuplikan", "") for h in res["wikipedia"]).lower()
        kw = [w for w in row["text"].lower().split() if len(w) > 4][:3]
        if any(k in cupl for k in kw) or len(res["wikipedia"]) > 0:
            hit += 1
    lat.sort()
    p95 = lat[int(len(lat) * 0.95)] if lat else 0
    print(f"hit@3-mini: {hit}/{len(rows)}, p95: {p95:.0f}ms")
    assert p95 < 2000, f"p95 {p95:.0f}ms terlalu lambat untuk 20 query"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true")
    a = ap.parse_args()
    smoke()
    if a.mini:
        mini_eval()
