"""Build index RAG offline: korpus.txt -> data/rag_index/."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from hoaxcore.rag import build_bm25, build_tfidf, chunk_teks, save_index, tokenize_id


def iter_articles(path, limit=0):
    n = 0
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(line)
            elif buf:
                art = "".join(buf).strip()
                buf = []
                if len(art) < 500:
                    continue
                yield art
                n += 1
                if limit and n >= limit:
                    return
        if buf:
            art = "".join(buf).strip()
            if len(art) >= 500:
                yield art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/korpus.txt")
    ap.add_argument("--out", dest="out", default="data/rag_index")
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not Path(a.inp).exists():
        raise SystemExit(f"korpus tidak ada: {a.inp} (jalankan scripts/build_corpus.py dulu)")
    chunks = []
    for art in iter_articles(a.inp, a.limit):
        chunks.extend(chunk_teks(art, a.chunk, a.overlap))
    if not chunks:
        raise SystemExit("korpus kosong / semua artikel <500 char")
    print(f"chunk: {len(chunks)} dari {a.inp}")
    toked = [tokenize_id(c) for c in chunks]
    bm25 = build_bm25(toked)
    vec, mat = build_tfidf(chunks)
    save_index(a.out, bm25, vec, mat, chunks)
    with open(Path(a.out) / "chunks.jsonl", "w", encoding="utf-8") as o:
        for i, c in enumerate(chunks):
            o.write(json.dumps({"id": i, "teks": c[:2000]}, ensure_ascii=False) + "\n")
    print(f"selesai -> {a.out} ({len(chunks)} chunk)")


if __name__ == "__main__":
    main()
