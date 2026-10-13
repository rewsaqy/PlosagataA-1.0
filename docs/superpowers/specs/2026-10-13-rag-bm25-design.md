# Desain RAG Standar Lokal — BM25 + TF-IDF Offline (Opsi B/A)

Tanggal: 2026-10-13
Status: Disetujui user (chat) — siap ke implementation plan
Scope: Bounded-architectural kecil — ganti `kata_kunci()` keyword dengan retrieval semantik lokal, tanpa model unduhan.

## 1. Intent (disepakati)

- Tujuan: `cari_sumber()` saat ini hanya `kata_kunci()` regex + Wikipedia live search. Tidak relevan untuk parafrase/slang, gagal saat offline/demo lomba.
- Constraints user:
  1. Murni lokal saja — tanpa `sentence-transformers`/OpenAI embedding, jaga semangat lomba "tanpa model unduhan".
  2. Korpus = Wikipedia offline (`data/korpus.txt` via `scripts/build_corpus.py`, ratusan MB).
  3. Boleh tambah 1-2 lib ringan, tetap CPU-only (mis. `rank-bm25`), `sklearn`+`numpy` sudah ada.
- Success: relevansi top-3 lebih baik dari baseline pada 20 klaim uji, latency <500ms di CPU laptop, fallback tidak pernah crash.

## 2. Arsitektur

```
data/korpus.txt (offline, build_corpus.py)
  │ scripts/build_index.py (sekali, di GPU/PC build)
  ▼
data/rag_index/{chunks.jsonl, bm25.pkl, tfidf.pkl, meta.json}  (di-gitignore, dibangun lokal)
  ▲
hoaxcore/rag.py — rag_search(teks, k=3)
  │ 1. normalize_slang + basic_clean (reuse)
  │ 2. BM25Okapi top-20 (rank-bm25)
  │ 3. TF-IDF cosine rerank top-3 (sklearn, sudah ada)
  │ 4. return [{judul, url/tanda offline, cuplikan, skor}]
  ▼
hoaxcore/retrieval.py — cari_sumber() dimodif:
  coba rag_search() dulu; jika index hilang/corrupt → fallback ke cari_wikipedia() live + link_cek_fakta()
  ▼
web/chatbot.py SagiEngine.reply() — tanpa ubah signature, hanya sumber lebih relevan.
```

Alasan BM25+TF-IDF: terbaik untuk hoaks ID yang keyword-heavy (nama tokoh, "BLT", "giveaway", angka), tanpa training, CPU hemat, 1 dep baru. Dense encoder sendiri ditolak (kualitas awal rendah, butuh latih).

## 3. Komponen & file

Baru:
- `hoaxcore/rag.py` — `chunk_teks()`, `load_index()`, `rag_search()`, lazy-load singleton, tanpa torch.
- `scripts/build_index.py` — `--in data/korpus.txt --out data/rag_index --chunk 300 --overlap 50 --limit 0`, tulis `chunks.jsonl` + `bm25.pkl` + `tfidf.pkl` + `meta.json`, log jumlah chunk + waktu.
- `scripts/eval_rag.py` (atau `tests/test_rag.py` jika mau pytest) — 20 klaim uji, hitung hit@3 vs baseline `kata_kunci()`.

Ubah:
- `hoaxcore/retrieval.py` — `cari_sumber()` coba lokal dulu, fallback live. Jaga return shape `{'kueri','wikipedia','cek_fakta'}` agar `chat.html` tidak rusak.
- `hoaxcore/__init__.py` — export `rag_search`.
- `requirements.txt` — tambah `rank-bm25` (pin versi).
- `.gitignore` — tambah `data/rag_index/` + `data/korpus.txt` (besar).
- `docs/ARSITEKTUR.md` — tambah baris RAG.

Tidak disentuh: `plosagata/*`, `hoaxcore/model.py`, `web/app.py` routes, prompt/guard.

## 4. Data flow (detail)

Build (offline, sekali):
1. Baca `data/korpus.txt` pisah `\n\n` → buang <500 char (sama seperti build_corpus).
2. Chunk per artikel 300 kata, overlap 50 kata, simpan `{id, judul/artikel_idx, teks}`.
3. Tokenize ID: `basic_clean` + split (reuse, konsisten dengan query).
4. Fit `BM25Okapi(tokenized_chunks)` + `TfidfVectorizer()` → pickle.

Query (online, <500ms):
1. `clean = basic_clean(normalize_slang(teks))`.
2. BM25 scores atas semua chunk → top-20 idx.
3. TF-IDF cosine hanya pada top-20 → top-3.
4. Format ke `[{judul: f"Korpus#{id}", url: "", cuplikan: teks[:220], skor}]` + tetap sertakan `cek_fakta` links.
5. Jika `data/rag_index/` tidak ada → `cari_wikipedia(kueri)` seperti sekarang.

## 5. Error handling

- Index hilang: warning sekali `[rag] index tidak ada, fallback live`, tidak raise.
- Pickle corrupt/versi beda: try/except → fallback live + pesan `jalankan build_index.py`.
- Korpus kosong/limit 0: build gagal dengan pesan jelas, tidak tulis index setengah.
- Query kosong/pendek (<4 char): return `[]`, biarkan cek_fakta links yang bicara.
- Thread-safety Flask: load index sekali saat import, read-only sesudahnya.

## 6. Testing

- Unit: `chunk_teks` overlap benar, `rag_search("giveaway iPhone klik link")` kembalikan 3 hasil dengan skor turun.
- Relevansi: 20 klaim (10 hoaks nyata dari `contoh_dataset.csv` + 10 valid) — hit@3 manual, target ≥ baseline +30%.
- Latensi: `time rag_search` 20x di CPU, p95 <500ms untuk index ≤50k chunk.
- Integrasi: `SagiEngine.reply()` tetap return shape lama, `cari_sumber` tidak raise saat index dihapus (uji fallback).
- Repo tanpa pytest saat ini → pakai `scripts/eval_rag.py` + assert sederhana agar konsisten dengan pola repo (no test framework baru).

## 7. Non-goals (YAGNI)

- No vector DB server (FAISS/Qdrant), no embedding training, no GPU requirement.
- No streaming SSE, no multi-turn memory (itu opsi A terpisah).
- No auto-update index dari live Wikipedia (manual rebuild via script).
- No perubahan UI `chat.html` selain menampilkan `skor` jika ada.

## 8. Self-review

- [x] No TBD/TODO — semua path fallback konkret.
- [x] Konsisten: return shape `cari_sumber` dijaga, `chat.html` tidak rusak.
- [x] Scope single plan: hanya retrieval, tidak sentuh model chat/klasifier.
- [x] Ambiguitas dihilangkan: chunk 300/50, top-20→top-3, lib `rank-bm25` saja, index di-gitignore.

## 9. Persetujuan

Disetujui via chat 2026-10-13 ("Setuju, tulis spec", pendekatan A). Lanjut ke `writing-plans` untuk implementation plan. Tidak auto-commit sesuai aturan repo (commit hanya saat diminta eksplisit).
