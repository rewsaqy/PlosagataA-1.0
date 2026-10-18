# scripts/ — Indeks (26 file, flat)

Semua script dijalankan **dari root repo** (bukan dari dalam `scripts/`):

```bash
.venv/bin/python scripts/NAMA.py --bantuan
```

Hasil latihan model selalu masuk **`checkpoints/`** (folder di-ignore git —
bobot `.pt` tidak pernah ikut ke GitHub). Salinan ke root repo tidak lagi
dibuat otomatis (dulu sumber duplikat ratusan MB; lihat `BUGS.md` B-24).

## 1. Latih model

| Script | Fungsi | Output |
|---|---|---|
| `train.py` | Latih **klasifier hoaks 12M** (`TinyHoaxTransformer`) dari CSV `text,label` | `checkpoints/vocab.json`, `best.pt`, `last.pt` |
| `train_12m.py` | Pipeline resmi **Plosagata_12M** (12,66M) dari `data/train_chat.jsonl` | `checkpoints/pure-Plosagata_12M*.pt` (+vocab & tokenizer) |
| `train_browser.py` | Pipeline resmi **Plosagata_Browser** (25,3M) dari `data/train_chat.jsonl` | `checkpoints/pure-Plosagata_Browser*.pt` (+vocab & tokenizer) |
| `train_pure.py` | Trainer GPT murni serbaguna (korpus besar / instruction-tune) | `pure-<size>*.pt` (tentukan `--out` sendiri) |
| `prepare_finetune.sh` | Siapkan bundel data untuk fine-tune di komputer GPU lain | paket data |

Contoh:

```bash
.venv/bin/python scripts/train.py --data data/contoh_dataset.csv --epochs 15 --batch 16
.venv/bin/python scripts/train_browser.py --epochs 4 --batch 8 --smoke-test
.venv/bin/python scripts/train_12m.py --epochs 5
```

## 2. Evaluasi & audit (regresi wajib lulus semua)

| Script | Yang diuji |
|---|---|
| `eval_verdict.py` | Confusion matrix + precision/recall klasifier 12M + 6 probe anekdot |
| `eval_sistem.py` | Akurasi END-TO-END (model + heuristik → `label_final` yang dipakai app) + pergeseran label |
| `eval_fiksi.py` | Deteksi artikel fiksi/rekaan + filter relevansi sumber |
| `eval_topik.py` | Topic guard multi-turn (berita vs follow-up vs off-topic) |
| `eval_security.py` | Audit guardrail keamanan 7 vektor (injection, PII, senjata, dll) |
| `eval_rag.py` | Smoke + hit@3 + latensi RAG (`--mini` untuk uji cepat) |

## 3. Data & korpus

| Script | Fungsi |
|---|---|
| `auto_collect.py` | Kolektor otomatis TurnBackHoax + RSS berita → CSV/JSONL |
| `build_corpus.py` | Bangun korpus pretrain dari Wikipedia ID (butuh internet) |
| `build_index.py` | Bangun index RAG `data/rag_index/` (BM25 + TF-IDF) |
| `csv_to_chat.py` | Ubah CSV `text,label` → JSONL percakapan instruction |
| `fetch_wiki_dump.py` | Unduh dump Wikipedia utuh → korpus GB-an (resume otomatis) |
| `fetch_vocab.py` | Tarik vocab Indonesia HF → `data/vocab_seed.txt` |
| `generate_modern_hoaks.py` | Generator dataset hoaks modern multi-turn (APK, deepfake, bansos, dll) |
| `tambah_dataset_analisa.py` | Tambah data penalaran investigasi multi-step |
| `lihat_data.py` | Cermin JSONL → tabel Markdown (`data/*.md`) biar enak dibaca |
| `gabung_dok.py` | Gabung semua dokumen → `docs/ULTIMATE_PLOSAGATA.md` (jangan edit manual) |

## 4. Pakai & koneksi

| Script | Fungsi |
|---|---|
| `pure_chat.py` | Chat CLI dengan model Plosagata sendiri (+ verdict 12M) |
| `infer.py` | Inferensi cepat: teks → label + probabilitas |
| `gpt_pure.py` | Cek arsitektur & forward pass model (kompat API lama; inti di `plosagata/`) |
| `sagi_remote.py` | Client Sagi jarak jauh (`SAGI_URL` / `SAGI_TOKEN`) |
| `ctx7.py` | Konektor Context7 — Sagi menarik dokumentasi via MCP |

## Catatan

- `scripts/` sengaja flat (tidak ada paket `__init__`); impor selalu lewat
  `ROOT` yang di-inject tiap script (`sys.path.insert(0, str(ROOT))`).
- Urutan kerja yang disarankan: `auto_collect.py` → `csv_to_chat.py` →
  `train.py` / `train_12m.py` / `train_browser.py` → semua `eval_*.py` →
  `gabung_dok.py` (sync dokumen).
- Rencana fitur historis ada di `docs/superpowers/`.
