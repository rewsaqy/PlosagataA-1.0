# PlosagataA-1.0 — Sagi, AI Validator Hoaks Buatan Sendiri

> Web + chatbot validator hoaks Bahasa Indonesia. 100% kode dan model buatan
> sendiri — tidak ada model download dari internet.

## Daftar isi

1. [Dua otak Sagi](#1-dua-otak-sagi) · 2. [Mulai cepat](#2-mulai-cepat) ·
   3. [Perintah `sagi`](#3-perintah-sagi) · 4. [Web](#4-web) ·
   5. [Training](#5-training) · 6. [Data & bahasa](#6-data--bahasa) ·
   7. [Remote & migrasi](#7-remote--migrasi) · 8. [Struktur folder](#8-struktur-folder)

## 1. Dua otak Sagi

| | Klasifier 12M | Plosagata (chat) |
|---|---|---|
| Fungsi | Menilai: Valid / Hoaks / Meragukan + skor 0-100 | Ngobrol +(generate) jawaban |
| Ukuran | **11.584.515** param | **Go 91.593.216** · **More 912.008.192** · **Next 1.549.927.424** |
| File | `hoaxcore/` + `checkpoints/best.pt` | `plosagata/` + `pure-*.pt` (setelah training) |
| Status | ✅ Terlatih (val 97%), siap demo | 🟡 Arsitektur jadi, menunggu training GPU |

Cek angkanya sendiri: `.venv/bin/python scripts/gpt_pure.py Plosagata_More`

Alur: teks/poster → tokenizer (kata→angka) → 12M kasih label → fact-check
kasih skor + sumber internet → tampil di web / chat.

## 2. Mulai cepat (Linux, CPU saja cukup)

```bash
git clone git@github.com:rewsaqy/PlosagataA-1.0.git
cd PlosagataA-1.0
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python scripts/infer.py --text "Giveaway iPhone klik link ini!"
.venv/bin/python web/app.py   # http://localhost:5000 dan /chat
```

## 3. Perintah `sagi` (terminal mana saja)

```bash
sagi                          # ngobrol interaktif (Plosagata_More)
sagi "Bener gak giveaway ini?"  # jawab sekali lalu keluar
```

Sapaan (`halo`, `hi`, `kawan`...) dijawab identitas Sagi, topik di luar
berita ditolak sopan + diarahkan balik, berita diproses model + verdict 12M.
Panggil namanya di tengah kalimat pun bisa: `Sagi, cek berita ini!`

## 4. Web

- `/` — validator: tempel teks / upload poster → label + skor + riwayat + statistik
- `/chat` — chatbot ala ChatGPT: sidebar, riwayat tersimpan, tombol cepat,
  indikator mengetik, badge verdict, tombol salin, pilih model
  (Go/More/Next), kolom token untuk server remote
- `/api/check`, `/api/chat`, `/api/history`, `/api/stats` — API JSON

Stabilitas: reloader mati default (bentrok thread torch), thread dibatasi.
Butuh auto-reload saat ngoding UI: `SAGI_RELOAD=1`.

## 5. Training

**12M (CPU bisa):**
```bash
.venv/bin/python scripts/train.py --data data/contoh_dataset.csv --epochs 15
```

**Plosagata di GPU teman — rencana 5 hari (RTX 3080 disarankan, 2060 bisa mode hemat):**
```bash
# hari 1: .venv/bin/python scripts/build_corpus.py --out data/korpus.txt
# hari 1-4 (pretrain dari nol):
.venv/bin/python scripts/train_pure.py --size Plosagata_More --corpus data/korpus.txt \
  --epochs 1 --batch 2 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit
# hari 5 (jadi Sagi):
.venv/bin/python scripts/train_pure.py --size Plosagata_More --data data/train_chat.jsonl \
  --epochs 5 --batch 1 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \
  --ckpt pure-Plosagata_More-pretrain.pt --out pure-Plosagata_More.pt
```
Detail per OS (CachyOS/Ubuntu/Windows/WSL2), OOM darurat, dan checklist bawa
pulang 3 file (`pure-*.pt` + `*.vocab.pt` + `checkpoints/`) ada di tutorial
lama — tanya saya kalau butuh dibacakan lagi.

**Mode berpikir-murni** (semua pesan lewat model, tanpa template — butuh model terlatih):
```bash
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --pure-think
SAGI_PURE_THINK=1 .venv/bin/python web/app.py
```

## 6. Data & bahasa

- `data/contoh_dataset.csv` (215, seimbang 75/75/65) → klasifier
- `data/contoh_chat.jsonl` (132) + generate → `data/train_chat.jsonl` (347) → Sagi
- `data/vocab_seed.txt` (400+ kata, otomatis dipakai) → kata penting jadi 1 token
- `plosagata/slang.py` (199 entri) → slang dinormalisasi saat training DAN chat
- Tambah data lalu regenerate:
  `python3 scripts/csv_to_chat.py && cat data/chat_hoaks.jsonl data/contoh_chat.jsonl > data/train_chat.jsonl`

## 7. Remote & migrasi (Sagi di GPU teman, dichat dari rumah)

```bash
# di komputer teman:
SAGI_SIZE=Plosagata_More SAGI_CKPT=pure-Plosagata_More.pt SAGI_TOKEN=RAHASIA \
  .venv/bin/python web/app.py
```
Dua-duanya install Tailscale (gratis) akun sama → dari browser rumah buka
`http://100.x.y.z:5000/chat`, isi token sekali. Terminal:
`SAGI_URL=... SAGI_TOKEN=... python scripts/sagi_remote.py`.

## 8. Struktur folder

```
PlosagataA-1.0/
├── plosagata/   chat murni (config, layers, model, guard, slang)
├── hoaxcore/    klasifier 12M (model, tokenizer, dataset, factcheck, ocr, retrieval)
├── scripts/     CLI (train, infer, chat, korpus, ctx7, remote)
├── web/         app.py, chatbot.py (SagiEngine), templates/
├── data/        dataset + bank kata
└── docs/        ARSITEKTUR.md (bedah per file), PANDUAN_FINETUNE.md
```

> Pitch lomba: "Model 12M jalan offline di laptop untuk skrining cepat,
> Sagi 912M hasil training sendiri untuk verifikasi. Saring sebelum Sharing."
