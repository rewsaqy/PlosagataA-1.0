# PlosagataA-1.0 🛡️

**Sagi — AI validator hoaks Bahasa Indonesia, dibangun dari nol.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![Models](https://img.shields.io/badge/Models-12M_%7C_92M_%7C_912M_%7C_1.5B-green)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

Proyek lomba AI terapan: sistem validasi hoaks end-to-end — klasifikasi teks,
OCR poster, cek fakta heuristik + sumber internet, dan chatbot khusus berita.
Seluruh arsitektur dan bobot model ditulis dan dilatih sendiri, tanpa memakai
model unduhan dari pihak mana pun.

---

## Daftar Isi

- [Arsitektur](#arsitektur)
- [Model](#model)
- [Mulai Cepat](#mulai-cepat)
- [Penggunaan](#penggunaan)
- [Pelatihan](#pelatihan)
- [Data & Bahasa](#data--bahasa)
- [Penyebaran & Akses Jarak Jauh](#penyebaran--akses-jarak-jauh)
- [Struktur Repositori](#struktur-repositori)
- [Peta Jalan](#peta-jalan)

---

## Arsitektur

```
                    ┌──────────────┐
  Teks ────────────►│  Tokenizer   │  kata → angka (BPE, 5K/8K/16K)
  Poster ──OCR─────►│  (hoaxcore)  │
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌───────────┐ ┌───────────┐
     │ Klasifier  │ │ Fact-     │ │ Plosagata │
     │ 12M        │ │ check +   │ │ (chat)    │
     │ Valid/     │ │ Retrieval │ │ tebak kata│
     │ Hoaks/     │ │ sumber    │ │ berikut   │
     │ Meragukan  │ │ internet  │ │           │
     └────────────┘ └───────────┘ └───────────┘
              └────────────┼────────────┘
                           ▼
              ┌────────────────────────┐
              │  Web / API / Terminal  │
              │  /  /chat  sagi  :5000 │
              └────────────────────────┘
```

Lapisan **guard** (`plosagata/guard.py`) memastikan Sagi hanya melayani topik
berita: sapaan dibalas identitas, off-topic ditolak sopan, klaim diproses penuh
(slang → tokenizer → model → verdict 12M + sumber).

---

## Model

| Model | Parameter | Peran | Status |
|---|---|---|---|
| Klasifier 12M | **11.584.515** | Penilai Valid / Hoaks / Meragukan + skor 0–100 | ✅ Terlatih (val 97%) |
| Plosagata_Go | **91.593.216** | Chat ringan — CPU/laptop | 🟡 Arsitektur siap |
| Plosagata_More | **912.008.192** | Chat utama Sagi — target training GPU | 🟡 Arsitektur siap |
| Plosagata_Next | **1.549.927.424** | Varian besar — GPU 12 GB+ / klaster | ⚪ Arsitektur saja |

Verifikasi mandiri:

```bash
.venv/bin/python scripts/gpt_pure.py Plosagata_Go
.venv/bin/python scripts/gpt_pure.py Plosagata_More
.venv/bin/python scripts/gpt_pure.py Plosagata_Next
```

Detail matematis tiap lapisan: `docs/ARSITEKTUR.md`.

---

## Mulai Cepat

**Prasyarat:** Linux, Python 3.12, (±4 GB ruang). Tanpa GPU tetap berjalan.

```bash
git clone git@github.com:rewsaqy/PlosagataA-1.0.git
cd PlosagataA-1.0
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python scripts/infer.py --text "Giveaway iPhone, klik link ini!"
.venv/bin/python web/app.py
# → http://localhost:5000  (validator)  +  /chat  (chatbot)
```

Opsional (OCR poster): `sudo pacman -S tesseract tesseract-data-ind`
(Debian/Ubuntu: `tesseract-ocr tesseract-ocr-ind`).

---

## Penggunaan

### Terminal — perintah `sagi`

```bash
sagi                             # mode interaktif (Plosagata_More)
sagi "Bener gak giveaway ini?"   # satu pertanyaan, langsung keluar
```

Contoh perilaku: `Sagi` → identitas; `Buatkan kode python` → ditolak sopan;
klaim berita → diproses model + verdict 12M.

### Web

| Rute | Fungsi |
|---|---|
| `/` | Validator: tempel teks / unggah poster → label, skor, riwayat, statistik |
| `/chat` | Chatbot ala ChatGPT: sidebar, riwayat tersimpan, tombol cepat, indikator mengetik, badge verdict, tombol salin, pemilih model, dukungan token remote |
| `/api/check` · `/api/chat` · `/api/history` · `/api/stats` | API JSON |

Stabilitas server: reloader nonaktif bawaan (konflik thread PyTorch),
thread CPU dibatasi. Aktifkan reload saat pengembangan UI:
`SAGI_RELOAD=1`.

### Utilitas

```bash
python3 scripts/ctx7.py "Flask" "streaming"   # cari dokumentasi via Context7
python scripts/sagi_remote.py "Sagi, halo!"   # client chat ke server jauh
```

---

## Pelatihan

### Klasifier 12M (CPU memadai)

Dataset `data/contoh_dataset.csv` (`text,label` → valid/hoaks/meragukan):

```bash
.venv/bin/python scripts/train.py --data data/contoh_dataset.csv --epochs 15 --batch 16
# → checkpoints/best.pt + checkpoints/vocab.json
```

Validasi dipisah **sebelum** augmentasi — angka akurasinya jujur.

### Plosagata — rencana 5 hari di GPU (RTX 3080 disarankan; 2060 mode hemat)

```bash
# Hari 1 — korpus Wikipedia Indonesia:
.venv/bin/python scripts/build_corpus.py --out data/korpus.txt
# Hari 1–4 — pra-latih dari nol:
.venv/bin/python scripts/train_pure.py --size Plosagata_More --corpus data/korpus.txt \
  --epochs 1 --batch 2 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \
  --out pure-Plosagata_More-pretrain.pt
# Hari 5 — instruction-tuning menjadi Sagi:
.venv/bin/python scripts/train_pure.py --size Plosagata_More --data data/train_chat.jsonl \
  --epochs 5 --batch 1 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \
  --ckpt pure-Plosagata_More-pretrain.pt --out pure-Plosagata_More.pt
```

Panduan per OS (CachyOS/Ubuntu/Windows/WSL2), darurat OOM, dan checklist
bawa pulang 3 berkas (`pure-*.pt`, `*.vocab.pt`, `checkpoints/`) — lihat
bagian tutorial di riwayat README atau `docs/PANDUAN_FINETUNE.md`.

Mode berpikir-murni (semua pesan lewat model, perlu model terlatih):

```bash
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --pure-think
SAGI_PURE_THINK=1 .venv/bin/python web/app.py
```

---

## Data & Bahasa

| Dataset | Isi | Untuk |
|---|---|---|
| `data/contoh_dataset.csv` | 215 baris, seimbang 75/75/65 | Klasifier |
| `data/contoh_chat.jsonl` | 132 percakapan (slang, penolakan, sapaan) | Gaya Sagi |
| `data/train_chat.jsonl` | 347 (gabungan + generate) | Instruction-tuning |
| `data/vocab_seed.txt` | 400+ kata inti | Build vocab otomatis |

- `plosagata/slang.py` — 199 entri slang → formal, diterapkan saat
  training **dan** inferensi agar konsisten.
- Tokenizer BPE mengeja kata asing per potongan (tanpa OOV); vocab lebih
  besar via `--vocab-size` + latih ulang.
- Regenerasi: `python3 scripts/csv_to_chat.py && cat data/chat_hoaks.jsonl data/contoh_chat.jsonl > data/train_chat.jsonl`

---

## Penyebaran & Akses Jarak Jauh

Jalankan Sagi di komputer GPU, akses dari mana saja seperti ChatGPT:

```bash
# Komputer teman (server):
SAGI_SIZE=Plosagata_More SAGI_CKPT=pure-Plosagata_More.pt SAGI_TOKEN=RAHASIA \
  .venv/bin/python web/app.py
```

Pasang Tailscale (gratis) dengan akun yang sama di kedua mesin, lalu dari
peramban rumah buka `http://100.x.y.z:5000/chat` dan isi token sekali
(tersimpan otomatis). Proteksi: endpoint chat memakai token
(`X-Sagi-Token`), CORS terbuka, bobot `.pt` tidak pernah masuk Git.

---

## Struktur Repositori

```
PlosagataA-1.0/
├── plosagata/    Model chat murni (config, layers, model, guard, slang)
├── hoaxcore/     Klasifier 12M (model, tokenizer, dataset, factcheck, ocr, retrieval)
├── scripts/      Antarmuka baris perintah (train, infer, chat, korpus, ctx7, remote)
├── web/          app.py, chatbot.py (SagiEngine), templates/
├── data/         Dataset + bank kata
├── docs/         ARSITEKTUR.md, PANDUAN_FINETUNE.md
├── sagi.sh       Peluncur (mis. SUPER+S di Hyprland)
├── requirements.txt
└── README.md
```

Seluruh skrip kembali otomatis ke akar repo — dapat dijalankan dari direktori
mana pun. Dokumentasi arsitektur per berkas: [`docs/ARSITEKTUR.md`](docs/ARSITEKTUR.md).

---

## Peta Jalan

- [x] Klasifier 12M terlatih (val 97%) + web validator + OCR poster
- [x] Arsitektur Plosagata Go/More/Next + guard topik + slang + retrieval sumber
- [x] Web chatbot + API + akses remote bertoken
- [ ] Pra-latih More di GPU (korpus Wikipedia, ±5 hari)
- [ ] Instruction-tuning → `pure-Plosagata_More.pt` rilis pertama
- [ ] Evaluasi: matriks konfusi, uji 100 klaim nyata, demo poster

> *Saring sebelum Sharing.*
