# PlosagataA-1.0

**Sagi: AI validator hoaks Bahasa Indonesia yang kami bangun dari nol.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Models](https://img.shields.io/badge/Models-12M_%7C_92M_%7C_912M_%7C_1.5B-green)](#model)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/rewsaqy/PlosagataA-1.0)](https://github.com/rewsaqy/PlosagataA-1.0/commits)
[![Stars](https://img.shields.io/github/stars/rewsaqy/PlosagataA-1.0?style=social)](https://github.com/rewsaqy/PlosagataA-1.0/stargazers)

Proyek lomba AI terapan: sistem validasi hoaks end-to-end untuk klasifikasi teks,
OCR poster, cek fakta heuristik + sumber internet, dan chatbot khusus berita.
Seluruh arsitektur dan bobot model ditulis dan dilatih sendiri, tanpa memakai
model unduhan dari pihak mana pun.

> **Peta dokumen:** `README.md` (dokumen ini) · `howitworks/Cara_kerja_sagi.md`
> (penjelasan santai) · `docs/ARSITEKTUR.md` (teknis) · `docs/PANDUAN_FINETUNE.md`
> (training) · `howitworks/cara_pindahinIQsagi.md` (pindah PC) ·
> `howitworks/Cara_nambah_file.md` (nambah file) ·
> `howitworks/Panduan_Training_12M.md` (latih 12M) · `BUGS.md` (catatan bug jujur)

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
- [Pembuat](#pembuat)
- [Lisensi](#lisensi)

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

Label final ditentukan satu pintu via `final_label` (heuristic menang saat
kuat: skor >= 60 selalu HOAKS), dan konten deklarasi fiksi/simulasi dideteksi
via `deteksi_fiksi` agar tidak dinilai sebagai berita faktual.

---

## Model

| Model | Parameter | Peran | Status |
|---|---|---|---|
| **Plosagata_Browser** | **25.379.840** | **Flagship 25M: Hybrid RAG, Real-Time Internet Web Search & Anti-Hoaks Kekinian** | Arsitektur siap — ckpt belum ada (latih: `scripts/train_browser.py`) |
| **Plosagata_12M** | **12.665.856** | Chatbot 12M Reworked: Super Ringan, Ramah, Percakapan Santai & Cek Fakta | **Terlatih di CPU** — ckpt `checkpoints/pure-Plosagata_12M-best.pt` |
| Klasifier 12M | **11.584.515** | Penilai Valid / Hoaks / Meragukan + skor risiko 0–100 | Terlatih (Akurasi 94,6%, probe regresi 6/6) |
| Plosagata_Flash | **12.665.856** | Chat super ringan — CPU kentang, demo kilat | Kompatibel 12M |
| Plosagata_Go | **91.593.216** | Chat respon cepat untuk CPU/laptop menengah | Pra-latih + instruction-tune (eksperimental) |
| Plosagata_More | **912.008.192** | Chat besar B1, target pelatihan klaster GPU 6GB+ | Arsitektur siap |
| Plosagata_Next | **1.549.927.424** | Varian raksasa untuk GPU 16 GB+ / multi-GPU | Arsitektur saja |

Ganti model beneran di browser: buka `/models` atau ketik `/model Plosagata_Browser` di chat — engine lama dibongkar total
(RAM kembali), engine baru nyala dengan sesi fresh. Bukan dropdown pura-pura.

Verifikasi mandiri:

```bash
.venv/bin/python scripts/gpt_pure.py Plosagata_Browser
.venv/bin/python scripts/gpt_pure.py Plosagata_12M
.venv/bin/python scripts/gpt_pure.py Plosagata_Go
.venv/bin/python scripts/eval_verdict.py   # akurasi + probe regresi klasifier
.venv/bin/python scripts/eval_fiksi.py     # uji deteksi fiksi/simulasi
.venv/bin/python scripts/eval_topik.py     # uji guard + memori sesi (7/7)
.venv/bin/python scripts/eval_rag.py --mini  # smoke RAG + latensi p95
```

Detail matematis tiap lapisan: `docs/ARSITEKTUR.md`.

---

## Mulai Cepat

**Prasyarat:** Python 3.12+, sekitar 4 GB ruang kosong. GPU NVIDIA opsional
(Torch harus build CUDA agar GPU terpakai. Build `+cpu` tidak menyentuh VRAM).
Tanpa GPU tetap berjalan di CPU.

```bash
git clone git@github.com:rewsaqy/PlosagataA-1.0.git
cd PlosagataA-1.0
uv venv --python 3.12
uv pip install -r requirements.txt
# Untuk GPU (contoh CUDA 12.6):
uv pip install --python .venv/bin/python --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python scripts/infer.py --text "Giveaway iPhone, klik link ini!"
.venv/bin/python web/app.py
# → http://localhost:5000  (validator)  +  /chat  (chatbot)
```

Windows: ganti `.venv/bin/python` dengan `.venv\Scripts\python.exe`.

Opsional (OCR poster): `sudo pacman -S tesseract tesseract-data-ind`
(Debian/Ubuntu: `tesseract-ocr tesseract-ocr-ind`).

---

## Penggunaan

### Terminal: perintah `sagi`

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
| `/chat` | Chatbot: sidebar, riwayat tersimpan, tombol cepat, indikator mengetik, badge verdict, tombol salin, dukungan token remote |
| `/models` | Daftar model + ganti beneran (engine lama dibongkar, sesi fresh) |
| `/api/check` · `/api/chat` · `/api/chat/stream` (SSE) · `/api/history` · `/api/stats` · `/api/models` | API JSON |

Slash-command di dalam chat: `/models`, `/model <nama>`, `/status`, `/help`.

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

Validasi dipisah **sebelum** augmentasi, jadi angka akurasinya jujur.

### Pelatihan Mandiri (Self-Training Lokal)

#### 1. Plosagata_Browser (25.3M Core) — Model Flagship OTN 2026
Arsitektur 25M yang memadukan penalaran Transformer Causal dengan Hybrid RAG & Real-Time Web Search:

```bash
# Latihan mandiri 4 epoch (selesai ~15 menit di 16-core CPU):
.venv/bin/python scripts/train_browser.py --epochs 4 --batch 8 --lr 3e-4

# Uji coba chat interaktif setelah training:
.venv/bin/python scripts/pure_chat.py --size Plosagata_Browser --ckpt checkpoints/pure-Plosagata_Browser-best.pt
```

#### 2. Plosagata_12M (12.6M Core) — Super Ringan & Cepat
Cocok untuk demonstrasi kilat dan latihan super cepat:

```bash
# Latihan 5 epoch (selesai ~10 menit di CPU):
.venv/bin/python scripts/train_12m.py --epochs 5 --batch 16 --lr 3e-4
```

#### 3. Generator Dataset Hoaks Kekinian
Perluas dataset hoaks modern (APK sniffing, AI deepfake, phishing bansos, megathrust, dll.):

```bash
.venv/bin/python scripts/generate_modern_hoaks.py --target 5000
```

Setiap proses pelatihan otomatis menyimpan pasangan checkpoint:
- `checkpoints/pure-*-best.pt` (bobot terbaik berdasarkan validation loss terendah)
- `checkpoints/pure-*.vocab.pt` & `checkpoints/pure-*.tokenizer.json` (kamus token pasangan)
- Menjalankan *smoke test* otomatis di terminal begitu training selesai.

---

## Data & Bahasa

| Dataset | Isi | Untuk |
|---|---|---|
| `data/train_chat.jsonl` | **10.866+ percakapan** (kasus hoaks modern, APK sniffing, deepfake, gaya Sagi, dll.) | Training Causal LM (Plosagata_Browser & 12M) |
| `data/contoh_dataset.csv` | 617 baris (322 valid / 165 hoaks / 130 meragukan) | Klasifier 12M |
| `data/contoh_chat.jsonl` | 249 percakapan dasar (slang, penolakan sopan, identitas, sapaan hangat) | Ground truth percakapan |
| `data/vocab_seed.txt` | 106 ribu kata dasar Indonesia | Build kamus subword BPE otomatis |
| `data/korpus.txt` | ±12 GB teks Wikipedia campur ID + EN (opsional, tidak masuk Git) | Pra-latih korpus raksasa |

- `plosagata/slang.py` berisi 2.173 entri slang → formal yang dipakai saat
  training **dan** inferensi agar konsisten.
- Tokenizer BPE mengeja kata asing per potongan (tanpa OOV); vocab lebih
  besar via `--vocab-size` + latih ulang.
- Koleksi data otomatis (TurnBackHoax + RSS, ada dedupe):
  `python scripts/auto_collect.py --tbh-pages 2 --rss 1`
- Regenerasi chat: `python3 scripts/csv_to_chat.py` lalu gabung
  `data/chat_hoaks.jsonl` + `data/contoh_chat.jsonl` menjadi
  `data/train_chat.jsonl`.

---

## Penyebaran & Akses Jarak Jauh

Jalankan Sagi di komputer GPU, akses dari mana saja:

```bash
# Komputer teman (server):
SAGI_SIZE=Plosagata_Go SAGI_CKPT=pure-Go-sagi-best.pt SAGI_TOKEN=RAHASIA \
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
├── sagi/         Mesin chat (engine streaming, prompt format training, sesi, sitasi)
├── scripts/      Antarmuka baris perintah (train, infer, chat, korpus, ctx7, remote)
├── web/          app.py, chatbot.py (shim → sagi), templates/
├── data/         Dataset + bank kata
├── docs/         ARSITEKTUR.md, PANDUAN_FINETUNE.md
├── sagi.sh       Peluncur (mis. SUPER+S di Hyprland)
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

Seluruh skrip kembali otomatis ke akar repo, jadi dapat dijalankan dari direktori
mana pun. Dokumentasi arsitektur per berkas: [`docs/ARSITEKTUR.md`](docs/ARSITEKTUR.md).

---

## Peta Jalan

- [x] Klasifier 12M terlatih (val 94,6%) + web validator + OCR poster
- [x] Arsitektur Plosagata Go/More/Next/Browser + guard topik + slang + retrieval sumber
- [x] Plosagata_12M chat-tuning di CPU (ckpt `pure-Plosagata_12M-best.pt`)
- [x] Web chatbot + API + akses remote bertoken
- [~] Checkpoint Go dari GPU teman (`pure-Go-sagi-best.pt`, via Drive — belum dievaluasi formal)
- [ ] Pra-latih More di GPU (korpus Wikipedia, ±5 hari)
- [ ] Instruction-tuning → `pure-Plosagata_More.pt` rilis pertama
- [ ] Evaluasi: matriks konfusi, uji 100 klaim nyata, demo poster

---

## Pembuat

- **rewsaqy** — pemilik ide, arsitektur, data, training, dan pengujian.
- **OpenCode (AI assistant)** — implementasi kode, audit, dan dokumentasi
  bersama pemilik selama pembangunan.

Proyek personal — tidak menerima kontribusi luar. Riwayat kerja tercatat
penuh via `git log`.

---

## Lisensi

Proyek ini berlisensi MIT, lihat [`LICENSE`](LICENSE).

> *Saring sebelum Sharing.*
