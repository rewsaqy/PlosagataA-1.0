# PlosagataA-1.0

**Sagi: AI validator hoaks Bahasa Indonesia yang kami bangun dari nol.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Models](https://img.shields.io/badge/Models-12M_%7C_92M_%7C_912M_%7C_1.5B-green)](#model)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/rewsaqy/PlosagataA-1.0)](https://github.com/rewsaqy/PlosagataA-1.0/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/rewsaqy/PlosagataA-1.0)](https://github.com/rewsaqy/PlosagataA-1.0/commits)
[![Stars](https://img.shields.io/github/stars/rewsaqy/PlosagataA-1.0?style=social)](https://github.com/rewsaqy/PlosagataA-1.0/stargazers)

Proyek lomba AI terapan: sistem validasi hoaks end-to-end untuk klasifikasi teks,
OCR poster, cek fakta heuristik + sumber internet, dan chatbot khusus berita.
Seluruh arsitektur dan bobot model ditulis dan dilatih sendiri, tanpa memakai
model unduhan dari pihak mana pun.

> **Peta dokumen:** `README.md` (dokumen ini) · `howitworks/Cara_kerja_sagi.md`
> (penjelasan santai) · `docs/ARSITEKTUR.md` (teknis) · `docs/PANDUAN_FINETUNE.md`
> (training) · `howitworks/cara_pindahinIQsagi.md` (pindah PC) ·
> `howitworks/Cara_nambah_file.md` (nambah file)

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
- [Kontributor](#kontributor)
- [Berkontribusi](#berkontribusi)
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
| Klasifier 12M | **11.584.515** | Penilai Valid / Hoaks / Meragukan + skor 0–100 | Terlatih (val 94,9%, probe regresi 6/6) |
| Plosagata_Go | **91.593.216** | Chat ringan untuk CPU/laptop | Pra-latih + instruction-tune (eksperimental) |
| Plosagata_More | **912.008.192** | Chat utama Sagi, target training GPU | Arsitektur siap |
| Plosagata_Next | **1.549.927.424** | Varian besar untuk GPU 12 GB+ / klaster | Arsitektur saja |

Verifikasi mandiri:

```bash
.venv/bin/python scripts/gpt_pure.py Plosagata_Go
.venv/bin/python scripts/gpt_pure.py Plosagata_More
.venv/bin/python scripts/gpt_pure.py Plosagata_Next
.venv/bin/python scripts/eval_verdict.py   # akurasi + probe regresi klasifier
.venv/bin/python scripts/eval_fiksi.py     # uji deteksi fiksi/simulasi
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
.venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
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
| `/chat` | Chatbot: sidebar, riwayat tersimpan, tombol cepat, indikator mengetik, badge verdict, tombol salin, pemilih model, dukungan token remote |
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

Validasi dipisah **sebelum** augmentasi, jadi angka akurasinya jujur.

### Plosagata: pra-latih lalu instruction-tuning (GPU disarankan)

```bash
# 1. Korpus Wikipedia Indonesia (sumber: wikimedia/wikipedia):
.venv/bin/python scripts/build_corpus.py --out data/korpus.wiki.txt --limit 2000
cat data/korpus.txt data/korpus.wiki.txt > data/korpus.full.txt
# 2. Pra-latih dari nol (ada split validasi otomatis):
.venv/bin/python scripts/train_pure.py --size Plosagata_Go --corpus data/korpus.full.txt \
  --epochs 1 --batch 16 --fp16 --val-split 0.02 --out pure-Go-pretrain.pt
# 3. Instruction-tuning menjadi Sagi (WAJIB pakai vocab pretrain agar embedding cocok):
.venv/bin/python scripts/train_pure.py --size Plosagata_Go --data data/train_chat.jsonl \
  --epochs 5 --batch 8 --fp16 --lr 2e-4 --val-split 0.1 \
  --ckpt pure-Go-pretrain-best.pt --vocab-file pure-Go-pretrain.pt.vocab.pt \
  --out pure-Go-sagi.pt
```

Setiap `--out X.pt` otomatis menyimpan `X.pt.vocab.pt` di sebelahnya, dan
checkpoint val terbaik (`X-best.pt` + `X-best.pt.vocab.pt`). Jangan pernah
memisahkan `.pt` dari `.vocab.pt` pasangannya.

Bobot `.pt` tidak masuk Git (lihat `.gitignore`). Bagikan via
Release/Drive. Panduan pindah PC: `howitworks/cara_pindahinIQsagi.md`.

Mode berpikir-murni (semua pesan lewat model, perlu model terlatih):

```bash
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --ckpt pure-Go-sagi-best.pt --pure-think
SAGI_PURE_THINK=1 .venv/bin/python web/app.py
```

---

## Data & Bahasa

| Dataset | Isi | Untuk |
|---|---|---|
| `data/contoh_dataset.csv` | 514 baris (tumbuh via `auto_collect.py`) | Klasifier |
| `data/contoh_chat.jsonl` | 187 percakapan (slang, penolakan, sapaan) | Gaya Sagi |
| `data/train_chat.jsonl` | 701 (gabungan + generate) | Instruction-tuning |
| `data/vocab_seed.txt` | Bank kata inti | Build vocab otomatis |

- `plosagata/slang.py` berisi 199 entri slang → formal yang dipakai saat
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
├── scripts/      Antarmuka baris perintah (train, infer, chat, korpus, ctx7, remote)
├── web/          app.py, chatbot.py (SagiEngine), templates/
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

- [x] Klasifier 12M terlatih (val 94,9%, probe 6/6) + web validator + OCR poster
- [x] Arsitektur Plosagata Go/More/Next + guard topik + slang + retrieval sumber
- [x] Web chatbot + API + akses remote bertoken
- [x] Pra-latih Go di GPU (korpus Wikipedia 2000 artikel) + instruction-tuning bervalidasi
- [ ] Pra-latih More di GPU (korpus Wikipedia, ±5 hari)
- [ ] Instruction-tuning → `pure-Plosagata_More.pt` rilis pertama
- [ ] Evaluasi: matriks konfusi, uji 100 klaim nyata, demo poster

---

## Kontributor

- **rewsaqy**: arsitektur, model, dan training awal
- **ZxWyvern**: training GPU, data, dan pengembangan lanjutan

Lihat riwayat kontribusi lengkap via `git shortlog -sne`.

---

## Berkontribusi

Kontribusi terbuka. Baca [`CONTRIBUTING.md`](CONTRIBUTING.md) dulu:
alur fork/branch/PR, evaluasi wajib (`eval_verdict.py`, `eval_fiksi.py`,
lapor train + val loss), dan aturan bobot (tidak ada `.pt` di Git).

---

## Lisensi

Proyek ini berlisensi MIT, lihat [`LICENSE`](LICENSE).

> *Saring sebelum Sharing.*
