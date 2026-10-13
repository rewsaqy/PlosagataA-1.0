# PlosagataA-1.0 — Sagi, AI Validator Hoaks Buatan Sendiri

Proyek lomba AI terapan: web + chatbot validator hoaks Bahasa Indonesia.
100% kode dan model buatan sendiri — tidak ada model download dari internet.

## Dua model Sagi (parameter terverifikasi)

| Model | Parameter | File | Fungsi | Status |
|---|---|---|---|---|
| Klasifier hoaks 12M | **11.584.515** (~11,6M) | `hoaxcore/`, `checkpoints/best.pt` | Nilai teks: Valid / Hoaks / Meragukan. Otak web + Sagi | Terlatih, val 97%, siap demo |
| Plosagata_Go (small) — respon cepat | **91.593.216** (~92M) | `plosagata/` | Chat ringan di CPU/laptop | Jalan, menunggu training lanjutan |
| Plosagata_More (B1) — medium | **912.008.192** (~912M) | `plosagata/`, `pure-Plosagata_More.pt` (nanti) | Chat utama Sagi, target training GPU | Arsitektur jadi, menunggu training |
| Plosagata_Next (b1x38) — besar | **1.549.927.424** (~1,55B) | `plosagata/` | Cadangan cluster (GPU 12GB+) | Arsitektur saja |

Cek sendiri jumlahnya:
```bash
.venv/bin/python scripts/gpt_pure.py Plosagata_Go    # 91.593.216
.venv/bin/python scripts/gpt_pure.py Plosagata_More  # 912.008.192
.venv/bin/python scripts/gpt_pure.py Plosagata_Next  # 1.549.927.424
```

**Alur kerja:** user kirim teks/poster → `ocr_poster.py` (poster jadi teks)
→ model 12M kasih skor → `factcheck.py` kasih verdict + sumber
→ tampil di web (`app.py`) atau chat Sagi (`sagi`).

## Model 1 — Klasifier 12M (siap pakai sekarang)

Arsitektur `model.py:TinyHoaxTransformer`: vocab 5000, dimensi 384,
6 layer, 6 head, feedforward 1280, 3 kelas (valid/hoaks/meragukan).

**Install (Linux, CPU saja cukup):**
```bash
cd PlosagataA-1.0
uv venv --python 3.12
uv pip install -r requirements.txt
# opsional OCR poster:
sudo apt install tesseract-ocr tesseract-ocr-ind   # Arch: sudo pacman -S tesseract tesseract-data-ind
```

**Pakai:**
```bash
.venv/bin/python scripts/infer.py --text "BAGIKAN! Minum ini sembuh kanker 3 hari!"
.venv/bin/python web/app.py   # buka http://localhost:5000 (teks + upload poster)
```

**Training ulang / fine-tune (data di `data/contoh_dataset.csv`, format `text,label`):**
```bash
.venv/bin/python scripts/train.py --data data/contoh_dataset.csv --epochs 15 --batch 16
# hasil: checkpoints/best.pt + checkpoints/vocab.json
```

## Model 2 — Plosagata b1 ~912M (Sagi chat, butuh training GPU)

Arsitektur `plosagata/`: decoder-only (mirip GPT), vocab 16000, dimensi 2048,
22 layer, 16 head. Murni kode sendiri + topic guard (hanya bahas berita).

**Pakai sekarang (CPU bisa, masih bodoh sebelum training):**
```bash
sagi                 # perintah terminal, dari folder mana saja
sagi "Sagi"          # Sagi memperkenalkan diri
sagi "Bener gak giveaway ini?"   # forward beneran via 912M (~7 dtk/token di CPU)
```

**Training = di komputer GPU teman (lihat tutorial 5 hari di bawah).**
Hasilnya `pure-Plosagata_More.pt` (~3,6GB) dicopy balik, lalu:
```bash
.venv/bin/python scripts/pure_chat.py --size Plosagata_More --ckpt pure-Plosagata_More.pt
```

## Tutorial training 5 hari di komputer teman (RTX 3080 disarankan, 2060 bisa)

Kenapa 3080: 10/12GB VRAM vs 6GB, ~2x lebih cepat. 2060 wajib mode hemat.

**Hari 0 — setup (Linux/CachyOS):**
```bash
nvidia-smi   # WAJIB keluar tabel GPU. Kalau error = driver belum kepasang, berhenti di sini
sudo pacman -Syu python git        # Ubuntu: sudo apt install python3 git
curl -LsSf astral.sh/uv/install.sh | sh
git clone git@github.com:rewsaqy/PlosagataA-1.0.git
cd PlosagataA-1.0
uv venv --python 3.12
uv pip install torch scikit-learn datasets bitsandbytes
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"  # WAJIB True
```
Cek gagal? 90% karena driver NVIDIA / Python >3.13 (torch belum support) — pakai Python 3.12.

**Hari 0 — setup Windows (kalau temannya Windows):**
- *Disarankan:* install WSL2 Ubuntu (`wsl --install -d Ubuntu`), lalu ikut langkah Linux di atas 1:1.
- *Native:* `winget install Python.Python.3.12 Git.Git`, `pip install torch scikit-learn datasets`
  (TANPA bitsandbytes — tidak support Windows; ganti `--optim adam8bit` → `--optim adamw`
  di semua perintah bawah, plus `--batch 1 --max-len 512` karena makan VRAM lebih).

**Hari 1 — bangun korpus (butuh internet, sekali saja):**
```bash
.venv/bin/python scripts/build_corpus.py --out data/korpus.txt
# Cek: ls -lh data/korpus.txt  (harus ratusan MB; kalau KB = gagal, ulangi)
# + tempel artikel FAKTA dari turnbackhoax.id ke file yang sama
```

**Hari 1-4 — pretrain b1 dari nol (jangan dimatikan):**
```bash
# RTX 3080:
.venv/bin/python scripts/train_pure.py --size Plosagata_More --corpus data/korpus.txt --epochs 1 \
  --batch 2 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \
  --out pure-Plosagata_More-pretrain.pt
# RTX 2060: sama persis tapi --batch 1 --grad-accum 32 --max-len 512
# Cek jalan: loss harus TURUN tiap jam (8.5 -> 6 -> 5...). Kalau stuck/nan = kabari saya.
# Jangan tutup terminal; pakai tmux/screen biar aman ditinggal tidur.
```

**Hari 5 — jadikan Sagi + bawa pulang:**
```bash
.venv/bin/python scripts/train_pure.py --size Plosagata_More --data data/train_chat.jsonl --epochs 5 \
  --batch 1 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \
  --ckpt pure-Plosagata_More-pretrain.pt --out pure-Plosagata_More.pt
# copy pure-Plosagata_More.pt (~3,6GB) + pure-Plosagata_More.vocab.pt + checkpoints/ ke flashdisk exFAT/NTFS, bawa pulang
.venv/bin/python scripts/pure_chat.py --size Plosagata_More --ckpt pure-Plosagata_More.pt   # tes ngobrol di sana
```

**Kepintaran = 3 file, bukan install ulang (checklist bawa pulang):**
1. `pure-Plosagata_More.pt` (~3,6GB) — bobot 912M = otaknya. Wajib.
2. `pure-Plosagata_More.vocab.pt` (KB-an) — tokenizer yang dipakai saat training. Wajib,
   kalau tidak cocok output jadi sampah walau bobot benar.
3. `checkpoints/` (~90MB) — model 12M kalau ikut dilatih ulang di sana.
- Flashdisk WAJIB exFAT/NTFS — FAT32 menolak file >4GB (b1 3,6GB mepet bahaya).
- Alternatif: Google Drive, atau LAN via `python3 -m http.server` + download.
- DILARANG `git push` file `.pt` — GitHub menolak >100MB. Pakai Release/Drive.
- Sampai di rumah: taruh ketiganya Sejajar di folder proyek, jalan seperti biasa —
  Sagi langsung sepintar di komputer teman.

**Kalau OOM (CUDA out of memory):** urutan darurat: `--batch 1` → `--max-len 512` →
`--max-len 256` → ganti `--optim adamw`. Jangan naikkan batch dulu.

## Data & kosakata (biar Sagi paham slang)

- **Nambah contoh:** `data/contoh_dataset.csv` (`text,label`) untuk klasifier,
  `data/contoh_chat.jsonl` (1 baris = 1 JSON percakapan) untuk chat. Lalu:
  ```bash
  python3 scripts/csv_to_chat.py
  cat data/chat_hoaks.jsonl data/contoh_chat.jsonl > data/train_chat.jsonl
  ```
  Sekarang: 215 CSV (75 valid / 75 hoaks / 65 meragukan) + 107 chat = 322.
- **Slang modern (`slang.py`):** 60+ kata (ngab, gaskeun, spill, maxwin, wkwk...)
  dinormalisasi otomatis sebelum masuk model, saat training DAN chat.
  Nambah kata baru = 1 baris di dict `SLANG`, tanpa retrain.
- **Kosakata model:** tokenizer BPE mengeja kata asing per potongan jadi
  kata baru langsung bisa. Mau vocab lebih besar: `--vocab-size 16000` + retrain.

## Ide pitch lomba

> "Model 12M ringan jalan di laptop tanpa internet untuk skrining cepat,
> Sagi 912M hasil training sendiri untuk diajak verifikasi berita.
> Saring sebelum Sharing."

## Struktur folder (rapih)

```
PlosagataA-1.0/
├── plosagata/      model murni b1 (config, layers, model, guard, slang)
├── hoaxcore/       klasifier 12M (model, tokenizer, dataset, factcheck, ocr)
├── scripts/        semua CLI (train, infer, chat, korpus, csv_to_chat)
├── web/            web validator + chatbot (app.py, chatbot.py, templates/)
├── data/           dataset + bank kata (csv, jsonl, vocab_seed.txt)
├── docs/           panduan fine-tune
├── sagi.sh         launcher (SUPER+S), requirements.txt, README.md
```

- `plosagata/` — model murni b1 (config, layers, model, guard, slang)
- `hoaxcore/` — klasifier 12M (model, tokenizer, dataset, factcheck, ocr_poster)
- `scripts/` — `train.py`, `train_pure.py`, `infer.py`, `pure_chat.py`, `gpt_pure.py`, `csv_to_chat.py`, `build_corpus.py`
- `web/` — `app.py`, `chatbot.py`, `templates/index.html`, `templates/chat.html`
- `data/` — `contoh_dataset.csv`, `contoh_chat.jsonl`, `train_chat.jsonl`, `vocab_seed.txt`
- Semua script bisa dijalankan dari folder mana saja (otomatis kembali ke root repo).
