# PlosagataA-1.0

# 🛡️ Validator Hoaks 12M + Deteksi Poster

Proyek lomba AI terapan: web validator hoaks Bahasa Indonesia.

**Alur:**
1. User input teks **atau upload poster hoaks** (gambar)
2. `ocr_poster.py` ekstrak teks dari poster (Tesseract)
3. `model.py` — Transformer classifier **~12 juta parameter** kasih skor cepat (Valid / Hoaks / Meragukan)
4. `factcheck.py` — cek silang heuristic + template integrasi LLM (Gemini/OpenAI) + saran sumber (turnbackhoax.id, cekfakta.com)
5. `app.py` — web Flask tampilkan hasil

## 1. Arsitektur model 12M

`model.py:TinyHoaxTransformer`
- vocab 5000, d_model 384, 6 layer, 6 head, ff 1280, max_len 128/256
- Token+Pos embedding + TransformerEncoder + mean-pool + MLP head
- 3 kelas: valid, hoaks, meragukan
- Total ≈ 11.6M parameter (disebut 12M)

Cek jumlah parameter:
```bash
uv run --python 3.12 python model.py
```

## 2. Install (CPU, tanpa GPU tetap jalan)

```bash
cd Projects/hoax-validator-12m
uv venv --python 3.12
uv pip install -r requirements.txt
# opsional OCR:
sudo apt install tesseract-ocr tesseract-ocr-ind
```

## 3. Training / Fine-tune

Dataset format CSV: `text,label` dengan label `valid,hoaks,meragukan`.
Contoh ada di `data/contoh_dataset.csv`.

```bash
uv run python train.py --data data/contoh_dataset.csv --epochs 15 --batch 16
# hasil: checkpoints/best.pt + vocab.json
```

Fine-tune dengan data lomba kamu:
1. Kumpulkan 500-2000 sampel hoaks Indonesia (turnbackhoax.id, cekfakta.com)
2. Tambahkan ke CSV
3. Jalankan train lagi dengan `--epochs 10 --lr 2e-4`

Untuk dataset besar, naikkan `--vocab-size 8000 --max-len 256`.

## 4. Coba inferensi

```bash
uv run python infer.py --text "BAGIKAN! Minum ini sembuh kanker 3 hari!"
```

## 5. Jalankan web

```bash
uv run python app.py
# buka http://localhost:5000
```

API:
```bash
curl -X POST -F "text=Info giveaway iPhone klik link ini" http://localhost:5000/api/check
curl -X POST -F "text=" -F "poster=@poster.jpg" http://localhost:5000/api/check
```

## 6. Sambung ke AI (nilai plus lomba)

`factcheck.py:llm_crosscheck` masih template. Untuk lomba:
- Daftar Gemini API gratis, isi `GEMINI_API_KEY`
- Prompt: "Apakah klaim ini hoaks? Beri verdict + 3 sumber URL."
- Tampilkan sumber di web sebagai bukti kebenaran.
- Tambahkan reverse image search link untuk poster:
  `https://lens.google.com/uploadbyurl?url=...`

## 7. Ide pitch lomba

> "Model 12M ringan jalan di HP/laptop tanpa internet untuk skrining cepat,
> lalu diverifikasi LLM + sumber tepercaya. Saring sebelum Sharing."

## 8. Sagi — model murni Plosagata (tanpa model luar)

```bash
# Perintah terminal (dari mana saja):
sagi                        # interaktif
sagi "Bener gak giveaway ini?"   # cek sekali

# Langsung via python:
.venv/bin/python pure_chat.py --size tiny --prompt "Sagi"
```

Training `b1` (~912M) di device GPU teman (RTX 2060 6GB+, lihat PANDUAN_FINETUNE.md):
```bash
pip install torch scikit-learn
python train_pure.py --size b1 --data data/train_chat.jsonl --epochs 3 --batch 1 --grad-accum 16
# copy pure-b1.pt balik ke sini, lalu:
.venv/bin/python pure_chat.py --size b1 --ckpt pure-b1.pt
```
Dataset: `data/train_chat.jsonl` (317 contoh). Tambahkan terus dari turnbackhoax.id.
Jujur: sebelum training besar, output model murni masih acak — itu normal.

Tambahkan: confusion matrix, demo poster, perbandingan sebelum/sesudah fine-tune.

## Struktur

- `plosagata/` — model murni (config, layers, model, guard)
- `model.py` — arsitektur 12M
- `tokenizer.py` — tokenizer BPE Indonesia
- `dataset.py` — loader CSV
- `train.py` / `train_pure.py` — training
- `infer.py` / `pure_chat.py` — prediksi & chat
- `sagi.sh` — launcher Sagi
- `ocr_poster.py` — poster -> teks
- `factcheck.py` — verifikasi silang
- `app.py` + `templates/index.html` — web
