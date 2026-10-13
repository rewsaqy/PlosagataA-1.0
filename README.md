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

## 8. Chat AI (nyala dulu, train nanti di GPU)

```bash
# Langsung nyala, tanpa download (template lokal + validator 12M):
python3 chat.py --dummy
python3 chat.py --dummy --once "Bener gak giveaway iPhone klik link ini?"

# Chat beneran kayak ChatGPT (download Qwen 0.5B sekali, ~1GB):
.venv/bin/python chat.py
.venv/bin/python chat.py --once "Halo! Jelaskan hoaks dalam 1 kalimat."

# Versi 1B pas:
.venv/bin/python chat.py --model meta-llama/Llama-3.2-1B-Instruct
# butuh: huggingface-cli login (akun HF + accept license Llama)
```

Training di device lain (GPU Colab/Kaggle):
```bash
pip install transformers peft trl datasets accelerate bitsandbytes
python train_chat_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --data data/contoh_chat.jsonl --out lora-hoax-chat
# copy folder lora-hoax-chat/ balik ke sini untuk inferensi
```
Dataset contoh: `data/contoh_chat.jsonl` (7 contoh). Tambahkan 200-500 contoh
sapaan + klarifikasi berita Indonesia sebelum fine-tune biar berasa milikmu.

Tambahkan: confusion matrix, demo poster, perbandingan sebelum/sesudah fine-tune.

## Struktur

- `model.py` — arsitektur 12M
- `tokenizer.py` — tokenizer kata Indonesia
- `dataset.py` — loader CSV
- `train.py` — training
- `infer.py` — prediksi
- `ocr_poster.py` — poster -> teks
- `factcheck.py` — verifikasi silang
- `app.py` + `templates/index.html` — web
