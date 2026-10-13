# 📚 Panduan Fine-Tune Lengkap

## Persiapan di Komputermu (Tanpa GPU)

### 1. Siapkan Data
```bash
bash prepare_finetune.sh
```

### 2. Copy File ke Komputer Temen
Copy seluruh folder `hoax-validator-12m/` ke flashdisk/Google Drive.

## Di Komputer Temen (Dengan GPU)

### Minimal Spesifikasi
- GPU: NVIDIA dengan VRAM 6GB+ (GTX 1060, RTX 2060, RTX 3060, dll)
- RAM: 16GB+
- Storage: 10GB+ kosong
- OS: Windows/Linux
- Python 3.10+

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl datasets accelerate bitsandbytes tokenizers scikit-learn flask pillow
```

### 2. Training Model Classifier 12M
Model ini yang mengklasifikasi teks jadi Valid/Hoaks/Meragukan.
```bash
python train.py --data data/contoh_dataset.csv --epochs 20 --batch 32 --vocab-size 5000 --max-len 128
```
- Hasil: `checkpoints/best.pt`, `checkpoints/vocab.json`
- Waktu: ~5-15 menit di GPU

### 3. Fine-Tune Chat AI (Qwen 0.5B + LoRA)
Ini yang bikin AI bisa ngobrol kayak ChatGPT tapi khusus validasi hoaks.
```bash
python train_chat_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --data data/train_chat.jsonl --out lora-hoax-chat --epochs 5 --lr 2e-4
```
- Hasil: folder `lora-hoax-chat/` (adapter LoRA ~50-100MB)
- Waktu: ~30-60 menit di GPU 8GB

### 4. (Opsional) Fine-Tune Model 1B untuk Hasil Lebih Bagus
```bash
# Login HuggingFace dulu (gratis):
huggingface-cli login

# Fine-tune Llama 3.2 1B:
python train_chat_lora.py --base meta-llama/Llama-3.2-1B-Instruct --data data/train_chat.jsonl --out lora-hoax-chat-1b --epochs 3 --rank 32
```

### 5. Copy Hasil Kembali
Copy folder ini ke komputermu:
- `checkpoints/` (model 12M)
- `lora-hoax-chat/` (adapter LoRA)

## Setelah Copy Balik ke Komputermu

### Test Chat
```bash
# Mode dummy (tanpa model chat, pakai template):
python chat.py --dummy

# Mode model (pakai Qwen + LoRA adapter):
python chat.py --adapter lora-hoax-chat

# Satu pesan:
python chat.py --adapter lora-hoax-chat --once "Bener gak sih vaksin bikin autis?"
```

### Jalankan Web
```bash
python app.py
# Buka http://localhost:5000
```

## Tips Agar AI Ngomong Lancar

1. **Perbanyak data chat** - Tambah contoh di `data/contoh_chat.jsonl` (target 200-500 contoh)
2. **Variasikan gaya bahasa** - Formal, informal, slang, Javanesse-mixed
3. **Multi-turn conversation** - Buat contoh percakapan 2-3 giliran
4. **Epoch yang cukup** - Untuk data kecil (<500), pakai 5-8 epoch
5. **Learning rate rendah** - Mulai 2e-4, kalau overfitting turunkan ke 1e-4

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| CUDA out of memory | Turunkan batch size: `--batch 1` dan naikkan `--grad-accum 16` |
| Model ngawur | Tambah data training, turunkan learning rate |
| Respons pendek/repetitif | Naikkan epoch, tambah contoh respons panjang |
| Import error tokenizers | `pip install tokenizers` |
