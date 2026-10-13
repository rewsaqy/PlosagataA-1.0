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
pip install torch scikit-learn
```

### 2. Training Model Classifier 12M
Model ini yang mengklasifikasi teks jadi Valid/Hoaks/Meragukan.
```bash
python scripts/train.py --data data/contoh_dataset.csv --epochs 20 --batch 32 --vocab-size 5000 --max-len 128
```
- Hasil: `checkpoints/best.pt`, `checkpoints/vocab.json`
- Waktu: ~5-15 menit di GPU

### 3. Training Model Murni Plosagata b1 (~912M)
Ini model buatan sendiri (arsitektur di `plosagata/`). Butuh GPU 24GB+;
di RTX 2060 6GB pakai `--batch 1 --grad-accum 16` dan bersabar (lambat).
```bash
python scripts/train_pure.py --size b1 --data data/train_chat.jsonl --epochs 3 --batch 1 --grad-accum 16
```
- Hasil: `pure-b1.pt` (~3.6GB)
- Waktu: berjam-jam di GPU kecil — wajar untuk 900M parameter dari nol

### 4. Copy Hasil Kembali
Copy file ini ke komputermu:
- `checkpoints/` (model 12M)
- `pure-b1.pt` (model murni terlatih)

## Setelah Copy Balik ke Komputermu

### Test Sagi
```bash
# Perintah terminal (dari mana saja):
sagi
sagi "Sagi"

# Chat model murni:
.venv/bin/python scripts/pure_chat.py --size tiny --prompt "Sagi"
# Setelah training b1 di GPU dan copy pure-b1.pt balik:
.venv/bin/python scripts/pure_chat.py --size b1 --ckpt pure-b1.pt
```

### Jalankan Web
```bash
python web/app.py
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
