#!/bin/bash
# Persiapan fine-tune untuk komputer temen (GPU)
# Jalankan di komputer ini dulu untuk menyiapkan data

set -e
echo "=== Persiapan Fine-Tune Hoax Validator ==="
echo ""

# 1. Generate chat dataset dari CSV
echo "[1/3] Konversi CSV -> chat JSONL..."
python3 scripts/csv_to_chat.py --in data/contoh_dataset.csv --out data/chat_hoaks.jsonl

# 2. Gabungkan semua data chat
echo "[2/3] Gabungkan semua data chat..."
cat data/chat_hoaks.jsonl data/contoh_chat.jsonl > data/train_chat.jsonl
WC=$(wc -l < data/train_chat.jsonl)
echo "   Total data chat: $WC contoh"

# 3. Info
echo "[3/3] Selesai!"
echo ""
echo "=== File yang perlu dicopy ke komputer temen ==="
echo "  1. data/train_chat.jsonl     (data training chat)"
echo "  2. data/contoh_dataset.csv   (data training classifier 12M)" 
echo "  3. scripts/train_pure.py        (script training model murni)"
echo "  4. scripts/train.py             (script training model 12M)"
echo "  5. hoaxcore/model.py            (arsitektur model 12M)"
echo "  6. plosagata/                 (arsitektur model murni b1)"
echo "  7. tokenizer.py               (tokenizer BPE)"
echo "  8. dataset.py                 (data loader)"
echo "  9. requirements.txt           (dependencies)"
echo ""
echo "=== Perintah di komputer temen (GPU) ==="
echo "  # Install dependencies:"
echo "  pip install torch scikit-learn"
echo ""
echo "  # Training model murni Plosagata b1:"
echo "  python scripts/train_pure.py --size Plosagata_More --data data/train_chat.jsonl --epochs 3 --batch 1 --grad-accum 16"
echo ""
echo "  # Training model classifier 12M:"
echo "  python scripts/train.py --data data/contoh_dataset.csv --epochs 20 --batch 32"
echo ""
echo "  # Setelah selesai, copy balik:"
echo "  #   pure-Plosagata_More.pt        -> model murni terlatih"
echo "  #   checkpoints/      -> model 12M terlatih"
