#!/bin/bash
# Persiapan fine-tune untuk komputer temen (GPU)
# Jalankan di komputer ini dulu untuk menyiapkan data

set -e
echo "=== Persiapan Fine-Tune Hoax Validator ==="
echo ""

# 1. Generate chat dataset dari CSV
echo "[1/3] Konversi CSV -> chat JSONL..."
python3 csv_to_chat.py --in data/contoh_dataset.csv --out data/chat_hoaks.jsonl

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
echo "  3. train_chat_lora.py         (script fine-tune LoRA)"
echo "  4. train.py                   (script training model 12M)"
echo "  5. model.py                   (arsitektur model 12M)"
echo "  6. tokenizer.py               (tokenizer BPE)"
echo "  7. dataset.py                 (data loader)"
echo "  8. requirements.txt           (dependencies)"
echo ""
echo "=== Perintah di komputer temen (GPU) ==="
echo "  # Install dependencies:"
echo "  pip install torch transformers peft trl datasets accelerate bitsandbytes tokenizers scikit-learn"
echo ""
echo "  # Fine-tune chat (Qwen 0.5B + LoRA):"
echo "  python train_chat_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --data data/train_chat.jsonl --out lora-hoax-chat --epochs 5"
echo ""
echo "  # Training model classifier 12M:"
echo "  python train.py --data data/contoh_dataset.csv --epochs 20 --batch 32"
echo ""
echo "  # Setelah selesai, copy balik:"
echo "  #   lora-hoax-chat/     -> folder adapter LoRA"
echo "  #   checkpoints/        -> model 12M terlatih"
