# Panduan Pelatihan Mandiri AI Plosagata 12M

Dokumen ini adalah panduan lengkap dan praktis bagi Anda untuk melatih (training) model kecerdasan buatan **Plosagata 12M** secara mandiri di komputer sendiri (baik menggunakan CPU laptop maupun GPU lokal).

---

## 1. Mengenal Arsitektur Plosagata 12M

**Plosagata 12M** adalah varian model *Causal Language Model (Decoder-Only Transformer)* karya anak bangsa yang dibangun mandiri dari nol tanpa ketergantungan API pihak ketiga (seperti OpenAI atau Anthropic).

### Spesifikasi Arsitektur:
- **Total Parameter**: `12,665,856` (12.66 Juta Parameter)
- **Ukuran Dimensi ($d_{model}$)**: `384`
- **Jumlah Lapisan (Layers)**: `6`
- **Jumlah Attention Heads**: `6` (masing-masing berdimensi 64)
- **Dimensi Feed-Forward ($d_{ff}$)**: `1536` ($4 \times d_{model}$)
- **Panjang Konteks Maksimum**: `256 token`
- **Ukuran Kosakata (Vocab)**: `5,000 subword` (dioptimasi khusus Bahasa Indonesia & slang medsos)
- **Komponen Modern**:
  - RMSNorm (normalisasi cepat & stabil)
  - RoPE (Rotary Position Embeddings)
  - SwiGLU Activation Function
  - Tied Embedding Weights (efisiensi memori)

> **Kenapa 12M sangat cocok untuk dilatih mandiri?**  
> Model berukuran 12M parameter memiliki titik temu sempurna antara kecerdasan bahasa dan efisiensi komputasi. Anda tidak memerlukan GPU server mahal; pelatihan 5-10 epoch dapat selesai dalam hitungan menit di CPU laptop biasa, dan di bawah 1 menit jika menggunakan kartu grafis Nvidia (CUDA).

---

## 2. Struktur Dataset Pelatihan

Dataset instruksi dan obrolan disimpan di folder `data/`:
1. `data/contoh_chat.jsonl` : Percakapan multi-turn, identitas Sagi, kapabilitas, literasi digital, dan klarifikasi fakta.
2. `data/chat_hoaks.jsonl` : Konversi ribuan kasus hoaks aktual di Indonesia.
3. `data/train_chat.jsonl` : Gabungan dataset lengkap siap latih (10.866+ sampel obrolan).

Format setiap baris adalah JSON Lines:
```json
{"messages": [
  {"role": "system", "content": "Kamu Sagi, asisten AI validator hoaks dan informasi digital Indonesia yang ramah, cerdas, dan komunikatif."},
  {"role": "user", "content": "Kamu bisa bantu apa aja?"},
  {"role": "assistant", "content": "Halo! Aku Sagi, asisten AI verifikasi fakta berbasis arsitektur Plosagata 12M..."}
]}
```

---

## 3. Langkah Menjalankan Pelatihan (Training)

Kami telah merancang skrip pelatihan khusus `scripts/train_12m.py` yang otomatis menangani:
- Tokenisasi subword Bahasa Indonesia
- Pembagian data latih (90%) dan data uji validasi (10%)
- Masking target padding (`-100`) agar model tidak menghafal padding
- Learning Rate Scheduler (*Cosine Decay* dengan *Linear Warmup*)
- Gradient Clipping untuk menjaga stabilitas pelatihan
- Penyimpanan otomatis checkpoint terbaik (*Best Validation Loss*)
- Smoke test interaktif langsung di terminal begitu pelatihan usai.

### A. Perintah Standar (Rekomendasi di CPU)
Cukup jalankan perintah berikut di terminal proyek:
```bash
python scripts/train_12m.py
```
*Skrip akan otomatis melatih model selama 5 epoch dengan batch size 8.*

### B. Menyesuaikan Parameter Latihan
Anda dapat mengatur jumlah epoch, batch size, atau learning rate sesuai keinginan:
```bash
# Melatih 10 epoch dengan batch 16:
python scripts/train_12m.py --epochs 10 --batch 16 --lr 3e-4

# Melanjutkan pelatihan dari checkpoint sebelumnya:
python scripts/train_12m.py --epochs 5 --resume checkpoints/pure-Plosagata_12M-best.pt
```

### C. Melatih di Komputer Ber-GPU (Nvidia CUDA)
Jika komputer Anda memiliki kartu grafis Nvidia, skrip otomatis mendeteksi CUDA dan mengaktifkan akselerasi FP16:
```bash
python scripts/train_12m.py --epochs 10 --batch 32 --device cuda --fp16
```

---

## 4. Hasil Checkpoint & Tokenizer

Setelah pelatihan selesai, skrip otomatis menghasilkan file berikut di folder `checkpoints/` (tanpa salinan otomatis ke root — sejak perbaikan B-24):

| File Checkpoint | Fungsi |
| :--- | :--- |
| `checkpoints/pure-Plosagata_12M-best.pt` | Bobot neural network model terbaik (berdasarkan validation loss terendah) |
| `checkpoints/pure-Plosagata_12M.pt` | Bobot checkpoint epoch terakhir |
| `checkpoints/pure-Plosagata_12M.vocab.pt` | Kamus token subword pasangan model |
| `checkpoints/pure-Plosagata_12M.tokenizer.json` | Konfigurasi tokenizer BPE penuh |

---

## 5. Menguji & Menggunakan Model Hasil Latihan

### A. Uji Coba Chat di Terminal (CLI)
Gunakan skrip `scripts/pure_chat.py`:

```bash
# Chat dengan sistem terintegrasi (guard + model 12M):
python scripts/pure_chat.py --size Plosagata_12M --ckpt checkpoints/pure-Plosagata_12M-best.pt --prompt "Halo Sagi!"

# Tanya kapabilitas Sagi:
python scripts/pure_chat.py --size Plosagata_12M --ckpt checkpoints/pure-Plosagata_12M-best.pt --prompt "Kamu bisa bantu apa aja?"

# Uji generasi murni neural network (Pure Think):
# Prompt otomatis dibungkus format training (sistem:/konsumen:/assistant:) —
# cukup tulis pesan mentahnya, jangan ketik "user: ..." sendiri.
python scripts/pure_chat.py --size Plosagata_12M --ckpt checkpoints/pure-Plosagata_12M-best.pt --pure-think --prompt "Bener gak bansos cair hari ini?"
```

### B. Menjalankan di Antarmuka Web Chatbot
Untuk menggunakan model 12M hasil latihan Anda di Web Chatbot (port 5000):
```bash
SAGI_SIZE=Plosagata_12M SAGI_CKPT=checkpoints/pure-Plosagata_12M-best.pt python web/app.py
```
Buka browser di `http://localhost:5000`. Sagi akan langsung ditenagai oleh model Plosagata 12M hasil latihan Anda sendiri!

---

## 6. Tips Mengembangkan Dataset Sendiri

Jika Anda ingin menambahkan variasi pertanyaan atau kasus hoaks baru ke model:
1. Buka file `data/contoh_chat.jsonl`.
2. Tambahkan baris baru dengan format JSON yang sama.
3. Jalankan kembali `python scripts/train_12m.py`.
Model akan mempelajari pola kalimat baru tersebut dan semakin fasih merespons!
