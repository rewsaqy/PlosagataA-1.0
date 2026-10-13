# Arsitektur Kode PlosagataA-1.0 (Penjelasan per Part)

Dokumen ini menjelaskan SEMUA file kode: gunanya apa dan cara kerjanya.
Untuk cara install/training, baca `README.md`.

## Peta besar (aliran data)

```
Teks / Poster
   │  poster → hoaxcore/ocr_poster.py (gambar jadi teks)
   ▼
hoaxcore/tokenizer.py ......... teks → angka (BPE)
   ├─► hoaxcore/model.py ...... angka → label Valid/Hoaks/Meragukan (12M)
   ├─► hoaxcore/factcheck.py .. label + teks → skor 0-100 + saran sumber
   └─► plosagata/model.py ..... angka → tebakan kata berikut (b1, Sagi chat)
         ▲ dilatih oleh scripts/train_pure.py
   ▼
web/app.py (validator) / web/chatbot.py + /chat (ngobrol)
```

## Part 1 — `hoaxcore/` (otak validator 12M)

| File | Guna (bahasa bayi) | Isi arsitektur |
|---|---|---|
| `model.py` | Mesin penilai: membaca angka, keluar label. Seperti juri yang cap VALID/HOAKS | `TinyHoaxTransformer`: embedding kata+posisi (5000×384) → 6 lapis TransformerEncoder (self-attention 6 head + feedforward 1280) → rata-rata → 2 lapis linear → 3 skor kelas. Total 11.584.515 parameter |
| `tokenizer.py` | Kamus kata↔angka (BPE). Tanpa ini model buta huruf | Latih pola potongan kata dari data (`build`), `encode` teks→id+mask, `decode` id→teks, `save/load` vocab.json. Kata asing dipotong per pecahan (tidak ada OOV) |
| `dataset.py` | Pelayan data: baca CSV, antar ke model per batch | `load_csv` (kolom text,label), `HoaxDataset` (encode + label jadi tensor), peta `LABEL2ID` |
| `factcheck.py` | Detektif pola hoaks + pemberi verdict akhir | `heuristic_signals`: cari kata umpan (giveaway, klik link), caps, tanda seru, short-URL, nomor HP, emoji. `local_verdict`: jumlahkan jadi skor 0-100 + kesimpulan + saran + sumber (turnbackhoax.id, cekfakta.com) |
| `ocr_poster.py` | Mata: baca tulisan di foto poster jadi teks | Bungkus Tesseract-OCR (Indonesia+Inggris). Gagal → pesan jelas, web tetap jalan |

## Part 2 — `plosagata/` (otak chat Sagi, murni buatan sendiri)

| File | Guna | Isi arsitektur |
|---|---|---|
| `config.py` | Cetak biru 3 ukuran model | `GPTConfig` dataclass (vocab, dim, layers, heads, ff, max_len, tie_weights). `tiny` 6M (tes CPU), `small` 92M (latihan), `b1` 912M (asli: vocab 16000, dim 2048, 22 layer, 16 head) |
| `layers.py` | Bata bangunan transformer | `CausalSelfAttention`: QKV → attention causal (tidak boleh intip kata depan) → proyeksi. `Block`: attention + feedforward GELU, masing-masing dibungkus LayerNorm + sisa (residual) |
| `model.py` | Model GPT utuh: tebak kata berikut | `PureGPT`: embedding kata+posisi → N Block → LayerNorm → head (dipinjam dari embedding = hemat 32M di b1) → distribusi 16.000 kata. `generate`: sampling suhu + top-k + penalti pengulangan + larangan token sampah. Init bobot ala GPT-2 (loss awal pas 8,5). Flag `use_checkpoint` untuk hemat VRAM |
| `guard.py` | Satpam topik: hanya berita yang lolos | Daftar kata berita/off-topic/sapaan + `is_news_topic` → greeting/news/offtopic. Kata panggilan `Sagi` (strip_wake). Sapaan & penolakan 3 varian (`sagi_hello`, `sagi_redirect`) |
| `slang.py` | Penerjemah gaul: 199 slang → formal | Dict + regex word-boundary. Dipakai training DAN chat agar konsisten. Nambah kata = 1 baris, tanpa retrain |

## Part 3 — `scripts/` (perintah-perintah)

| File | Perintah contoh | Input → Output |
|---|---|---|
| `train.py` | `python scripts/train.py --epochs 15` | CSV → `checkpoints/best.pt` + `vocab.json`. Split dulu baru augmentasi train (tidak bocor), AdamW + cosine, simpan terbaik |
| `train_pure.py` | `python scripts/train_pure.py --size b1 --corpus data/korpus.txt --fp16 --grad-checkpoint --optim adam8bit` | Korpus/JSONL → `pure-b1.pt`. Mode hemat VRAM untuk GPU kecil, log token/detik + ETA, lanjutkan via `--ckpt` |
| `infer.py` | `python scripts/infer.py --text "..."` | Teks → label + probabilitas 3 kelas |
| `pure_chat.py` | `python scripts/pure_chat.py --size tiny` | Chat terminal: guard → Sagi/generate. Model di-load malas (hemat RAM) |
| `gpt_pure.py` | `python scripts/gpt_pure.py b1` | Cetak jumlah parameter + tes forward (bukti arsitektur hidup) |
| `csv_to_chat.py` | `python3 scripts/csv_to_chat.py` | CSV → `data/chat_hoaks.jsonl` (template jawaban per label) |
| `build_corpus.py` | `python scripts/build_corpus.py` | Wikipedia Indonesia → `data/korpus.txt` (pretrain b1) |
| `prepare_finetune.sh` | `bash scripts/prepare_finetune.sh` | Siapkan + gabung data, cetak checklist copy ke GPU |

Semua script otomatis `chdir` ke root repo — bisa dijalankan dari folder mana saja.

## Part 4 — `web/` (muka untuk manusia)

| File | Guna |
|---|---|
| `app.py` | Server Flask: `/` validator (teks+poster), `/chat` halaman ngobrol, `/api/check` + `/api/chat` + `/api/history` + `/api/stats`. Riwayat 20 cek terakhir in-memory |
| `chatbot.py` | `SagiEngine`: guard → Sagi/generate model murni + verdict 12M (label+skor) ditempel di jawaban. Ukuran/ckpt via env `SAGI_SIZE`/`SAGI_CKPT` |
| `templates/index.html` | Form cek + badge hasil + riwayat/stats otomatis refresh |
| `templates/chat.html` | Chat bubble (user kanan, Sagi kiri) + label 12M di bawah jawaban berita |

## Part 5 — `data/` (bahan bakar)

- `contoh_dataset.csv` (215, seimbang 75/75/65) — latih klasifier
- `contoh_chat.jsonl` (132) — latih gaya Sagi (slang, penolakan, sapaan)
- `chat_hoaks.jsonl` (215, hasil generate) + `train_chat.jsonl` (347, gabungan)
- `vocab_seed.txt` (400+ kata) — dicampur otomatis tiap build vocab agar kata
  penting jadi 1 token utuh (tes: tsunami 7 potong → 1)

## Kalau mau ubah sesuatu, edit mana?

- Tambah slang → `plosagata/slang.py` (1 baris)
- Tambah kata kunci topik → `plosagata/guard.py` (`NEWS_KEYWORDS` / `OFFTOPIC_KEYWORDS`)
- Ganti ukuran model → `plosagata/config.py`
- Ganti arsitektur → `plosagata/layers.py` + `model.py`
- Tambah data → `data/` + regenerate (lihat README)
- Ubah tampilan web → `web/templates/`

## Urutan baca untuk pemula

1. `plosagata/config.py` (pendek, paham ukurannya)
2. `plosagata/layers.py` (batanya)
3. `plosagata/model.py` (rumahnya)
4. `scripts/gpt_pure.py` → jalankan, lihat angka parameter
5. `hoaxcore/model.py` (juri 12M) → `scripts/train.py` (cara ngajarin)
