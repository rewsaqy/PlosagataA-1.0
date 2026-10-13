# Cara Nambah File (Data / Kode Baru) Tanpa Merusak Proyek

> Aturan emas: **file baru wajib didaftarkan** di dokumen + `git add`.
> File yang tidak terdokumentasi = file yang rusak diam-diam.

## 1. Nambah data (aman, kapan saja)

| Mau nambah | Taruh di | Lalu jalankan |
|---|---|---|
| Contoh klasifier | `data/contoh_dataset.csv` (`text,label`) | `python3 scripts/csv_to_chat.py` + gabung ulang `train_chat.jsonl` |
| Contoh chat | `data/contoh_chat.jsonl` (1 baris = 1 JSON) | gabung ulang `train_chat.jsonl` (lihat perintah di atas) |
| Kata tokenizer | `data/vocab_seed.txt` (1 baris = 1 frasa) | otomatis dipakai build berikutnya |
| Slang baru | `plosagata/slang.py` dict `SLANG` | langsung aktif (tanpa retrain) |
| Kata topik berita | `plosagata/guard.py` `NEWS_KEYWORDS` | langsung aktif |

Perintah gabung ulang (hafalkan):
```bash
python3 scripts/csv_to_chat.py
cat data/chat_hoaks.jsonl data/contoh_chat.jsonl > data/train_chat.jsonl
```

## 2. Nambah kode (wajib 4 langkah)

1. **Tempatnya benar:** model chat → `plosagata/`, klasifier → `hoaxcore/`,
   perintah → `scripts/`, web → `web/`, dokumen → `docs/` atau `howitworks/`.
2. **Import benar:** `from plosagata import ...` / `from hoaxcore import ...`.
   Script CLI wajib blok ini di atas (agar jalan dari mana saja):
   ```python
   ROOT = Path(__file__).resolve().parent.parent
   sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
   ```
3. **Tes sebelum commit:**
   ```bash
   .venv/bin/python -m compileall -q <file>.py
   ```
4. **Daftarkan:** tulis 1 baris di `howitworks/Cara_kerja_sagi.md` (bab yang cocok)
   + sebut di `README.md` kalau fitur besar.

## 3. Nambah model baru (ukuran ke-4, ke-5...)

1. Tambah 1 baris di `plosagata/config.py` dict `CONFIGS`.
2. Cek parameter: `.venv/bin/python scripts/gpt_pure.py NAMA_BARU`.
3. Update tabel model di `README.md` + dropdown `web/templates/chat.html`.
4. Semua perintah (`--size ...`) otomatis kenal nama baru — tidak ada yang perlu diubah lagi.

## 4. Peta dokumen (mulai dari mana?)

```
baru datang  → README.md (gambaran + perintah)
mau paham    → howitworks/Cara_kerja_sagi.md (14 bab, bahasa santai)
mau ngoding  → docs/ARSITEKTUR.md (teknis per berkas)
mau training → docs/PANDUAN_FINETUNE.md + README bagian 5
mau pindah PC → howitworks/cara_pindahinIQsagi.md
nambah file  → file ini
```

Semua dokumen wajib punya link ke peta ini. Kalau kamu bikin dokumen baru,
tambahkan juga linknya di sini + di `README.md`.
