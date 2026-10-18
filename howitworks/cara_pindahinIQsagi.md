# Cara Pindahin IQ Sagi + Cara Kerja Sistemnya

> Peta: `README.md` · cara kerja: `Cara_kerja_sagi.md` (folder ini) ·
> nambah file: `Cara_nambah_file.md` (folder ini)
>
> IQ Sagi = **file**, bukan install-an. Pindahin file = pindahin kepintaran.

## 1. File apa saja yang membawa IQ?

| File | Isi | Ukuran | Wajib? |
|---|---|---|---|
| `checkpoints/pure-Plosagata_Browser-best.pt` (+ `.pt.vocab.pt`, `.tokenizer.json`) | Otak chat 25M terbaik (retrain 18 Okt) | ±100 MB | Ya (chat utama) |
| `checkpoints/pure-Plosagata_12M-best.pt` (+ pasangan vocab/tokenizer) | Otak chat 12M terbaik | ±50 MB | Ya (alternatif super ringan) |
| `checkpoints/best.pt` + `checkpoints/vocab.json` | Klasifier 12M (juri validator web) | ±46 MB | Ya (validator web pakai ini) |
| `pure-Go-*` / `pure-Plosagata_More.pt` | Bobot lama Go/More — tidak ada di disk sejak bersih-bersih 18 Okt | ±350 MB–3,6 GB | Kalau punya salinan Drive/Release |

Catatan nama: `--out X.pt` otomatis bikin `X.pt.vocab.pt` di sebelahnya
(bukan `X.vocab.pt`), dan tiap best baru otomatis dicopykan
`X-best.pt.vocab.pt`.
**Jangan pernah** memisahkan `.pt` dari `.vocab.pt` pasangannya.
Contoh benar: `pure-Go-sagi-best.pt` ↔ `pure-Go-sagi-best.pt.vocab.pt`.

## 2. Cara pindah (pilih satu)

### A. Flashdisk (paling gampang)
1. Colok, pastikan format **exFAT/NTFS** (klik kanan → Properties).
   FAT32 **menolak** file > 4 GB — More 3,6 GB mepet bahaya.
2. Copy ckpt + pasangannya: isi folder `checkpoints/` (bobot `.pt`,
   `.pt.vocab.pt`, `.tokenizer.json` untuk model yang mau dibawa).
3. Di komputer tujuan: copy ke **dalam folder proyek** (sejajar `scripts/`, `data/`).
4. Tes: `.venv/bin/python scripts/pure_chat.py --size Plosagata_Browser --ckpt checkpoints/pure-Plosagata_Browser-best.pt --prompt "halo"`
   (vocab pasangan terdeteksi otomatis; kalau mau eksplisit tambah
   `--vocab-file checkpoints/pure-Plosagata_Browser-best.pt.vocab.pt`).

### B. Google Drive
Upload → download di tujuan → taruh sejajar folder proyek → tes sama seperti di atas.

### C. Kabel LAN / WiFi sama (tanpa flashdisk)
Di komputer sumber (folder proyek):
```bash
python3 -m http.server 8000
```
Di komputer tujuan (browser): `http://IP-SUMBER:8000` → download file → taruh sejajar.

### D. GitHub Release (arsip permanen, bukan push biasa)
`git push` **menolak** file > 100 MB. Caranya: repo → Releases → Draft new release
→ upload `.pt` → Publish. Tempel link-nya di README.

## 3. Verifikasi IQ pindah dengan benar

```bash
# 1. Ukuran wajar? (Browser ±100MB, 12M ±50MB, vocab KB-an)
ls -lh checkpoints/pure-*.pt*
# 2. Sapa (harus identitas Sagi, instan):
.venv/bin/python scripts/pure_chat.py --size Plosagata_Browser --ckpt checkpoints/pure-Plosagata_Browser-best.pt --prompt "halo"
# 3. Berita (masih word-salad Indonesia tahap ini, bukan byte sampah):
.venv/bin/python scripts/pure_chat.py --size Plosagata_Browser --ckpt checkpoints/pure-Plosagata_Browser-best.pt --prompt "Bener gak giveaway iPhone klik link ini?"
```
Gagal langkah 2-3 = salah pasangan vocab atau file korup saat copy → ulangi transfer.
Tanda khas salah vocab: muncul `[PERINGATAN: vocab pasangan tidak ketemu...]`
atau output byte sampah (`Ê Ī þ`). Obat: pastikan `X-best.pt.vocab.pt`
ada di sebelah `X-best.pt`, atau ulang training yang menyimpan best+vocab
otomatis (`train_pure.py --val-split 0.1`).

> Alur training saat ini: pretrain Wiki (`--corpus data/korpus.full.txt`)
> → tune (`--ckpt ...-best.pt --vocab-file ...pt.vocab.pt --val-split 0.1`).
> Tune WAJIB pakai `--vocab-file` dari pretrain agar embedding cocok.

## 4. Cara kerja sistem Sagi (lengkap, berurutan)

```
1. INPUT   user ketik (terminal `sagi` / web /chat) atau upload poster
2. OCR     poster → teks (hoaxcore/ocr_poster.py, Tesseract)
3. SLANG   gaul → formal (plosagata/slang.py, 2.173 entri)
4. GUARD   topik? (plosagata/guard.py)
   ├─ sapaan ──────────────→ template identitas Sagi
   ├─ off-topic ───────────→ tolak sopan + ajak balik ke berita
   └─ berita ──────────────→ lanjut ↓
5. TOKENIZER  teks → angka (hoaxcore/tokenizer.py, BPE + bank 106 ribu kata)
6. MODEL MURNI (plosagata)  angka → tebak kata berikut, diulang
   (Go 12 layer / More 22 layer / Next 38 layer)
7. VALIDATOR 12M (hoaxcore) label Valid/Hoaks/Meragukan + probabilitas
8. FACT-CHECK (hoaxcore)    pola hoaks → skor 0-100 + saran
9. RETRIEVAL (hoaxcore)     kata kunci → Wikipedia ID + link TurnBackHoax/CekFakta
10. OUTPUT  jawaban + badge verdict + sumber klik (web) / teks (terminal)
```

**Dua otak, dua tugas:** 12M menilai (pasti, cepat), Plosagata mengobrol
(fasih hasil training). Guard + template = perancah darurat; pensiun total
begitu model cukup terlatih (mode `--pure-think` / `SAGI_PURE_THINK=1`).

## 5. Kalau pindah mesin, apa yang TIDAK perlu dibawa?

- `.venv/` (install ulang via `requirements.txt`, 5 menit;
  ingat Torch harus versi CUDA `+cu126` agar GPU kepakai, bukan `+cpu`)
- `data/korpus.wiki.txt`, `data/korpus.full.txt` (bangun ulang via
  `scripts/build_corpus.py` — sumber `wikimedia/wikipedia`, bukan `wikipedia` lama)
- `__pycache__/`, `web.log`, file `.tmp`
- Kode jelas ikut via `git clone` (bukan copy manual)

## 6. Masalah umum

| Gejala | Sebab | Obat |
|---|---|---|
| Output byte sampah (`Ê Ī þ`) | Salah pasangan `.vocab.pt` | Samakan ckpt + vocab dari training yang sama (`X-best.pt` ↔ `X-best.pt.vocab.pt`) |
| `[PERINGATAN: vocab pasangan tidak ketemu]` | File `.pt.vocab.pt` tidak di sebelah ckpt | Copy vocab-nya juga, atau pakai `--vocab-file` eksplisit |
| `size mismatch` saat load | Vocab beda ukuran | Jangan campur ckpt Go dengan More |
| Copy gagal di tengah | Flashdisk FAT32 / kabel lepas | exFAT + ulangi, cek `ls -lh` |
| `git push` ditolak | File > 100 MB | Pakai Release/Drive, jangan push `.pt` |
