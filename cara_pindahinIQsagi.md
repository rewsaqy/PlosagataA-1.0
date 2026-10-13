# Cara Pindahin IQ Sagi + Cara Kerja Sistemnya

> IQ Sagi = **file**, bukan install-an. Pindahin file = pindahin kepintaran.

## 1. File apa saja yang membawa IQ?

| File | Isi | Ukuran | Wajib? |
|---|---|---|---|
| `pure-Plosagata_Go.pt` | Bobot Go 92M = otaknya | ±350 MB | Ya |
| `pure-Plosagata_Go.vocab.pt` | Kamus saat training | KB-an | Ya (tanpa ini output sampah) |
| `pure-Plosagata_More.pt` | Bobot More 912M | ±3,6 GB | Kalau training More |
| `pure-Plosagata_More.vocab.pt` | Kamus More | KB-an | Kalau training More |
| `checkpoints/` | Model 12M + vocab | ±90 MB | Kalau 12M ikut dilatih ulang |

Catatan nama: `--out X.pt` otomatis bikin `X.vocab.pt` di sebelahnya.
**Jangan pernah** memisahkan `.pt` dari `.vocab.pt` pasangannya.

## 2. Cara pindah (pilih satu)

### A. Flashdisk (paling gampang)
1. Colok, pastikan format **exFAT/NTFS** (klik kanan → Properties).
   FAT32 **menolak** file > 4 GB — More 3,6 GB mepet bahaya.
2. Copy 3 file di atas ke flashdisk.
3. Di komputer tujuan: copy ke **dalam folder proyek** (sejajar `scripts/`, `data/`).
4. Tes: `.venv/bin/python scripts/pure_chat.py --size Plosagata_More --ckpt pure-Plosagata_More.pt --prompt "Sagi"`

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
# 1. Ukuran wajar? (Go ±350MB, More ±3,6GB, vocab KB-an)
ls -lh pure-*.pt
# 2. Sapa (harus identitas Sagi, instan):
.venv/bin/python scripts/pure_chat.py --size Plosagata_More --ckpt pure-Plosagata_More.pt --prompt "Sagi"
# 3. Berita (harus Bahasa Indonesia nyambung, bukan byte sampah):
.venv/bin/python scripts/pure_chat.py --size Plosagata_More --ckpt pure-Plosagata_More.pt --prompt "Bener gak giveaway iPhone klik link ini?"
```
Gagal langkah 2-3 = salah pasangan vocab atau file korup saat copy → ulangi transfer.

## 4. Cara kerja sistem Sagi (lengkap, berurutan)

```
1. INPUT   user ketik (terminal `sagi` / web /chat) atau upload poster
2. OCR     poster → teks (hoaxcore/ocr_poster.py, Tesseract)
3. SLANG   gaul → formal (plosagata/slang.py, 199 entri)
4. GUARD   topik? (plosagata/guard.py)
   ├─ sapaan ──────────────→ template identitas Sagi
   ├─ off-topic ───────────→ tolak sopan + ajak balik ke berita
   └─ berita ──────────────→ lanjut ↓
5. TOKENIZER  teks → angka (hoaxcore/tokenizer.py, BPE + bank 400 kata)
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

- `.venv/` (install ulang via `requirements.txt`, 5 menit)
- `data/korpus.txt` (bangun ulang via `scripts/build_corpus.py`)
- `__pycache__/`, `web.log`, file `.tmp`
- Kode jelas ikut via `git clone` (bukan copy manual)

## 6. Masalah umum

| Gejala | Sebab | Obat |
|---|---|---|
| Output byte sampah (`Ê Ī þ`) | Salah pasangan `.vocab.pt` | Samakan ckpt + vocab dari training yang sama |
| `size mismatch` saat load | Vocab beda ukuran | Jangan campur ckpt Go dengan More |
| Copy gagal di tengah | Flashdisk FAT32 / kabel lepas | exFAT + ulangi, cek `ls -lh` |
| `git push` ditolak | File > 100 MB | Pakai Release/Drive, jangan push `.pt` |
