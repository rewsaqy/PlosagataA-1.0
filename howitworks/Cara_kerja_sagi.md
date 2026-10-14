# Cara Kerja Sagi (Versi Santai)

> Detail lengkap dan terkini ada di dokumentasi resmi:
> **`docs/ARSITEKTUR.md`** (bedah per modul + aliran data).

## Versi 1 menit

1. Kamu ketik berita → Sagi ubah jadi angka (tokenizer).
2. **Juri 12M** menilai: VALID / HOAKS / MERAGUKAN + skor.
3. **Detektif** cek pola hoaks + cari sumber internet.
4. **Sagi chat** merangkai jawaban (model Plosagata Go/More/Next).
5. **Satpam** memastikan topiknya tetap berita.

Mau yang detail (nama fungsi, bentuk tensor, urutan panggil)?
→ `docs/ARSITEKTUR.md` Bab 1–2.

Mau melatih, memindah, atau nambah file?
→ `docs/PANDUAN_FINETUNE.md` · `cara_pindahinIQsagi.md` (folder ini) ·
`Cara_nambah_file.md` (folder ini).
