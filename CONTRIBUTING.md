# Berkontribusi ke PlosagataA-1.0

Terima kasih sudah mau berkontribusi. Panduan ini menjaga kualitas repo:
angka evaluasi jujur, bobot besar tidak masuk Git, dan setiap perubahan
terverifikasi sebelum di-merge.

## Cara cepat mulai

1. Fork repo, lalu clone fork tersebut.
2. Buat venv dan install dependensi (`requirements.txt`).
3. Buat branch dari `main` dengan nama jelas:
   `feat/...`, `fix/...`, atau `docs/...`.
4. Kerjakan dalam lingkup kecil — satu branch, satu tujuan.
5. Jalankan evaluasi yang relevan sebelum commit (lihat bawah).
6. Push branch ke fork, buka Pull Request ke `main` dengan deskripsi:
   apa yang diubah, kenapa, dan hasil evaluasinya.

## Aturan commit

- Pesan ringkas mengikuti gaya repo, contoh:
  `feat(train): ...`, `fix(validasi): ...`, `docs(readme): ...`.
- Satu commit = satu perubahan logis. Jangan campur kode + data besar.

## Aturan evaluasi (wajib)

- Perubahan klasifier 12M: lampirkan hasil `scripts/eval_verdict.py`
  (akurasi + probe regresi).
- Perubahan deteksi fiksi: lampirkan hasil `scripts/eval_fiksi.py`.
- Perubahan training: laporkan train loss **dan** val loss
  (`train_pure.py --val-split` menyimpannya otomatis). Kenaikan val loss
  = overfitting, PR tidak diterima sampai diatasi (tambah data,
  turunkan LR/epoch).
- Validasi klasifier selalu dipisah **sebelum** augmentasi.

## Aturan data dan bobot

- Jangan commit file `*.pt` / `*.bin` dan folder `checkpoints/`
  (sudah di-ignore — bobot dibagikan via Release/Drive).
- Jangan commit korpus hasil generate (`data/korpus.wiki.txt`,
  `data/korpus.full.txt`) — bangun ulang via `scripts/build_corpus.py`.
- Dataset CSV/JSONL boleh di-commit jika kecil dan relevan dengan PR.
- Setiap `--out X.pt` wajib menjaga pasangannya `X.pt.vocab.pt`
  (jangan dipisahkan).

## Aturan kode

- Ikuti struktur yang ada (`plosagata/` murni model,
  `hoaxcore/` klasifier + util, `scripts/` CLI, `web/` server).
- Tanpa emoji di README/dokumen. Bahasa dokumen: Indonesia.
- Jaga fungsi tetap kecil dan beri docstring singkat bila menambah
  fungsi publik baru.

## Melapor masalah

Buka Issue dengan: langkah reproduksi, perintah yang dijalankan,
output/error lengkap, dan info lingkungan (OS, Python, Torch, GPU).
