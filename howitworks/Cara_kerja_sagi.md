# Cara Kerja Sagi — Penjelasan Semua Kode per Bab

> Ditulis untuk semua orang, bukan cuma programmer. Istilah sulit selalu
> dijelaskan dulu dengan perumpamaan sehari-hari.

---

## Bab 0 — Sagi itu apa sih? (30 detik)

Sagi = program yang menebak kata. Kamu ketik berita, dia menebak jawaban
kata-per-kata sampai jadi kalimat. Kepintarannya berasal dari latihan
(training): jutaan kali menebak → salah → dikoreksi. Makin banyak latihan
+ data, makin tepat tebakannya.

Dua otaknya:
- **Klasifier 12M** — juri cap: VALID / HOAKS / MERAGUKAN + skor 0-100.
- **Plosagata (Go/More/Next)** — penceramah: merangkai jawaban Bahasa Indonesia.

---

## Bab 1 — `hoaxcore/tokenizer.py`: Kamus Kata ↔ Angka

Model buta huruf — cuma paham angka. Tokenizer = penerjemah:
`Sagi cek giveaway!` → `[2, 55, 388, ...]` → dibalikkan lagi jadi teks.

Tekniknya **BPE**: kata asing dipotong per pecahan yang dikenal
(`giveaway` → `give`+`away`), jadi tidak ada kata "tidak dikenal".
Kamus dilatih dari datamu + bank 400 kata (`data/vocab_seed.txt`).

## Bab 2 — `hoaxcore/model.py`: Juri 12M (11.584.515 parameter)

Cara nilai satu kalimat, 4 langkah:
1. Tiap angka jadi vektor 384 angka desimal (koordinat makna).
2. 6 lapis Transformer: tiap kata "menoleh" ke kata lain (attention 6 kepala)
   agar paham konteks — `bank` di "bank sampah" beda arti dengan "bank BCA".
3. Rata-ratakan semua kata → 2 lapis hitungan akhir.
4. Keluar 3 skor → terbesar menang (mis. Hoaks 99%).

## Bab 3 — `hoaxcore/factcheck.py`: Detektif Pola Hoaks

Tanpa AI pun hoaks ada polanya: kata umpan (`giveaway`, `klik link ini`),
HURUF KAPITAL, tanda seru!!!!, link pendek (bit.ly), nomor HP, emoji bejibun.
Tiap pola = poin. Total = skor 0-100 + kesimpulan + saran + sumber
(turnbackhoax.id, cekfakta.com).

## Bab 4 — `hoaxcore/retrieval.py`: Tukang Cari Sumber Internet

Diberi klaim → ambil kata pentingnya → cari artikel Wikipedia Indonesia
(gratis, tanpa kunci) + buatkan link pencarian TurnBackHoax/CekFakta.
Hasilnya ditempel di bawah jawaban sebagai link klik. Offline? Otomatis
dilewati, jawaban tetap keluar.

## Bab 5 — `hoaxcore/ocr_poster.py`: Mata Pembaca Poster

Foto poster → tulisan (pakai Tesseract). Tulisannya lalu dicek seperti teks
biasa. Cocok untuk lomba: hoaks Indonesia banyak beredar sebagai gambar.

## Bab 6 — `hoaxcore/dataset.py`: Pelayan Data

Tugas sepele tapi penting: baca CSV (`text,label`), ubah jadi paket-paket
kecil (batch) yang disuapkan ke model saat training. Tanpa ini model tidak
bisa "makan".

## Bab 7 — `plosagata/config.py`: Cetak Biru 3 Model

Satu tabel berisi 3 ukuran: **Go** 92M/12 lapis (cepat),
**More** 912M/22 lapis (utama), **Next** 1,55M/38 lapis (raksasa).
Ganti angka di sini = ganti besar otak. Semua perintah (`--size ...`)
membacanya dari sini.

## Bab 8 — `plosagata/layers.py`: Bata Bangunan Otak

Dua bata yang disusun berlapis-lapis:
- **Attention**: tiap kata bertanya ke semua kata lain "siapa yang penting
  buatku?" — inilah yang membuat model paham kalimat, bukan sekadar kata.
- **Feedforward**: "merenung" dua lapis hitungan per kata.
Masing-masing dibungkus penyeimbang (LayerNorm) + jalan pintas (residual)
agar sinyal tidak hilang di model dalam.

## Bab 9 — `plosagata/model.py`: Otak Utuh yang Menebak

Rangkaiannya: embedding → 22 bata → penyeimbang akhir → skor 16.000 kata.
Kata skor tertinggi (atau diundi di antara top) ditempel ke kalimat, ulangi
sampai cukup panjang. Itulah "berpikir"-nya AI: menebak kata berikut,
berulang-ulang, sangat cepat.

Fitur hemat: `tie_weights` (kepala pinjam bobot embedding, hemat 32M),
`use_checkpoint` (hemat VRAM), penalti pengulangan (tidak ngelantur).

## Bab 10 — `plosagata/guard.py`: Satpam Topik Berita

Sebelum model mikir, satpam cek dulu pesannya:
- Sapaan (`halo`, `kawan`, `Sagi!`) → dibalas perkenalan (3 varian).
- Di luar berita (PR, coding, resep) → ditolak sopan + diajak balik.
- Berita → lolos diproses penuh.
Kata panggilan `Sagi` bisa disisipkan di mana saja. Ini perancah darurat —
pensiun total begitu model cukup pintar (mode `--pure-think`).

## Bab 11 — `plosagata/slang.py`: Penerjemah Bahasa Gaul

Kamus 199 slang → formal (`gaskeun ngab` → `lanjutkan bang`). Diterapkan
saat training DAN saat chat agar konsisten. Nambah kata = 1 baris, tanpa
latih ulang tokenizer.

## Bab 12 — `scripts/`: Tombol-Tombol Perintah

| Perintah | Dipencet untuk |
|---|---|
| `train.py` | Latih juri 12M dari CSV |
| `train_pure.py` | Latih Sagi (Go/More/Next), mode hemat GPU |
| `infer.py` | Tes nilai 1 kalimat |
| `pure_chat.py` | Ngobrol di terminal (+ `--pure-think`) |
| `gpt_pure.py` | Lihat jumlah parameter (bukti hidup) |
| `csv_to_chat.py` | CSV → contoh chat otomatis |
| `build_corpus.py` | Download Wikipedia → bahan latih raksasa |
| `auto_collect.py` | Cari berita internet otomatis → masuk database |
| `ctx7.py` | Tanya dokumentasi coding terbaru |
| `sagi_remote.py` | Ngobrol ke Sagi di komputer jauh |
| `sagi` (terminal) | Pintasan ngobrol dari mana saja |

Semua otomatis kembali ke akar repo — bisa dijalankan dari folder mana pun.

## Bab 13 — `web/`: Muka untuk Manusia

- `app.py` — pelayan: 4 rute (`/` validator, `/chat` ngobrol, API check/chat/
  history/stats), token pengaman, auto-nyala 12M saat start.
- `chatbot.py` (`SagiEngine`) — peracik jawaban: guard → model → verdict 12M
  → sumber. Malas-load (model berat dibuka hanya saat dibutuhkan).
- `templates/` — `index.html` (validator + riwayat) dan `chat.html`
  (chatbot ala ChatGPT: sidebar, kartu saran, typing, salin).

## Bab 14 — `data/`: Bahan Bakar

- `contoh_dataset.csv` (375, seimbang) → makanan juri 12M.
- `contoh_chat.jsonl` (147) → makanan gaya bicara Sagi.
- `train_chat.jsonl` (522, gabungan) → makanan training chat.
- `vocab_seed.txt` → vitamin tokenizer.
Rumusnya: contoh beragam > contoh banyak > epoch banyak.

---

## Urutan baca kalau baru pertama kali

1. Bab 7 (pendek, paham ukurannya) → 2. Bab 8 (batanya) → 3. Bab 9 (otaknya)
4. Jalankan `scripts/gpt_pure.py Plosagata_Go` → 5. Bab 2 + Bab 12 (juri + tombol)

Versi teknis per berkas (untuk programmer): `docs/ARSITEKTUR.md`.
Cara pindah kepintaran antar komputer: `cara_pindahinIQsagi.md` (folder ini).
