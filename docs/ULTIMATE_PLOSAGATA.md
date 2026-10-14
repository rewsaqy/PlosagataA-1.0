# ULTIMATE PlosagataA-1.0 — Satu Kesatuan Dokumentasi

> Digenerate otomatis 2026-10-14 oleh `scripts/gabung_dok.py`. JANGAN edit file ini — edit file sumbernya, lalu generate ulang.

## Daftar Isi

1. [README — Gambaran + Perintah](#bagian-1)
2. [Riwayat 0-Sekarang](#bagian-2)
3. [Arsitektur Total](#bagian-3)
4. [Panduan Fine-Tune](#bagian-4)
5. [Pintu How-It-Works](#bagian-5)
6. [Pindah IQ](#bagian-6)
7. [Nambah File](#bagian-7)
8. [Catatan Bug](#bagian-8)
9. [Kontribusi](#bagian-9)


---

<a id="bagian-1"></a>

# Bagian 1: README — Gambaran + Perintah

(Sumber: `README.md`)

# PlosagataA-1.0

**Sagi: AI validator hoaks Bahasa Indonesia yang kami bangun dari nol.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Models](https://img.shields.io/badge/Models-12M_%7C_92M_%7C_912M_%7C_1.5B-green)](#model)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/rewsaqy/PlosagataA-1.0)](https://github.com/rewsaqy/PlosagataA-1.0/commits)
[![Stars](https://img.shields.io/github/stars/rewsaqy/PlosagataA-1.0?style=social)](https://github.com/rewsaqy/PlosagataA-1.0/stargazers)

Proyek lomba AI terapan: sistem validasi hoaks end-to-end untuk klasifikasi teks,
OCR poster, cek fakta heuristik + sumber internet, dan chatbot khusus berita.
Seluruh arsitektur dan bobot model ditulis dan dilatih sendiri, tanpa memakai
model unduhan dari pihak mana pun.

> **Peta dokumen:** `README.md` (dokumen ini) · `howitworks/Cara_kerja_sagi.md`
> (penjelasan santai) · `docs/ARSITEKTUR.md` (teknis) · `docs/PANDUAN_FINETUNE.md`
> (training) · `howitworks/cara_pindahinIQsagi.md` (pindah PC) ·
> `howitworks/Cara_nambah_file.md` (nambah file) · `BUGS.md` (catatan bug jujur)

---

## Daftar Isi

- [Arsitektur](#arsitektur)
- [Model](#model)
- [Mulai Cepat](#mulai-cepat)
- [Penggunaan](#penggunaan)
- [Pelatihan](#pelatihan)
- [Data & Bahasa](#data--bahasa)
- [Penyebaran & Akses Jarak Jauh](#penyebaran--akses-jarak-jauh)
- [Struktur Repositori](#struktur-repositori)
- [Peta Jalan](#peta-jalan)
- [Pembuat](#pembuat)
- [Lisensi](#lisensi)

---

## Arsitektur

```
                    ┌──────────────┐
  Teks ────────────►│  Tokenizer   │  kata → angka (BPE, 5K/8K/16K)
  Poster ──OCR─────►│  (hoaxcore)  │
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌───────────┐ ┌───────────┐
     │ Klasifier  │ │ Fact-     │ │ Plosagata │
     │ 12M        │ │ check +   │ │ (chat)    │
     │ Valid/     │ │ Retrieval │ │ tebak kata│
     │ Hoaks/     │ │ sumber    │ │ berikut   │
     │ Meragukan  │ │ internet  │ │           │
     └────────────┘ └───────────┘ └───────────┘
              └────────────┼────────────┘
                           ▼
              ┌────────────────────────┐
              │  Web / API / Terminal  │
              │  /  /chat  sagi  :5000 │
              └────────────────────────┘
```

Lapisan **guard** (`plosagata/guard.py`) memastikan Sagi hanya melayani topik
berita: sapaan dibalas identitas, off-topic ditolak sopan, klaim diproses penuh
(slang → tokenizer → model → verdict 12M + sumber).

Label final ditentukan satu pintu via `final_label` (heuristic menang saat
kuat: skor >= 60 selalu HOAKS), dan konten deklarasi fiksi/simulasi dideteksi
via `deteksi_fiksi` agar tidak dinilai sebagai berita faktual.

---

## Model

| Model | Parameter | Peran | Status |
|---|---|---|---|
| Klasifier 12M | **11.584.515** | Penilai Valid / Hoaks / Meragukan + skor 0–100 | Terlatih (val 94,9%, probe regresi 6/6) |
| Plosagata_Flash | **12.665.856** | Chat super ringan — CPU kentang, demo kilat | Arsitektur siap |
| Plosagata_Go | **91.593.216** | Chat ringan untuk CPU/laptop | Pra-latih + instruction-tune (eksperimental) |
| Plosagata_More | **912.008.192** | Chat utama Sagi, target training GPU | Arsitektur siap |
| Plosagata_Next | **1.549.927.424** | Varian besar untuk GPU 12 GB+ / klaster | Arsitektur saja |

Ganti model beneran di browser: buka `/models` — engine lama dibongkar total
(RAM kembali), engine baru nyala dengan sesi fresh. Bukan dropdown pura-pura.

Verifikasi mandiri:

```bash
.venv/bin/python scripts/gpt_pure.py Plosagata_Flash
.venv/bin/python scripts/gpt_pure.py Plosagata_Go
.venv/bin/python scripts/gpt_pure.py Plosagata_More
.venv/bin/python scripts/gpt_pure.py Plosagata_Next
.venv/bin/python scripts/eval_verdict.py   # akurasi + probe regresi klasifier
.venv/bin/python scripts/eval_fiksi.py     # uji deteksi fiksi/simulasi
.venv/bin/python scripts/eval_topik.py     # uji guard + memori sesi (7/7)
.venv/bin/python scripts/eval_rag.py --mini  # smoke RAG + latensi p95
```

Detail matematis tiap lapisan: `docs/ARSITEKTUR.md`.

---

## Mulai Cepat

**Prasyarat:** Python 3.12+, sekitar 4 GB ruang kosong. GPU NVIDIA opsional
(Torch harus build CUDA agar GPU terpakai. Build `+cpu` tidak menyentuh VRAM).
Tanpa GPU tetap berjalan di CPU.

```bash
git clone git@github.com:rewsaqy/PlosagataA-1.0.git
cd PlosagataA-1.0
uv venv --python 3.12
uv pip install -r requirements.txt
# Untuk GPU (contoh CUDA 12.6):
.venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python scripts/infer.py --text "Giveaway iPhone, klik link ini!"
.venv/bin/python web/app.py
# → http://localhost:5000  (validator)  +  /chat  (chatbot)
```

Windows: ganti `.venv/bin/python` dengan `.venv\Scripts\python.exe`.

Opsional (OCR poster): `sudo pacman -S tesseract tesseract-data-ind`
(Debian/Ubuntu: `tesseract-ocr tesseract-ocr-ind`).

---

## Penggunaan

### Terminal: perintah `sagi`

```bash
sagi                             # mode interaktif (Plosagata_More)
sagi "Bener gak giveaway ini?"   # satu pertanyaan, langsung keluar
```

Contoh perilaku: `Sagi` → identitas; `Buatkan kode python` → ditolak sopan;
klaim berita → diproses model + verdict 12M.

### Web

| Rute | Fungsi |
|---|---|
| `/` | Validator: tempel teks / unggah poster → label, skor, riwayat, statistik |
| `/chat` | Chatbot: sidebar, riwayat tersimpan, tombol cepat, indikator mengetik, badge verdict, tombol salin, dukungan token remote |
| `/models` | Daftar 4 model + ganti beneran (engine lama dibongkar, sesi fresh) |
| `/api/check` · `/api/chat` · `/api/history` · `/api/stats` · `/api/models` | API JSON |

Slash-command di dalam chat: `/models`, `/model <nama>`, `/status`, `/help`.

Stabilitas server: reloader nonaktif bawaan (konflik thread PyTorch),
thread CPU dibatasi. Aktifkan reload saat pengembangan UI:
`SAGI_RELOAD=1`.

### Utilitas

```bash
python3 scripts/ctx7.py "Flask" "streaming"   # cari dokumentasi via Context7
python scripts/sagi_remote.py "Sagi, halo!"   # client chat ke server jauh
```

---

## Pelatihan

### Klasifier 12M (CPU memadai)

Dataset `data/contoh_dataset.csv` (`text,label` → valid/hoaks/meragukan):

```bash
.venv/bin/python scripts/train.py --data data/contoh_dataset.csv --epochs 15 --batch 16
# → checkpoints/best.pt + checkpoints/vocab.json
```

Validasi dipisah **sebelum** augmentasi, jadi angka akurasinya jujur.

### Plosagata: pra-latih lalu instruction-tuning (GPU disarankan)

```bash
# 1. Korpus Wikipedia Indonesia (sumber: wikimedia/wikipedia):
.venv/bin/python scripts/build_corpus.py --out data/korpus.wiki.txt --limit 2000
cat data/korpus.txt data/korpus.wiki.txt > data/korpus.full.txt
# 2. Pra-latih dari nol (ada split validasi otomatis):
.venv/bin/python scripts/train_pure.py --size Plosagata_Go --corpus data/korpus.full.txt \
  --epochs 1 --batch 16 --fp16 --val-split 0.02 --out pure-Go-pretrain.pt
# 3. Instruction-tuning menjadi Sagi (WAJIB pakai vocab pretrain agar embedding cocok):
.venv/bin/python scripts/train_pure.py --size Plosagata_Go --data data/train_chat.jsonl \
  --epochs 5 --batch 8 --fp16 --lr 2e-4 --val-split 0.1 \
  --ckpt pure-Go-pretrain-best.pt --vocab-file pure-Go-pretrain.pt.vocab.pt \
  --out pure-Go-sagi.pt
```

Setiap `--out X.pt` otomatis menyimpan `X.pt.vocab.pt` di sebelahnya, dan
checkpoint val terbaik (`X-best.pt` + `X-best.pt.vocab.pt`). Jangan pernah
memisahkan `.pt` dari `.vocab.pt` pasangannya.

Bobot `.pt` tidak masuk Git (lihat `.gitignore`). Bagikan via
Release/Drive. Panduan pindah PC: `howitworks/cara_pindahinIQsagi.md`.

Mode berpikir-murni (semua pesan lewat model, perlu model terlatih):

```bash
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --ckpt pure-Go-sagi-best.pt --pure-think
SAGI_PURE_THINK=1 .venv/bin/python web/app.py
```

---

## Data & Bahasa

| Dataset | Isi | Untuk |
|---|---|---|
| `data/contoh_dataset.csv` | 514 baris (tumbuh via `auto_collect.py`) | Klasifier |
| `data/contoh_chat.jsonl` | 187 percakapan (slang, penolakan, sapaan) | Gaya Sagi |
| `data/train_chat.jsonl` | 701 (gabungan + generate) | Instruction-tuning |
| `data/vocab_seed.txt` | Bank kata inti | Build vocab otomatis |

- `plosagata/slang.py` berisi 199 entri slang → formal yang dipakai saat
  training **dan** inferensi agar konsisten.
- Tokenizer BPE mengeja kata asing per potongan (tanpa OOV); vocab lebih
  besar via `--vocab-size` + latih ulang.
- Koleksi data otomatis (TurnBackHoax + RSS, ada dedupe):
  `python scripts/auto_collect.py --tbh-pages 2 --rss 1`
- Regenerasi chat: `python3 scripts/csv_to_chat.py` lalu gabung
  `data/chat_hoaks.jsonl` + `data/contoh_chat.jsonl` menjadi
  `data/train_chat.jsonl`.

---

## Penyebaran & Akses Jarak Jauh

Jalankan Sagi di komputer GPU, akses dari mana saja:

```bash
# Komputer teman (server):
SAGI_SIZE=Plosagata_Go SAGI_CKPT=pure-Go-sagi-best.pt SAGI_TOKEN=RAHASIA \
  .venv/bin/python web/app.py
```

Pasang Tailscale (gratis) dengan akun yang sama di kedua mesin, lalu dari
peramban rumah buka `http://100.x.y.z:5000/chat` dan isi token sekali
(tersimpan otomatis). Proteksi: endpoint chat memakai token
(`X-Sagi-Token`), CORS terbuka, bobot `.pt` tidak pernah masuk Git.

---

## Struktur Repositori

```
PlosagataA-1.0/
├── plosagata/    Model chat murni (config, layers, model, guard, slang)
├── hoaxcore/     Klasifier 12M (model, tokenizer, dataset, factcheck, ocr, retrieval)
├── scripts/      Antarmuka baris perintah (train, infer, chat, korpus, ctx7, remote)
├── web/          app.py, chatbot.py (SagiEngine), templates/
├── data/         Dataset + bank kata
├── docs/         ARSITEKTUR.md, PANDUAN_FINETUNE.md
├── sagi.sh       Peluncur (mis. SUPER+S di Hyprland)
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

Seluruh skrip kembali otomatis ke akar repo, jadi dapat dijalankan dari direktori
mana pun. Dokumentasi arsitektur per berkas: [`docs/ARSITEKTUR.md`](docs/ARSITEKTUR.md).

---

## Peta Jalan

- [x] Klasifier 12M terlatih (val 94,9%, probe 6/6) + web validator + OCR poster
- [x] Arsitektur Plosagata Go/More/Next + guard topik + slang + retrieval sumber
- [x] Web chatbot + API + akses remote bertoken
- [x] Pra-latih Go di GPU (korpus Wikipedia 2000 artikel) + instruction-tuning bervalidasi
- [ ] Pra-latih More di GPU (korpus Wikipedia, ±5 hari)
- [ ] Instruction-tuning → `pure-Plosagata_More.pt` rilis pertama
- [ ] Evaluasi: matriks konfusi, uji 100 klaim nyata, demo poster

---

## Pembuat

- **rewsaqy** — pemilik ide, arsitektur, data, training, dan pengujian.
- **OpenCode (AI assistant)** — implementasi kode, audit, dan dokumentasi
  bersama pemilik selama pembangunan.

Proyek personal — tidak menerima kontribusi luar. Riwayat kerja tercatat
penuh via `git log`.

---

## Lisensi

Proyek ini berlisensi MIT, lihat [`LICENSE`](LICENSE).

> *Saring sebelum Sharing.*


---

<a id="bagian-2"></a>

# Bagian 2: Riwayat 0-Sekarang

(Sumber: `docs/RIWAYAT_PLOSAGATA.md`)

# Riwayat Pembangunan PlosagataA-1.0 — Nol Sampai Sekarang

> Dokumen konteks untuk AI/pembaca baru. Baca file ini dulu sebelum
> menyentuh kode. Terakhir diperbarui: 14 Okt 2026. Status: aktif.

## 1. Identitas proyek

- **Nama:** PlosagataA-1.0. Produk chat: **Sagi** (asisten validator berita).
- **Repo:** `github.com/rewsaqy/PlosagataA-1.0`. Lisensi MIT.
- **Tujuan:** lomba AI terapan — sistem validasi hoaks Bahasa Indonesia end-to-end.
- **Prinsip keras:** 100% arsitektur + bobot buatan sendiri. Tidak ada model unduhan
  (Qwen sempat dipakai lalu **dihapus total**: kode + 1,2 GB cache).
- **Tim:** pemilik (rewsaqy: ide + data + testing) dan asisten AI OpenCode
  (arsitektur + implementasi + audit).

## 2. Lini masa (fase, bukan tanggal per jam)

1. **Awal — Klasifier 12M.** Transformer encoder (vocab 5000, dim 384, 6 layer,
   6 head) + tokenizer kata + web Flask + 20 contoh. Parameter: 11.584.515.
2. **Chat murni — Plosagata.** Decoder-only GPT dari nol (`plosagata/`):
   tiny 6M → small 92M → b1 912M (22 layer) → varian dalam → final 4 nama resmi:
   Flash 12.665.856, Go 91.593.216, More 912.008.192, Next 1.549.927.424.
   Teknik standar: weight tying, init GPT-2, repetition penalty, no-repeat-ngram,
   KV-cache (teruji identik 1,7e-6), EOS stop.
3. **Guard + identitas.** Topic guard keyword → stemming + kamus akar,
   wake-word "Sagi", sapaan/penolakan 3 varian deterministik (md5),
   memori sesi + follow-up aman (kata off-topic mem-veto).
4. **Data.** CSV 20 → 582 (kolektor otomatis TurnBackHoax+RSS + manual),
   chat 7 → 701 contoh, korpus Wikipedia bertahap (10 blok → 104 chunk RAG),
   bank kata 400 → 106 ribu (HF indonesia_vocabulary), slang 53 → 199.
5. **Retrieval (RAG).** BM25 + rerank TF-IDF offline atas korpus lokal,
   fallback Wikipedia live + link TurnBackHoax/CekFakta, ambang relevansi,
   deteksi fiksi/simulasi (label biru, bukan merah HOAKS).
6. **Deployment.** Web validator + chatbot ala ChatGPT, API JSON + token,
   halaman /models (ganti engine beneran: bongkar total + preload + smoke),
   slash-command (/models /model /status /help), server remote via Tailscale,
   perintah terminal `sagi`.
7. **Disiplin.** Evals (fiksi, topik 7/7, verdict 6/6, RAG), `BUGS.md` (13 bug
   terdokumentasi + diperbaiki), Context7 MCP untuk dokumentasi library.

## 3. Keputusan arsitektur penting (dan alasannya)

- **Dua otak, bukan satu:** 12M menilai (pasti, cepat, terlatih 94%),
  Plosagata mengobrol (fasih hasil training). Juri + heuristic menggabung
  lewat `final_label` tunggal — badge tak boleh kontradiksi kesimpulan.
- **Template sebelum model pintar:** guard + jawaban ekstraktif (verdict+sumber)
  adalah perancah darurat; pensiun via `--pure-think` setelah training cukup.
- **Bobot tak pernah masuk git** (`.gitignore`): kode publik, otak privat
  (Release/Drive). Pernah nyangkut dump 849 MB → dibersihkan dari history.
- **Eksplisit > implisit:** auto-attach checkpoint liar dihapus setelah
  merusak determinisme eval. Ukuran ckpt terdeteksi (`detect_size`), salah
  pasang ditolak dengan pesan manusiawi.
- **Skala jujur:** target realistis = spesialis hoaks Indonesia, bukan ChatGPT.
  5 hari RTX 3080 ≈ 3-4 epoch More atas korpus 4 GB.

## 4. Angka kunci (terverifikasi, bukan klaim)

- Klasifier: val 94,9% (data 582, split bersih pasca-fix leakage).
- Tokenizer: `tsunami` 7 potong → 1 token (bank kata bekerja).
- Training tiny: loss 8,53 (teori acak) → turun per epoch.
- RAG: smoke OK, hit@3 rendah di korpus kecil (obat: korpus, bukan kode).
- Go CPU: ~390 token/detik train; b1 forward ~7 dtk/token CPU.

## 5. Pelajaran (untuk pengembang berikutnya)

1. Augmentasi sebelum split = angka kembung (leakage).
2. Loss tanpa -100 = model menghafal padding.
3. `hash()` Python acak per proses — jangan untuk determinisme.
4. Substring matching butuh stemming + kamus akar.
5. Verifikasi > reputasi sumber (satu klaim analis gugur, satu smoke test
   menangkap bug asli).
6. Baris kode bukan prestasi; 7.056 baris cukup — target 20.000 ditolak.

## 6. Status + berikutnya

Selesai: semua di Peta Jalan README kecuali training besar.
Berikutnya: pra-latih More di GPU (korpus 4 GB berjalan) → `pure-Plosagata_More.pt`
→ evaluasi 100 klaim nyata → demo poster → lomba.


---

<a id="bagian-3"></a>

# Bagian 3: Arsitektur Total

(Sumber: `docs/ARSITEKTUR.md`)

# Arsitektur PlosagataA-1.0 — Bedah Total Bagaimana Semua Kode Berbicara

> Kembali: `README.md` · Santai: `howitworks/Cara_kerja_sagi.md` ·
> Nambah file: `howitworks/Cara_nambah_file.md` · Bug: `BUGS.md`

Dokumen ini menjawab: modul apa memanggil modul apa, dengan data apa,
berbentuk apa, sebesar apa. Angka tensor ditulis eksplisit.

## 0. Peta ketergantungan modul (siapa import siapa)

```
scripts/* ──► plosagata.* ──► (torch saja)
   │              ├── config   (murni data, tanpa import internal)
   │              ├── layers   (torch) → dipakai model
   │              ├── model    (config, layers) → PureGPT, detect_size
   │              ├── guard    (mandiri: re, hashlib)
   │              └── slang    (mandiri: re)
   │
   ├─► hoaxcore.* ──► (torch, sklearn, tokenizers)
   │        ├── model      (torch) → TinyHoaxTransformer
   │        ├── tokenizer  (tokenizers lib) → WordTokenizer + piece_*
   │        ├── dataset    (torch + slang) → HoaxDataset
   │        ├── factcheck  (re) → signals, verdict, final_label, ringkasan
   │        ├── retrieval  (urllib + rag) → cari_sumber
   │        ├── rag        (rank_bm25, sklearn, slang) → index + search
   │        ├── ocr_poster (PIL/tesseract, lazy) → teks poster
   │        └── shared     (model+tokenizer) → SATU instance 12M
   │
   └─► web/* ──► (flask + dua paket di atas)
            ├── app.py     (routing, token, registry engine, history)
            ├── chatbot.py (SagiEngine: guard→model→verdict→sumber)
            └── templates/ (index, chat, models)
```

Aturan: `plosagata` tidak boleh import `hoaxcore` (tetap murni).
`hoaxcore` boleh memakai `plosagata.slang` (satu arah).
`web` boleh memakai keduanya. `scripts` hanya pemanggil.

## 1. Siklus hidup satu chat: "Bener gak BLT cair?" (detail penuh)

Ikuti nomornya — tiap panah = pemanggilan fungsi nyata:

```
[Browser] POST /api/chat {message, session}
   │
[web/app.py:chat_api]
   ├─ _cek_token() → lolos (tanpa token = 401)
   ├─ teks diawali "/"? → cmd_chat (jalur slash-command, selesai)
   └─ get_engine() → SagiEngine(Plosagata_Go) [cache _engines]
        │
[web/chatbot.py:SagiEngine.reply]
   ├─ 1. normalize_slang: "Bener gak..." (slang→formal, regex \b)
   ├─ 2. is_news_topic → "news" (stem "cair" + keyword BLT/bansos...
   │      guard STEM dulu: "mencairkan"→"cair", baru cocok kata-per-kata)
   ├─ 3. _sesi_get(session_id): slot {topik, aktif}; topik="..." disimpan
   ├─ 4. verdict() → shared.predict_proba:
   │      tok.encode → ids[128] → TinyHoaxTransformer → softmax
   │      → {"label":"meragukan","probs":[.2,.5,.3]}
   ├─ 5. local_verdict(teks, v): heuristic_signals (keyword×20, caps,
   │      URL, HP, emoji, fitnah_tokoh +50, fiksi-deteksi) → skor 0-100
   │      → final_label SATU label (fiksi→biru, ≥60→hoaks, dst)
   ├─ 6. cari_sumber: RAG lokal (BM25 top-20 → TF-IDF cosine top-3,
   │      ambang 0.25) → kosong? fallback Wikipedia live + link cek fakta
   ├─ 7a. Model BELUM terlatih → ringkasan_hasil: verdict + 2 sumber
   │      (ekstraktif, deterministik, tanpa generate)
   ├─ 7b. Model TERLATIH/pure-think → PureGPT.generate:
   │      encode → 22×(attention+FFN) per token + KV-cache + EOS stop
   │      + no-repeat-3gram → decode → teks
   └─ 8. JSON {reply, topic, verdict{label,skor,kesimpulan}, sumber}
        → browser render bubble + badge + link sumber
```

Bentuk data antar tahap: `str → str(bersih) → label str → ids[128] int →
probs[3] float → skor int → JSON`. Tidak ada yang hilang/berubah diam-diam:
tipe tiap panah diuji eval (`eval_topik`, `eval_fiksi`, `eval_verdict`).

## 2. Siklus validator web: tempel teks + upload poster

```
POST /api/check (form: text + file poster?)
 ├─ ocr_poster(path) → teks poster (Tesseract ind+eng; gagal → "")
 ├─ text + poster_text digabung
 ├─ model_predict → get_model (singleton shared) → label+probs
 ├─ local_verdict → skor + label_final + saran
 ├─ simpan _history (deque 20: waktu, cuplikan, label FINAL, skor)
 └─ JSON {input, poster_text, model, verifikasi}
GET /api/history → 20 terakhir · GET /api/stats → hitung per label
```

Badge UI membaca `verifikasi.label_final` — TIDAK PERNAH label mentah model.
Itulah obat bug "skor 100 tapi badge VALID".

## 3. Ganti model beneran (`/models`, `/model X`)

```
pilih X → select_model(X):
  1. unload_engine(lama): del objek + gc + cuda.empty_cache → RAM kembali
  2. ACTIVE = X; bangun SagiEngine(X) baru (tokenizer + sesi kosong)
  3. PRELOAD bobot: _get_model() sekarang juga (bukan malas)
     ├─ ckpt cocok? detect_size(ckpt)==X → load, lapor TERLATIH ✅
     ├─ tak cocok? detect_size → alihkan ukuran + rebuild tokenizer
     └─ gagal total? fallback Go + pesan manusiawi (bukan traceback)
  4. Smoke: generate 3 token → ukur detik → "X NYALA dalam N dtk"
  5. Chat berikut memakai engine baru (sesi lama ikut mati bersamanya)
```

`/api/models` list: nama, display, params (tabel terverifikasi, tanpa bangun
model), layers, dim, loaded, trained, ckpt_tersedia (deteksi file saja).

## 4. Training: dua jalur, dua bahan

```
Jalur A — klasifier (scripts/train.py):
  CSV → split DULU (perimentasi strata) → augmentasi KHUSUS train
  → HoaxDataset (slang-normalize!) → DataLoader batch
  → AdamW + cosine → val tiap epoch → best.pt + vocab.json
  Aturan suci: augmentasi sebelum split = angka kembung (pernah kejadian).

Jalur B — chat (scripts/train_pure.py):
  Korpus (.txt, blok \n\n) → CorpusDataset (chunk max_len, collate -100)
  ATAU chat (.jsonl) → ChatLMDataset (target pad → -100, JANGAN latih pad!)
  → fp16 + grad-checkpoint + adam8bit (GPU kecil) → --ckpt lanjutkan
  → out.pt + out.vocab.pt + out.tokenizer.json (backend BPE penuh!)
  Urutan load inference: .tokenizer.json > .vocab.pt+pieces > build-seed.
```

## 5. Tokenizer: tiga nyawa

1. **Penuh** (`.tokenizer.json`): backend BPE persis training. Ideal.
2. **Dict + pieces** (`.vocab.pt`): longest-match darurat per kunci `Ġ`.
3. **Build ulang** (seed + data): hanya untuk eksperimen, ID tak dijamin sama
   (terbukti: 266/8000 cocok) — JANGAN untuk ckpt orang.

## 6. Guard: 3 lapis saringan

```
strip_wake ("Sagi, ...") → _is_greeting (kata utuh, bukan substring)
→ ada_kata_berita (stem+frasa) → ada_kata_offtopic
→ "?"-rule → follow-up aman HANYA bila ada topik sesi DAN tanpa kata off-topic
   ("Siapa pacarmu?" + topik aktif = tetap DITOLAK)
```

Stemmer mini + kamus `PENGECUALIAN` (aturan buta akar: memberitakan→mberita
tanpa kamus). Sapaan varian via md5 (stabil lintas restart — `hash()` dilarang).

## 7. RAG: lokal dulu, live cadangan

```
rag_search: tokenize(slang+clean) → BM25 top-20 (tanpa gate skor>0, karena
  IDF bisa nol) → TF-IDF cosine → buang cosine≤0 → top-3
  → kosong? cari_wikipedia live → tetap kosong? link cek-fakta manual
saring_relevan: minimal kata bersama judul (anti-anime-nyasar)
Index: build_index.py (chunk 300/50) → data/rag_index/ (gitignore, build lokal)
```

## 8. Memori & proses

- Satu proses Flask: 12M SATU instance (`shared.py`), engine chat per ukuran
  (max 1 aktif + cache), sesi `{topik, aktif}` dibatasi 500 + kadaluarsa 2 jam.
- Thread torch dibatasi setengah core; reloader MATI bawaan (fork+thread=crash).
- Tanpa GPU: Go ~390 tok/detik train; b1 forward ~7 dtk/token. Jujur di README.

## 9. Kalau mau ubah sesuatu, edit mana? (lengkap)

| Keinginan | File | Fungsi |
|---|---|---|
| Kata slang baru | `plosagata/slang.py` | 1 baris dict |
| Kata topik | `plosagata/guard.py` | `*_KEYWORDS` + `PENGECUALIAN` bila berimbuhan aneh |
| Ukuran model baru | `plosagata/config.py` | 1 baris CONFIGS + PARAMS + DISPLAY |
| Sampling/anti-ngelantur | `plosagata/model.py` generate args | temp/top_k/no_repeat/eos |
| Skor/aturan verdict | `hoaxcore/factcheck.py` | bobot `local_verdict`, `final_label` |
| Sumber internet | `hoaxcore/retrieval.py` + `rag.py` | ambang, fallback |
| Endpoint web | `web/app.py` | route + `select_model` |
| Perilaku chat | `web/chatbot.py` | `SagiEngine.reply` |
| Tampilan | `web/templates/` | html/js langsung |
| Data | `data/` + regenerate | `csv_to_chat` → gabung `train_chat.jsonl` |


---

<a id="bagian-4"></a>

# Bagian 4: Panduan Fine-Tune

(Sumber: `docs/PANDUAN_FINETUNE.md`)

# 📚 Panduan Fine-Tune Lengkap

> Kembali: `README.md` · Pindah PC: `howitworks/cara_pindahinIQsagi.md` ·
> Nambah file: `howitworks/Cara_nambah_file.md`

## Persiapan di Komputermu (Tanpa GPU)

### 1. Siapkan Data
```bash
bash prepare_finetune.sh
```

### 2. Copy File ke Komputer Temen
Copy seluruh folder `hoax-validator-12m/` ke flashdisk/Google Drive.

## Di Komputer Temen (Dengan GPU)

### Minimal Spesifikasi
- GPU: NVIDIA dengan VRAM 6GB+ (GTX 1060, RTX 2060, RTX 3060, dll)
- RAM: 16GB+
- Storage: 10GB+ kosong
- OS: Windows/Linux
- Python 3.10+

### 1. Install Dependencies
```bash
pip install torch scikit-learn
```

### 2. Training Model Classifier 12M
Model ini yang mengklasifikasi teks jadi Valid/Hoaks/Meragukan.
```bash
python scripts/train.py --data data/contoh_dataset.csv --epochs 20 --batch 32 --vocab-size 5000 --max-len 128
```
- Hasil: `checkpoints/best.pt`, `checkpoints/vocab.json`
- Waktu: ~5-15 menit di GPU

### 3. Training Model Murni Plosagata b1 (~912M)
Ini model buatan sendiri (arsitektur di `plosagata/`). Butuh GPU 24GB+;
di RTX 2060 6GB pakai `--batch 1 --grad-accum 16` dan bersabar (lambat).
```bash
python scripts/train_pure.py --size Plosagata_More --data data/train_chat.jsonl --epochs 3 --batch 1 --grad-accum 16
```
- Hasil: `pure-Plosagata_More.pt` (~3.6GB)
- Waktu: berjam-jam di GPU kecil — wajar untuk 900M parameter dari nol

### 4. Copy Hasil Kembali
Copy file ini ke komputermu:
- `checkpoints/` (model 12M)
- `pure-Plosagata_More.pt` (model murni terlatih)

## Setelah Copy Balik ke Komputermu

### Test Sagi
```bash
# Perintah terminal (dari mana saja):
sagi
sagi "Sagi"

# Chat model murni:
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --prompt "Sagi"
# Setelah training b1 di GPU dan copy pure-Plosagata_More.pt balik:
.venv/bin/python scripts/pure_chat.py --size Plosagata_More --ckpt pure-Plosagata_More.pt
```

### Jalankan Web
```bash
python web/app.py
# Buka http://localhost:5000
```

## Tips Agar AI Ngomong Lancar

1. **Perbanyak data chat** - Tambah contoh di `data/contoh_chat.jsonl` (target 200-500 contoh)
2. **Variasikan gaya bahasa** - Formal, informal, slang, Javanesse-mixed
3. **Multi-turn conversation** - Buat contoh percakapan 2-3 giliran
4. **Epoch yang cukup** - Untuk data kecil (<500), pakai 5-8 epoch
5. **Learning rate rendah** - Mulai 2e-4, kalau overfitting turunkan ke 1e-4

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| CUDA out of memory | Turunkan batch size: `--batch 1` dan naikkan `--grad-accum 16` |
| Model ngawur | Tambah data training, turunkan learning rate |
| Respons pendek/repetitif | Naikkan epoch, tambah contoh respons panjang |
| Import error tokenizers | `pip install tokenizers` |


---

<a id="bagian-5"></a>

# Bagian 5: Pintu How-It-Works

(Sumber: `howitworks/Cara_kerja_sagi.md`)

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


---

<a id="bagian-6"></a>

# Bagian 6: Pindah IQ

(Sumber: `howitworks/cara_pindahinIQsagi.md`)

# Cara Pindahin IQ Sagi + Cara Kerja Sistemnya

> Peta: `README.md` · cara kerja: `Cara_kerja_sagi.md` (folder ini) ·
> nambah file: `Cara_nambah_file.md` (folder ini)
>
> IQ Sagi = **file**, bukan install-an. Pindahin file = pindahin kepintaran.

## 1. File apa saja yang membawa IQ?

| File | Isi | Ukuran | Wajib? |
|---|---|---|---|
| `pure-Go-sagi-best.pt` | Bobot Go 92M terbaik (val 0.10) = otaknya | ±350 MB | Ya |
| `pure-Go-sagi-best.pt.vocab.pt` | Kamus pasangan best | KB-an | Ya (tanpa ini output sampah) |
| `pure-Go-pretrain-best.pt` + `.pt.vocab.pt` | Bobot pretrain Wiki (sebelum instruction-tune) | ±350 MB | Kalau mau tune ulang sendiri |
| `pure-Plosagata_Go-10ep.pt` + `.pt.vocab.pt` | Bobot lama tanpa pretrain/val (jangan dipakai) | ±350 MB | Tidak |
| `pure-Plosagata_More.pt` | Bobot More 912M | ±3,6 GB | Kalau training More |
| `checkpoints/` | Model 12M + vocab | ±90 MB | Ya (validator web pakai ini) |

Catatan nama: `--out X.pt` otomatis bikin `X.pt.vocab.pt` di sebelahnya
(bukan `X.vocab.pt`), dan tiap best baru otomatis dicopykan
`X-best.pt.vocab.pt`.
**Jangan pernah** memisahkan `.pt` dari `.vocab.pt` pasangannya.
Contoh benar: `pure-Go-sagi-best.pt` ↔ `pure-Go-sagi-best.pt.vocab.pt`.

## 2. Cara pindah (pilih satu)

### A. Flashdisk (paling gampang)
1. Colok, pastikan format **exFAT/NTFS** (klik kanan → Properties).
   FAT32 **menolak** file > 4 GB — More 3,6 GB mepet bahaya.
2. Copy pasangan best + validator: `pure-Go-sagi-best.pt`,
   `pure-Go-sagi-best.pt.vocab.pt`, dan folder `checkpoints/`.
3. Di komputer tujuan: copy ke **dalam folder proyek** (sejajar `scripts/`, `data/`).
4. Tes: `.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --ckpt pure-Go-sagi-best.pt --prompt "halo"`
   (vocab pasangan terdeteksi otomatis; kalau mau eksplisit tambah
   `--vocab-file pure-Go-sagi-best.pt.vocab.pt`).`

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
ls -lh pure-*.pt*
# 2. Sapa (harus identitas Sagi, instan):
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --ckpt pure-Go-sagi-best.pt --prompt "halo"
# 3. Berita (masih word-salad Indonesia tahap ini, bukan byte sampah):
.venv/bin/python scripts/pure_chat.py --size Plosagata_Go --ckpt pure-Go-sagi-best.pt --prompt "Bener gak giveaway iPhone klik link ini?"
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


---

<a id="bagian-7"></a>

# Bagian 7: Nambah File

(Sumber: `howitworks/Cara_nambah_file.md`)

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


---

<a id="bagian-8"></a>

# Bagian 8: Catatan Bug

(Sumber: `BUGS.md`)

# BUGS.md — Catatan Bug Jujur PlosagataA-1.0

> Aturan: setiap bug yang ditemukan WAJIB masuk sini dengan status.
> Status: `FIXED` (diperbaiki + teruji) · `DOKUMEN` (keterbatasan sadar) · `BUKA` (belum).

## Kritis (sudah FIXED semua)

| ID | Bug | Perbaikan | Bukti |
|---|---|---|---|
| B-01 | Badge VALID padahal skor 100 (label model vs kesimpulan heuristic kontradiksi) | `final_label()` satu pintu di `hoaxcore/factcheck.py` | Kasus bumbu-dapur → HOAKS 100 |
| B-02 | Tokenizer BPE backend hilang (train simpan dict saja → inferensi `<unk>` massal) | `train_pure` simpan `.tokenizer.json` penuh; loader `JSON > dict+pieces > seeds` | ckpt Go ke-load + generate kata ID |
| B-03 | Guard bypass `?` ("Siapa pacarmu?" jadi follow-up saat ada topik) | `is_follow_up_safe`: kata off-topic mem-veto | `pacar? → False`, `kapan cairnya? → True` |
| B-04 | `eval_fiksi` baris 44 gagal | **TIDAK TERBUKTI** — lulus 3x di sini. Dugaan analis: versi basi. Tetap dimonitor tiap push | `EVAL-FIKSI LULUS` |
| B-05 | Substring guard ("buang"→"uang") | Stemmer mini + cocok kata-per-kata (frasa multi-kata tetap substring) | `buang→offtopic`, `memberitakan→news` |
| B-06 | Sapaan acak antar restart (`hash()` Python) | `hashlib.md5` deterministik | stabil lintas proses |
| B-07 | Loss melatih token padding (mode instruction) | Target pad → `-100` di `ChatLMDataset` | loss turun wajar, bukan hafal pad |
| B-08 | Frontend cari `shortened_urls[]`/`phone_numbers[]`, backend kirim boolean | Backend kirim array hasil `findall` (+ bool lama kompat) | sinyal tampil di UI |
| B-09 | Slang tak dinormalisasi di jalur 12M | `normalize_slang` di dataset(train), app, chatbot, infer | konsisten hulu-hilir |
| B-10 | Model 12M dimuat 2x (app + engine) | `hoaxcore/shared.py`: satu instance per proses | `is` sama, RAM hemat |
| B-11 | Generate tanpa EOS stop + O(N²) tanpa cache | Stop di `<sep>` + KV-cache (teruji identik 1,7e-6) | berhenti dini + cepat |
| B-12 | Auto-ckpt liar: file `.pt` tetangga mengubah perilaku engine (eval_topik gagal misterius) | Auto-attach **dihapus**; ckpt hanya eksplisit (`--ckpt`/env). `/api/models` tampilkan `ckpt_tersedia` saja | `EVAL-TOPIK LULUS` stabil |
| B-13 | Salah pasang ukuran (ckpt Go → model More = dinding `size mismatch`) | `detect_size()` + auto-alih + pesan manusiawi | request More+ckpt Go → jalan sebagai Go |

## Keterbatasan sadar (DOKUMEN, bukan bug)

| ID | Batasan | Artinya |
|---|---|---|
| L-01 | Template/UI pakai Google Fonts + Tailwind CDN | Demo **offline** rusak tampilannya. Fungsionalitas tetap. Perlu mode offline penuh |
| L-02 | Hit RAG rendah di korpus kecil (hit@3 3-7/20) | Obatnya korpus 100+ chunk, bukan kode |
| L-03 | Model < terlatih penuh = cadel | `ringkasan_hasil` ekstraktif menutupinya; `--pure-think` jujur apa adanya |
| L-04 | More/Next di RAM 19GB berisiko OOM-killer (SIGKILL) | `select_model` fallback ke Go + pesan jelas |

## Cara lapor bug baru (untuk kontributor)

1. Tulis baris reproduksi minimal (perintah + input + output salah).
2. Tambah ke tabel atas dengan status `BUKA`.
3. Perbaiki → ubah status `FIXED` + tulis bukti uji (bukan klaim).


---

<a id="bagian-9"></a>

# Bagian 9: Kontribusi

(Sumber: `CONTRIBUTING.md`)

# Kontribusi

PlosagataA-1.0 adalah proyek personal oleh **rewsaqy** (dibangun bersama
asisten AI OpenCode) dan **tidak menerima kontribusi luar** saat ini.

Alasannya: lomba menilai karya pemilik. Fork untuk belajar dipersilakan
(lisensi MIT), tapi pull request tidak akan di-merge.


---
Cermin data (tidak digabung, terlalu besar): `data/contoh_chat.md`, `data/chat_hoaks.md`.
