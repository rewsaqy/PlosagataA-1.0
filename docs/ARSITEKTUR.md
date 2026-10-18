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
   ├─ 7a. Ckpt chat tak termuat → ringkasan_hasil: jawaban ekstraktif
   │      (verdict + saran; daftar sumber tampil di UI, bukan di teks)
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
 ├─ model_predict → predict_proba (singleton shared) → label+probs
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
