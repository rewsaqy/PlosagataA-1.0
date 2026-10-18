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
   ├─► sagi/* ──► mesin chat (dipakai web & CLI)
   │        ├── engine  → SagiEngine (guard→prompt→verdict→generate streaming)
   │        ├── prompt  → transkrip format training + pembersih output
   │        ├── session → memori sesi (topik + riwayat giliran)
   │        └── sources → normalisasi sumber + blok sitasi
   │
   └─► web/* ──► (flask + paket di atas)
            ├── app.py     (routing, token, registry engine, history, SSE)
            ├── chatbot.py (shim kompatibilitas → re-export `sagi`)
            └── templates/ (index, chat, models)
```

Aturan: `plosagata` tidak boleh import `hoaxcore` (tetap murni).
`hoaxcore` boleh memakai `plosagata.slang` (satu arah).
`sagi` (mesin chat) boleh memakai keduanya; `web`/`scripts` memakainya.

## 1. Siklus hidup satu chat: "Bener gak BLT cair?" (detail penuh)

Ikuti nomornya — tiap panah = pemanggilan fungsi nyata:

```
[Browser] POST /api/chat/stream {message, session}     ← UI kini pakai SSE
   │      (klien lama tetap dilayani: POST /api/chat → reply())
[web/app.py:chat_stream_api]
   ├─ _cek_token() → lolos (tanpa token = 401)
   ├─ teks diawali "/"? → cmd_chat (dibungkus kejadian meta→token→final)
   └─ get_engine() → SagiEngine(Plosagata_Go) [cache _engines]
        │
[sagi/engine.py:SagiEngine.reply_stream]   (generator kejadian SSE)
   ├─ 0. inspect_security: ancaman? → balasan konstitusional + final
   ├─ 1. is_news_topic (guard stem: "mencairkan"→"cair") → news/
   │      greeting/followup/offtopic; follow-up merujuk topik sesi
   ├─ 2. SesiStore.ambil(session): {topik, giliran[≤16]} — B-36
   ├─ 3. verdict() → shared.predict_proba → label+probs[3]
   ├─ 4. local_verdict: heuristic_signals (keyword×20, caps, URL, HP,
   │      emoji, fitnah_tokoh +50, fiksi) → skor 0-100 → label_final
   ├─ 5. cari_sumber: RAG → fallback Wikipedia live + link cek fakta
   │      → kejadian "meta" {topic, verdict, sumber}   (UI: badge)
   ├─ 6a. tanpa bobot → ringkasan_hasil (template) → 1 kejadian "token"
   ├─ 6b. TERLATIH → sagi/prompt.bangun_prompt(≤16 giliran riwayat,
   │        label ternormalisasi "sistem:/konsumen:") → tokenisasi
   │        (buang <sep>!) → budget max_len−max_new − giliran tua
   │        → PureGPT.generate(on_token=…) di thread pekerja + antrean
   │        → kejadian "token" tiap potongan (decode inkremental)
   ├─ 7. bersihkan_output (potong bocoran peran) + sesi.catat_giliran
   └─ 8. kejadian "final" {text, security}; blok sitasi
          "🔎 Verdict · 📚 Sumber" nempel utk news/followup
```

Bentuk data antar tahap: `str → str(bersih) → label str → ids int →
probs[3] float → skor int → kejadian SSE meta→token*→final → render`.
Sejak rombak 18 Okt, bus data UI = SSE; diuji `scripts/eval_chat.py`
(bagian 1 logika tanpa bobot, bagian 2 opsional dengan ckpt).

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
  (max 1 aktif + cache), sesi `{topik, giliran[≤16], aktif}` dibatasi 500 +
  kadaluarsa 2 jam (`sagi/session.py:SesiStore`).
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
| Endpoint web | `web/app.py` | route + `select_model` (+ SSE `/api/chat/stream`) |
| Perilaku chat | `sagi/engine.py` | `SagiEngine.reply_stream` / `reply` |
| Format prompt & pembersih | `sagi/prompt.py` | `bangun_prompt`/`bersihkan_output` |
| Memori percakapan | `sagi/session.py` | `SesiStore` (topik + giliran) |
| Blok sitasi teks | `sagi/sources.py` | `blok_sitasi`/`kumpulkan_sumber` |
| Tampilan | `web/templates/` | html/js langsung |
| Data | `data/` + regenerate | `csv_to_chat` → gabung `train_chat.jsonl` |
