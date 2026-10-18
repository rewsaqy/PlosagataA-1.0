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

## 7. 18 Okt 2026 — hari reset & rombak chat

- Pagi: folder bobot terhapus → semua dilatih ulang dari nol (klasifier val
  95,7%; sistem end-to-end 91,4%). 25M Browser & eksperimen LS dihentikan —
  arah semua-ke-12M chat (latihan 8 epoch jalan sendiri di terminal user).
- Audit mesin chat sebelum demo bobot menemukan 3 bug fondasi:
  B-35 prompt tak sinkron training (label ternormalisasi `sistem:/konsumen:`
  + `<sep>` bocor ke prompt), B-36 memori sesi cuma topik, B-37 generate
  tanpa callback token.
- Rombak: paket `sagi/` (engine/prompt/session/sources); `web/chatbot.py` jadi
  shim; endpoint SSE `/api/chat/stream`; bubble chat mengalir token-per-token;
  blok sitasi "🔎 Verdict · 📚 Sumber" menempel di jawaban teks; CLI
  `pure_chat.py` memakai mesin yang sama dengan web (satu kebenaran).
- Uji: `eval_chat` (baru, 2 bagian) LULUS; `eval_topik` + `eval_security`
  LULUS; SSE lolos curl (template & model). TUI ditunda — web tetap dipakai.
- Catatan jujur: smoke test yang jalan di proses training malam ini masih
  versi lama (`user:` mentah) — sudah difix di file untuk run berikutnya.
