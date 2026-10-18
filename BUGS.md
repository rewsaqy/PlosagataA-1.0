# BUGS.md — Catatan Bug Jujur PlosagataA-1.0

> Aturan: setiap bug yang ditemukan WAJIB masuk sini dengan status.
> Status: `FIXED` (diperbaiki + teruji) · `DOKUMEN` (keterbatasan sadar) · `BUKA` (belum).

## Kritis (sudah FIXED semua)

| ID | Bug | Perbaikan | Bukti |
|---|---|---|---|
| B-01 | Badge VALID padahal skor 100 (label model vs kesimpulan heuristic kontradiksi) | `final_label()` satu pintu di `hoaxcore/factcheck.py` | Kasus bumbu-dapur → HOAKS 100 |
| B-02 | Tokenizer BPE backend hilang (train simpan dict saja → inferensi `<unk>` massal) | `train_pure` simpan `.tokenizer.json` penuh; loader `JSON > dict+pieces > seeds` | ckpt Go ke-load + generate kata ID |
| B-03 | Guard bypass `?` ("Siapa pacarmu?" jadi follow-up saat ada topik) | `is_follow_up_safe`: kata off-topic mem-veto | `pacar? → False`, `kapan cairnya? → True` |
| B-04 | `eval_fiksi` baris 44 gagal (`belum terverifikasi` hilang dari kesimpulan `no_source`) | Sinkronisasi `kesimpulan` di `hoaxcore/factcheck.py` memuat frasa `belum terverifikasi` | `EVAL-FIKSI LULUS` 100% |
| B-05 | Substring guard ("buang"→"uang") | Stemmer mini + cocok kata-per-kata (frasa multi-kata tetap substring) | `buang→offtopic`, `memberitakan→news` |
| B-06 | Sapaan acak antar restart (`hash()` Python) | `hashlib.md5` deterministik | stabil lintas proses |
| B-07 | Loss melatih token padding (mode instruction) | Target pad → `-100` di `ChatLMDataset` | loss turun wajar, bukan hafal pad |
| B-08 | Frontend cari `shortened_urls[]`/`phone_numbers[]`, backend kirim boolean | Backend kirim array hasil `findall` (+ bool lama kompat) | sinyal tampil di UI |
| B-09 | Slang tak dinormalisasi di jalur 12M | `normalize_slang` di dataset(train), app, chatbot, infer | konsisten hulu-hilir |
| B-10 | Model 12M dimuat 2x (app + engine) | `hoaxcore/shared.py`: satu instance per proses | `is` sama, RAM hemat |
| B-11 | Generate tanpa EOS stop + O(N²) tanpa cache | Stop di `<sep>` + KV-cache (teruji identik 1,7e-6) | berhenti dini + cepat |
| B-12 | Auto-ckpt liar: file `.pt` tetangga mengubah perilaku engine (eval_topik gagal misterius) | Auto-attach **dihapus**; ckpt hanya eksplisit (`--ckpt`/env). `/api/models` tampilkan `ckpt_tersedia` saja | `EVAL-TOPIK LULUS` stabil |
| B-13 | Salah pasang ukuran (ckpt Go → model More = dinding `size mismatch`) | `detect_size()` + auto-alih + pesan manusiawi | request More+ckpt Go → jalan sebagai Go |
| B-14 | Follow-up crash `TypeError: unsupported operand + for NoneType and str` saat topik None | Fallback aman `klaim = topik or text` dan `kueri = f"{topik} {text}".strip() if topik else text` | chat followup tanpa topik tetap aman |
| B-15 | `/api/check` & `/api/chat` payload error / AttributeError saat body non-form / bertukar key | Handler seragam `request.get_json(silent=True)` + deteksi fleksibel `message` / `text` | request REST API aman dari malformed JSON |
| B-16 | Checkpoint best (`-best.pt`) kehilangan `.tokenizer.json` BPE backend saat auto-save | `train_pure.py` otomatis copy `.tokenizer.json` bersamaan dengan `.vocab.pt` | loader menemukan BPE backend utuh di ckpt best |
| B-17 | Dictionary key duplikat `"healing"` dan `"bsk"` di `plosagata/slang.py` | Hapus entry duplikat dan optimasi `k in SLANG` | dictionary bersih & linter valid |
| B-18 | Regex ref tag di `fetch_wiki_dump.py` memakai `{{0,1}}` literal di non-fstring | Perbaiki pola regex `<ref[^>]*?/>|<ref.*?</ref>` | pembersihan sitasi akurat tanpa artefak |
| B-19 | PyTorch UserWarning konversi `float(loss)` requiring grad di `gpt_pure.py` | Gunakan `loss.item()` standar PyTorch | forward berjalan bersih tanpa warning |
| B-20 | Tombol kirim terkunci permanen setelah error API (UI mati, `btn.disabled` tidak pernah pulih) | Reset `btn.disabled=false` di semua jalur error/selesai `web/templates/chat.html` | uji paksa error → tombol hidup lagi (bee26f6) |
| B-21 | Potensi XSS pada link sumber (href dirender dari data, bisa `javascript:`) | `safeUrl()` whitelist `http/https` sebelum render link | render link sumber lewat `safeUrl()` (bee26f6) |
| B-22 | `localStorage` korup → halaman chat blank (`JSON.parse` melempar saat load) | `try/catch` fallback `[]` + regenerate `sagi_sid` | reload dengan storage korup tetap jalan (bee26f6) |
| B-23 | `eval_rag` smoke gagal `AssertionError` setelah `cari_sumber` menambah kunci `web` | Assert kunci disinkronkan: `{kueri, wikipedia, web, cek_fakta}` | `eval_rag --mini` lulus lagi |
| B-24 | Auto-copy hasil training ke root repo (duplikat ±305 MB tiap latihan; file 12M & Browser kembar di root) | Blok `shutil.copy` dihapus dari `train_12m.py` & `train_browser.py`; ckpt kanonik hanya di `checkpoints/` | `grep shutil` = 0 hasil; root bersih |
| B-25 | `strip_wake` cocok substring: "sagittarius" dianggap panggilan "Sagi" (sisa pesan ikut terpotong) | Regex batas kata `\bsagi\b` (deteksi + strip) | `eval_topik` LULUS; hanya "sagi" utuh yang memicu |
| B-26 | Fallback `sumber` di `web/chatbot.py` tanpa kunci `web` (bentuk beda dari `cari_sumber`) | Kunci `web: []` ditambahkan — bentuk seragam | `eval_rag --mini` shape LULUS |
| B-27 | `train_12m.py` memproses 256 token/sampel walau teks pendek (padding statis = komputasi boros) | Potong trailing pad sebelum collate (sama seperti `train_browser.py`) | Batch dinamis per-teks (mekanisme identik dgn train_browser); compile+ruff bersih |
| B-28 | Alias typo `Plosata_Rowser` (duplikat 25M di CONFIGS/PARAMS/DISPLAY + `gpt_pure.py`) | Dihapus semua; `Plosagata_Browser` satu-satunya nama 25M | `/api/models` = 6 model (dulu 7); `grep Rowser` = 0 |
| B-29 | REGRESI: `safeUrl()` (fix B-21) hilang saat refactor `chat.html` 18 Okt → href sumber dirender mentah | Dipasang ulang + dipakai di 3 render link sumber (Wiki/Web/CekFakta) | `grep safeUrl` = 4 (1 definisi + 3 pakai) |
| B-30 | Kode & param mati: stub `llm_crosscheck`, `_cari_ckpt_otomatis`, param `sumber`/`saran` tak terpakai, impor mubazir (`math`/`nn`/`re` ganda) | Dihapus semua; impor & alias dirapikan | `ruff check --select F,E9`: All checks passed |

## Keterbatasan sadar (DOKUMEN, bukan bug)

| ID | Batasan | Artinya |
|---|---|---|
| L-01 | Template/UI pakai Google Fonts + Tailwind CDN | Demo **offline** rusak tampilannya. Fungsionalitas tetap. Perlu mode offline penuh |
| L-02 | Hit RAG rendah di korpus kecil (hit@3 3-7/20) | Obatnya korpus 100+ chunk, bukan kode |
| L-03 | Model < terlatih penuh = cadel | `ringkasan_hasil` ekstraktif menutupinya; `--pure-think` jujur apa adanya |
| L-04 | More/Next di RAM 19GB berisiko OOM-killer (SIGKILL) | `select_model` fallback ke Go + pesan jelas |
| L-05 | DuckDuckGo tidak bisa diakses dari jaringan ini (handshake TLS gagal) → `cari_web_live` selalu `[]` | Web live fail-safe (tanpa crash), tapi belum terbukti memberi hasil; butuh provider/API alternatif atau jaringan tanpa blokir |
| L-06 | Val klasifier hanya 93 sampel (split sama dgn `train.py`/`eval_verdict.py`) | Selisih 1 sampel ≈ 1,1% — angka akurasi (18 Okt: 95,7%, probe 6/6) dibaca dengan margin ini |

## Cara lapor bug baru (untuk kontributor)

1. Tulis baris reproduksi minimal (perintah + input + output salah).
2. Tambah ke tabel atas dengan status `BUKA`.
3. Perbaiki → ubah status `FIXED` + tulis bukti uji (bukan klaim).
