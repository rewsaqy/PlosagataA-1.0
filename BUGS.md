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
