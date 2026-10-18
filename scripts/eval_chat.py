"""Eval inti chat Sagi: format prompt (sama dgn training), pembersih output,
memori sesi, sitasi — plus uji opsional mesin penuh dengan ckpt kecil.

Bagian 1 (selalu jalan, ringan — tanpa bobot):
  1. bangun_prompt berakhir "assistant:" + riwayat terbawa urut lama->baru.
  2. Pesan kepanjangan dipotong (elipsis) & newline dinetralkan.
  3. bersihkan_output memotong bocoran peran & merapikan spasi.
  4. SesiStore: topik, cap giliran, batas jumlah sesi.
  5. kumpulkan_sumber + blok_sitasi.

Bagian 2 (opsional, dgn --ckpt): token prompt TIDAK berakhir <sep> &
  tidak berisi padding; aliran meta -> token* -> final konsisten.

Contoh:
  .venv/bin/python scripts/eval_chat.py
  .venv/bin/python scripts/eval_chat.py --ckpt /tmp/opencode/tiny_ckpt/tiny-best.pt
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

GAGAL = []


def cek(nama, syarat, info=""):
    tanda = "LULUS" if syarat else "GAGAL"
    print(f"  [{tanda}] {nama}" + (f" — {info}" if info else ""))
    if not syarat:
        GAGAL.append(nama)


def bagian1_logika():
    from sagi.prompt import bangun_prompt, bersihkan_output
    from sagi.session import MAKS_GILIRAN, SesiStore
    from sagi.sources import blok_sitasi, kumpulkan_sumber

    print("== Bagian 1: logika dasar (tanpa bobot) ==")

    from plosagata import normalize_slang
    from sagi.prompt import LABEL_KONSUMEN, LABEL_SISTEM, SISTEM_DEFAULT

    p = bangun_prompt("Bener gak bansos cair?")
    # Uji terkuat: prompt == transformasi training (train_12m.py) persis.
    seharusnya = normalize_slang(
        f"system: {SISTEM_DEFAULT}\nuser: Bener gak bansos cair?\nassistant:")
    cek("prompt == transformasi training", p == seharusnya, repr(p[:60]))
    cek("prompt berakhir 'assistant:'", p.rstrip().endswith("assistant:"), repr(p[-40:]))
    cek("label peran ternormalisasi (konsumen/sistem)",
        f"{LABEL_KONSUMEN}:" in p and f"{LABEL_SISTEM}:" in p)
    cek("prompt tanpa baris kosong", "\n\n" not in p)

    p2 = bangun_prompt("kapan cairnya?",
                       riwayat=[("Bener gak bansos cair?", "Iya, ini penjelasannya.")])
    i_lama = p2.find("benar tidak bansos")
    i_baru = p2.find("kapan cairnya?")
    cek("riwayat ikut terbawa", "penjelasannya" in p2 and i_lama >= 0)
    cek("urutan transkrip lama->baru", 0 <= i_lama < i_baru, f"{i_lama} < {i_baru}")
    cek("jumlah baris user = 2 (riwayat + baru)", p2.count(f"{LABEL_KONSUMEN}:") == 2)

    panjang = bangun_prompt("kata " * 500)
    cek("pesan kepanjangan dipotong (…)", "…" in panjang and len(panjang.split()) < 200,
        f"{len(panjang.split())} kata")

    b1 = bersihkan_output("Halo ini jawaban\nuser: lanjut nanya\nassistant: hi")
    cek("bocoran peran dipotong", "user:" not in b1 and "assistant:" not in b1, repr(b1))
    b3 = bersihkan_output("Halo ini jawaban\nkonsumen: lanjut nanya")
    cek("bocoran label ternormalisasi dipotong", "konsumen:" not in b3, repr(b3))
    b2 = bersihkan_output("assistant:  jawaban   dengan   spasi  ganda .")
    cek("label awal dibuang & spasi dirapikan",
        not b2.lower().startswith("assistant") and "  " not in b2, repr(b2))

    st = SesiStore(maks_sesi=3, umur=60)
    s1 = st.ambil("a")
    cek("sesi dibuat & topik awal None", s1 is not None and st.topik("a") is None)
    s1.topik = "Berita X"
    cek("topik tersimpan", st.topik("a") == "Berita X")
    for i in range(MAKS_GILIRAN + 5):
        st.catat_giliran("a", f"tanya {i}", f"jawab {i}")
    cek("giliran dibatasi", len(st.ambil("a").giliran) == MAKS_GILIRAN,
        str(len(st.ambil("a").giliran)))
    for k in ("b", "c", "d", "e"):
        st.ambil(k)
    cek("jumlah sesi dibatasi", st.jumlah() <= st.maks_sesi + 1, str(st.jumlah()))
    cek("sesi tanpa id -> None", st.ambil("") is None)

    sm = kumpulkan_sumber({
        "wikipedia": [{"judul": "Vaksin", "url": "https://id.wikipedia.org/wiki/Vaksin", "cuplikan": "x"}],
        "web": [{"judul": "Kabar", "url": "https://contoh.id/a"}],
        "cek_fakta": [{"judul": "TBH", "url": "https://turnbackhoax.id/x"}],
    })
    cek("sumber dinormalisasi (maks 3)", len(sm) == 3 and sm[0]["jenis"] == "CekFakta", str(sm))
    blok = blok_sitasi({"label": "hoaks", "skor": 87},
                       {"cek_fakta": [{"judul": "TBH", "url": "u"}]})
    cek("blok sitasi memuat verdict & sumber",
        "HOAKS" in blok and "87/100" in blok and "TBH" in blok)
    cek("blok kosong bila tak ada info", blok_sitasi(None, None) == "")


def bagian2_mesin(ckpt, size):
    print("\n== Bagian 2: mesin penuh (uji dengan ckpt) ==")
    # Paksa semua pesan lewat jalur model (bukan guard halo/off-topic)
    os.environ["SAGI_PURE_THINK"] = "1"
    from sagi import SagiEngine

    eng = SagiEngine(size=size, ckpt=ckpt)
    cek("engine trained", eng.trained)

    ids = eng._tokenisasi("user: halo sagi\nassistant:")
    cek("prompt tidak berakhir <sep>", bool(ids) and ids[-1] != eng.tok.sep_id,
        f"ekor={ids[-3:] if ids else []}")
    cek("prompt tanpa padding", all(i != eng.tok.pad_id for i in ids))

    riwayat = [("berita lama " * 80, "balasan lama " * 80) for _ in range(6)]
    ids2 = eng._ids_prompt("pertanyaan terakhir yang penting?", riwayat, max_new=20)
    cek("prompt muat dalam budget", len(ids2) <= eng.cfg.max_len - 20 - 2, f"{len(ids2)} tok")

    kejadian = list(eng.reply_stream("halo sagi, apa kabar?", max_new=12,
                                     session_id="eval-chat"))
    tipe = [e.get("type") for e in kejadian]
    cek("aliran meta -> token* -> final",
        bool(tipe) and tipe[0] == "meta" and tipe[-1] == "final" and "token" in tipe,
        str(tipe[:6]))
    final = kejadian[-1]
    cek("final punya teks str", isinstance(final.get("text"), str))
    cek("meta punya topic", kejadian[0].get("topic") in
        ("news", "greeting", "offtopic", "followup"), str(kejadian[0].get("topic")))

    ses = eng.sesi.ambil("eval-chat")
    cek("giliran tercatat di memori sesi", len(ses.giliran) >= 1, str(len(ses.giliran)))

    r = eng.reply("halo sagi, apa kabar?", max_new=12, session_id="eval-chat-2")
    cek("reply() non-stream jalan", isinstance(r.get("reply"), str) and "security" in r)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--size", default="Plosagata_12M")
    a = ap.parse_args()

    bagian1_logika()
    if a.ckpt:
        bagian2_mesin(a.ckpt, a.size)
    else:
        print("\n(lewati Bagian 2 — tambahkan --ckpt untuk uji mesin penuh)")

    print("\n" + ("SEMUA LULUS ✅" if not GAGAL else f"GAGAL: {GAGAL} ❌"))
    return 1 if GAGAL else 0


if __name__ == "__main__":
    sys.exit(main())
