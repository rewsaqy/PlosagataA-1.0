"""Uji deteksi fiksi + filter relevansi (murni, offline, cepat).

Kasus nyata: artikel yg eksplisit menyatakan dirinya fiktif tidak boleh
lolos sebagai VALID, dan hasil live ngawur harus tersaring.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from hoaxcore import deteksi_fiksi, local_verdict, saring_relevan

ARTIKEL_FIKSI = """
JAKARTA, BERITA NUSANTARA — Pemerintah dalam simulasi ini tengah menguji
sistem peringatan dini bencana berbasis kecerdasan buatan (AI).
"Tujuan pengujian ini adalah melihat bagaimana informasi dari berbagai
sumber dapat disatukan," ujar Arman Pradipta, pejabat fiktif dalam
skenario tersebut. Catatan: nama, pernyataan, dan peristiwa dalam artikel
ini sepenuhnya rekaan untuk simulasi literasi media. Dalam skenario
tersebut, pengujian dilakukan dengan data sintetis. Artikel ini merupakan
contoh penulisan jurnalistik fiktif. Tidak ada pengumuman pemerintah.
"""

BERITA_LATIHAN_ASLI = """
BPBD menggelar simulasi evakuasi bencana gempa di sekolah. Kegiatan
latihan diikuti 500 siswa dan berjalan lancar menurut kepala dinas.
"""


def main():
    assert deteksi_fiksi(ARTIKEL_FIKSI), "artikel fiktif harus terdeteksi"
    lv = local_verdict(ARTIKEL_FIKSI, {"label": "valid"})
    assert "FIKSI" in lv["kesimpulan"], lv["kesimpulan"]
    assert any("fiksi" in s for s in lv["saran"]), lv["saran"][:1]
    print("fiksi OK:", lv["kesimpulan"][:60])

    # Berita latihan bencana ASLI (1 penanda lemah) tidak boleh kena
    assert not deteksi_fiksi(BERITA_LATIHAN_ASLI), "false-positive latihan!"
    lv2 = local_verdict("Gempa magnitudo 5 mengguncang selatan.", {"label": "valid"})
    assert "FIKSI" not in lv2["kesimpulan"]
    assert "belum terverifikasi" in lv2["kesimpulan"], lv2["kesimpulan"]
    print("nonfiksi OK:", lv2["kesimpulan"][:60])

    # Filter relevansi: junk tanpa kata bersama -> buang, yg relevan lolos
    junk = [
        {"judul": "The Last Man", "url": "u1", "cuplikan": "novel tentang sistem wabah"},
        {"judul": "Ferdy Sambo", "url": "u2", "cuplikan": "mantan perwira tinggi Polri"},
    ]
    r = saring_relevan("ferdy sambo penjara", junk)
    assert [h["judul"] for h in r] == ["Ferdy Sambo"], r
    assert saring_relevan("ferdy sambo penjara", junk[:1]) == []
    # 1 kata bersama hanya di cuplikan (bukan judul) -> tetap buang
    assert saring_relevan("peringatan bencana", junk[:1]) == []
    print("saring OK")
    print("EVAL-FIKSI LULUS")


if __name__ == "__main__":
    main()
