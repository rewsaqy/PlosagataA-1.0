"""Gabung semua dokumen proyek -> SATU file ultimate (tergenerate, jangan edit manual).

  python scripts/gabung_dok.py
  -> docs/ULTIMATE_PLOSAGATA.md

Edit file sumbernya masing-masing, lalu jalankan ulang script ini.
Cermin data (data/*.md, ratusan baris tabel) TIDAK ikut — dirujuk saja.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BAGIAN = [
    ("README.md", "README — Gambaran + Perintah"),
    ("docs/RIWAYAT_PLOSAGATA.md", "Riwayat 0-Sekarang"),
    ("docs/ARSITEKTUR.md", "Arsitektur Total"),
    ("docs/PANDUAN_FINETUNE.md", "Panduan Fine-Tune"),
    ("howitworks/Cara_kerja_sagi.md", "Pintu How-It-Works"),
    ("howitworks/cara_pindahinIQsagi.md", "Pindah IQ"),
    ("howitworks/Cara_nambah_file.md", "Nambah File"),
    ("howitworks/Panduan_Training_12M.md", "Panduan Training 12M"),
    ("BUGS.md", "Catatan Bug"),
    ("CONTRIBUTING.md", "Kontribusi"),
]


def main():
    out = [f"# ULTIMATE PlosagataA-1.0 — Satu Kesatuan Dokumentasi\n",
           f"> Digenerate otomatis {date.today()} oleh `scripts/gabung_dok.py`. "
           "JANGAN edit file ini — edit file sumbernya, lalu generate ulang.\n",
           "## Daftar Isi\n"]
    for i, (_, judul) in enumerate(BAGIAN, 1):
        out.append(f"{i}. [{judul}](#bagian-{i})")
    out.append("")
    for i, (path, judul) in enumerate(BAGIAN, 1):
        p = ROOT / path
        isi = p.read_text(encoding="utf-8") if p.exists() else "* (file hilang)*"
        out.append(f"\n---\n\n<a id=\"bagian-{i}\"></a>\n")
        out.append(f"# Bagian {i}: {judul}\n\n(Sumber: `{path}`)\n")
        out.append(isi.strip())
        out.append("")
    out.append("\n---\nCermin data (tidak digabung, terlalu besar): "
               "`data/contoh_chat.md`, `data/chat_hoaks.md`.\n")
    target = ROOT / "docs/ULTIMATE_PLOSAGATA.md"
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"{target} ({target.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
