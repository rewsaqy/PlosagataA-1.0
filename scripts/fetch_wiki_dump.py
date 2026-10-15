"""Download dump Wikipedia utuh -> korpus GB-an (jalan berjam-jam, background).

  python scripts/fetch_wiki_dump.py --lang id --max-gb 4
  python scripts/fetch_wiki_dump.py --lang en --max-gb 1   (opsional Inggris)

Sumber: dumps.wikimedia.org (legal, CC BY-SA). Resume otomatis (wget -c).
Korpus di-gitignore (terlalu besar) — tiap mesin training build sendiri.
Jalankan di background: nohup ... &  atau biarkan OpenCode background.
"""
import argparse
import bz2
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DUMP = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"


def bersihkan(wiki: str) -> str:
    wiki = re.sub(r"<ref[^/]*?/>{{0,1}}|<ref.*?</ref>", "", wiki, flags=re.S)
    wiki = re.sub(r"\{\{[^{}]*\}\}", "", wiki)
    wiki = re.sub(r"\{\|.*?\|\}", "", wiki, flags=re.S)
    wiki = re.sub(r"\[\[File:[^\]]*\]\]", "", wiki)
    wiki = re.sub(r"\[\[Berkas:[^\]]*\]\]", "", wiki)
    wiki = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", wiki)
    wiki = re.sub(r"\[\[([^\]]+)\]\]", r"\1", wiki)
    wiki = re.sub(r"'''+|''", "", wiki)
    wiki = re.sub(r"^==+([^=]+)==+$", r"\1.", wiki, flags=re.M)
    wiki = re.sub(r"<[^>]+>", "", wiki)
    wiki = re.sub(r"https?://\S+", "", wiki)
    wiki = re.sub(r"[ \t]+", " ", wiki)
    wiki = re.sub(r"\n{3,}", "\n\n", wiki)
    return wiki.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="id")
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--out", default="data/korpus.txt")
    ap.add_argument("--min-char", type=int, default=500)
    ap.add_argument("--hapus-dump", action="store_true",
                    help="Hapus file .bz2 setelah parse selesai (hemat disk)")
    a = ap.parse_args()

    url = DUMP.format(lang=a.lang)
    dump = Path(f"data/wiki-{a.lang}.xml.bz2")
    target = int(a.max_gb * 1e9)
    punya = Path(a.out).stat().st_size if Path(a.out).exists() else 0
    print(f"korpus sekarang: {punya/1e9:.2f} GB, target tambah: {a.max_gb} GB")

    import urllib.request
    def remote_size(url):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Sagi/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                n = r.headers.get("Content-Length")
                return int(n) if n else 0
        except Exception:
            return 0
    perlu = remote_size(url)
    lokal = dump.stat().st_size if dump.exists() else 0
    if not dump.exists() or (perlu and lokal < perlu):
        print(f"download/resume {url} ... (lokal {lokal/1e9:.2f}GB / remote {perlu/1e9:.2f}GB)")
        subprocess.run(["wget", "-c", "--tries=20", "--timeout=60", "-O", str(dump), url], check=True)
    else:
        print(f"dump lengkap ({lokal/1e9:.2f}GB), langsung parse.")

    pola = re.compile(rb"<text[^>]*>(.*?)</text>", re.S)
    buf, n_artikel, n_tulis, ditulis = b"", 0, 0, punya
    with bz2.open(dump, "rb") as f, open(a.out, "a", encoding="utf-8") as o:
        while True:
            pot = f.read(4_000_000)
            if not pot:
                break
            buf += pot
            # parse semua blok <text> lengkap di buffer
            for m in list(pola.finditer(buf)):
                mentah = m.group(1).decode("utf-8", errors="ignore")
                bersih = bersihkan(mentah)
                n_artikel += 1
                if len(bersih) >= a.min_char:
                    potong = bersih[:12000]
                    o.write("\n" + potong + "\n")
                    ditulis += len(potong)
                    n_tulis += 1
                    if n_tulis % 2000 == 0:
                        print(f"  {n_tulis} artikel, +{(ditulis-punya)/1e9:.2f} GB", flush=True)
                    if ditulis - punya >= target:
                        print(f"TARGET tercapai: {n_tulis} artikel")
                        break
            # sisakan ekor tak-lengkap
            tail = buf.rfind(b"</text>")
            buf = buf[tail + 7:] if tail != -1 else buf[-1_000_000:]
            if ditulis - punya >= target:
                break
    print(f"selesai: {n_tulis}/{n_artikel} artikel -> {a.out}")
    if a.hapus_dump:
        try:
            dump.unlink()
            print(f"dump {dump} dihapus.")
        except Exception as e:
            print(f"gagal hapus dump: {e}")


if __name__ == "__main__":
    main()
