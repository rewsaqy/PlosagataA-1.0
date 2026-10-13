"""Kolektor data otomatis: internet -> database CSV/JSONL proyek.

Sumber:
  1. TurnBackHoax (listing /articles?page=N): judul ada tag [SALAH]/[PENIPUAN]/
     [BENAR], slug URL mengandung verdict -> label otomatis.
  2. RSS berita (Antara/detik/Kompas): judul + deskripsi -> label valid.

Jalankan berkala untuk menggemukkan dataset (ada dedupe otomatis):
  python scripts/auto_collect.py --tbh-pages 3 --rss 1
  python scripts/auto_collect.py --tbh-pages 10 --rss 1 --no-regen

Sopan: jeda antar request, limit halaman, User-Agent jelas.
Hormati situsnya: jangan parallel/agresif, cukup 1x sehari.
"""
import argparse
import csv
import hashlib
import html
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

UA = {"User-Agent": "SagiDatasetBot/1.0 (riset hoaks; kontak: repo PlosagataA-1.0)"}
TBH = "https://turnbackhoax.id"

RSS_FEEDS = [
    "https://www.antaranews.com/rss/terkini.xml",
    "https://news.detik.com/rss",
]
RSS_PER_FEED = 15  # batasi agar label valid tidak membanjiri hoaks


def get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def label_dari_slug(slug: str) -> str | None:
    s = slug.lower()
    if "-salah-" in s or "-penipuan-" in s or "-sesat-" in s or "-hoaks-" in s:
        return "hoaks"
    if "-benar-" in s:
        return "valid"
    return None


def bersihkan_judul(judul: str) -> str:
    judul = html.unescape(judul).strip()
    judul = re.sub(r"^\[(SALAH|BENAR|PENIPUAN|SE SAT|SESAT|KLARIFIKASI)[^\]]*\]\s*", "", judul, flags=re.I)
    judul = re.sub(r"\s+", " ", judul).strip(" .…")
    return judul


def kumpulkan_tbh(pages: int, jeda: float = 1.5) -> list[tuple[str, str, str]]:
    """Return [(text, label, sumber_url)]."""
    hasil, lihat = [], set()
    for p in range(1, pages + 1):
        try:
            html_page = get(f"{TBH}/articles?page={p}")
        except Exception as e:
            print(f"[tbh] halaman {p} gagal: {e}")
            continue
        # pasangan link artikel + judul di kartu
        for m in re.finditer(
                r'href="(https://turnbackhoax\.id/articles/(\d+-[a-z0-9\-]+))"[^>]*>.*?'
                r'(?:##\s*)?(\[[A-Z]+\][^<\n]{5,160})',
                html_page, re.S):
            url, slug, judul_raw = m.group(1), m.group(2), m.group(3)
            label = label_dari_slug(slug)
            judul = bersihkan_judul(judul_raw)
            if not label or len(judul) < 15 or url in lihat:
                continue
            lihat.add(url)
            hasil.append((judul, label, url))
        print(f"[tbh] halaman {p}: {len(hasil)} terkumpul")
        time.sleep(jeda)
    return hasil


def kumpulkan_rss(feeds: list[str]) -> list[tuple[str, str, str]]:
    hasil = []
    for url in feeds:
        try:
            xml = get(url)
            root = ET.fromstring(xml)
        except Exception as e:
            print(f"[rss] {url} gagal: {e}")
            continue
        n = 0
        for item in root.iter("item"):
            if n >= RSS_PER_FEED:
                break
            title = (item.findtext("title") or "").strip()
            desc = re.sub(r"<[^>]+>", "", item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            teks = f"{title}. {desc[:200]}".strip()
            if len(title) >= 20:
                hasil.append((re.sub(r"\s+", " ", html.unescape(teks)), "valid", link or url))
                n += 1
        print(f"[rss] {url}: {n} berita")
        time.sleep(1.0)
    return hasil


def teks_hash(teks: str) -> str:
    norm = re.sub(r"\s+", " ", teks.lower()).strip()
    return hashlib.md5(norm.encode()).hexdigest()


def muat_csv(path: str) -> tuple[list, set]:
    rows,pozn = [], set()
    p = Path(path)
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                pozn.add(teks_hash(r["text"]))
    return rows, pozn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbh-pages", type=int, default=2)
    ap.add_argument("--rss", type=int, default=1, help="0 = matikan RSS")
    ap.add_argument("--csv", default="data/contoh_dataset.csv")
    ap.add_argument("--no-regen", action="store_true")
    a = ap.parse_args()

    rows, pozn = muat_csv(a.csv)
    print(f"CSV awal: {len(rows)} baris")

    baru = []
    if a.tbh_pages > 0:
        baru += kumpulkan_tbh(a.tbh_pages)
    if a.rss:
        baru += kumpulkan_rss(RSS_FEEDS)

    tambah = 0
    with open(a.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for teks, label, sumber in baru:
            if len(teks) < 20 or teks_hash(teks) in pozn:
                continue
            pozn.add(teks_hash(teks))
            w.writerow([teks, label])
            tambah += 1
    print(f"Baru masuk: {tambah} (sumber: turnbackhoax+RSS). Total: {len(rows)+tambah}")

    if not a.no_regen:
        import subprocess
        subprocess.run([sys.executable, "scripts/csv_to_chat.py"], check=True)
        with open("data/train_chat.jsonl", "w", encoding="utf-8") as o:
            for src in ["data/chat_hoaks.jsonl", "data/contoh_chat.jsonl"]:
                p = Path(src)
                if p.exists():
                    o.write(p.read_text(encoding="utf-8"))
        print("train_chat.jsonl diregenerasi.")


if __name__ == "__main__":
    main()
