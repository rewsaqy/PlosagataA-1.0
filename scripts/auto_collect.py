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
    """Return [(text, label, sumber_url)]. Judul dari listing."""
    return [(t, l, u) for (t, l, u) in _listing_tbh(pages, jeda)]


def _listing_tbh(pages: int, jeda: float = 1.5) -> list[tuple[str, str, str]]:
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


def isi_artikel(url: str, timeout: int = 25) -> str:
    """Ambil isi <article> halaman TurnBackHoax -> teks bersih."""
    html_page = get(url, timeout=timeout)
    m = re.search(r"<article[^>]*>(.*?)</article>", html_page, re.S)
    if not m:
        return ""
    teks = re.sub(r"<script.*?</script>|<style.*?</style>", " ", m.group(1), flags=re.S)
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"\s+", " ", html.unescape(teks)).strip()
    return teks if len(teks) >= 500 else ""


def kumpulkan_tbh_full(max_artikel: int, jeda: float = 2.5,
                       seen_path: str = "data/tbh_seen.txt") -> int:
    """Unduh ISI artikel (bukan judul) -> append korpus.txt. Resume otomatis.

    Sopan: jeda 2.5 dtk/artikel (~35KB/mnt, tak terasa buat main game).
    15.600 artikel ≈ 9-10 jam. Bisa di-stop (Ctrl+C) dan dilanjut kapan saja.
    """
    seen = set()
    p_seen = Path(seen_path)
    if p_seen.exists():
        seen = set(p_seen.read_text(encoding="utf-8").splitlines())
    halaman, tambah, total, kosong_beruntun = 1, 0, 0, 0
    with open(seen_path, "a", encoding="utf-8") as f_seen, \
         open("data/korpus.txt", "a", encoding="utf-8") as o:
        while tambah < max_artikel and kosong_beruntun < 5:
            html_page = ""
            for coba in range(4):
                try:
                    html_page = get(f"{TBH}/articles?page={halaman}")
                    break
                except Exception as e:
                    tunggu = 10 * (coba + 1)
                    print(f"[full] listing {halaman} gagal ({e}), coba lagi {tunggu}d...")
                    time.sleep(tunggu)
            if not html_page:
                print(f"[full] listing {halaman} mati total, loncat halaman.")
                halaman += 1
                continue
            urls = re.findall(r'href="(https://turnbackhoax\.id/articles/\d+-[a-z0-9\-]+)"', html_page)
            urls = list(dict.fromkeys(urls))
            if not urls:
                # Halaman 200 tapi kosong = server ngadat, BUKAN habis.
                kosong_beruntun += 1
                print(f"[full] halaman {halaman} kosong ({kosong_beruntun}x), loncat + jeda 30d.")
                time.sleep(30)
                halaman += 1
                continue
            kosong_beruntun = 0
            for url in urls:
                if tambah >= max_artikel or url in seen:
                    continue
                try:
                    isi = isi_artikel(url)
                except Exception:
                    isi = ""
                seen.add(url)
                f_seen.write(url + "\n")
                f_seen.flush()
                if isi:
                    o.write("\n" + isi[:12000] + "\n")
                    o.flush()
                    tambah += 1
                    total += len(isi)
                    if tambah % 50 == 0:
                        print(f"[full] {tambah} artikel (+{total/1e6:.1f} MB)", flush=True)
                time.sleep(jeda)
            halaman += 1
            time.sleep(jeda)
    print(f"[full] selesai: {tambah} artikel baru (+{total/1e6:.1f} MB)")
    return tambah


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
    ap.add_argument("--tbh-full", type=int, default=0,
                    help="Unduh ISI N artikel TurnBackHoax -> korpus.txt (resume otomatis)")
    ap.add_argument("--rss", type=int, default=1, help="0 = matikan RSS")
    ap.add_argument("--csv", default="data/contoh_dataset.csv")
    ap.add_argument("--no-regen", action="store_true")
    a = ap.parse_args()

    rows, pozn = muat_csv(a.csv)
    print(f"CSV awal: {len(rows)} baris")

    baru = []
    if a.tbh_full > 0:
        kumpulkan_tbh_full(a.tbh_full)
        print("Mode full: hanya isi korpus, selesai.")
        return
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
