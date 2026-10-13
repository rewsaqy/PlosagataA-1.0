"""Retrieval sumber internet untuk Sagi (kerja server, bukan model).

Diberi klaim -> cari artikel terkait (Wikipedia ID, gratis tanpa API key)
+ link pencarian situs cek fakta. Gagal jaringan = fallback link manual.
"""
import json
import re
import urllib.parse
import urllib.request

from hoaxcore.tokenizer import STOPWORDS

STOP = STOPWORDS  # alias lama, dipakai kata_kunci()

WIKI_API = "https://id.wikipedia.org/w/api.php"


def kata_kunci(teks: str, maks: int = 6) -> str:
    words = re.findall(r"[a-zA-Z]{4,}", teks.lower())
    unik, lihat = [], set()
    for w in words:
        if w not in STOP and w not in lihat:
            lihat.add(w)
            unik.append(w)
        if len(unik) >= maks:
            break
    return " ".join(unik)


def cari_wikipedia(kueri: str, jumlah: int = 3, timeout: int = 10) -> list[dict]:
    if not kueri.strip():
        return []
    try:
        qs = urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": kueri,
            "srlimit": jumlah, "format": "json",
        })
        req = urllib.request.Request(WIKI_API + "?" + qs,
                                     headers={"User-Agent": "SagiHoaxBot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        hasil = []
        for item in data.get("query", {}).get("search", []):
            judul = item["title"]
            url = "https://id.wikipedia.org/wiki/" + urllib.parse.quote(judul.replace(" ", "_"))
            cuplikan = re.sub(r"<[^>]+>", "", item.get("snippet", ""))[:220]
            hasil.append({"judul": judul, "url": url, "cuplikan": cuplikan})
        return hasil
    except Exception:
        return []


def link_cek_fakta(teks: str) -> list[dict]:
    q = urllib.parse.quote_plus(teks[:120])
    return [
        {"judul": "Cari di TurnBackHoax", "url": f"https://turnbackhoax.id/?s={q}"},
        {"judul": "Cari di CekFakta", "url": f"https://cekfakta.com/?s={q}"},
    ]


def cari_sumber(teks: str) -> dict:
    """Return {'kueri', 'wikipedia': [...], 'cek_fakta': [...]}."""
    kueri = kata_kunci(teks)
    wiki = []
    try:
        from hoaxcore.rag import rag_search
        lokal = rag_search(teks, k=3)
        for h in lokal:
            wiki.append({"judul": h["judul"], "url": "", "cuplikan": h["cuplikan"]})
    except Exception:
        lokal = []
    if not wiki:
        # Kueri 6 kata sering terlalu sempit (AND implisit) -> hasil kosong.
        # Mundur progresif: coba 3 kata pertama, lalu 2 kata pertama.
        toks = kueri.split()
        for n in (3, 2):
            if len(toks) > n:
                wiki = cari_wikipedia(" ".join(toks[:n]))
                if wiki:
                    break
    return {"kueri": kueri,
            "wikipedia": wiki,
            "cek_fakta": link_cek_fakta(teks)}
