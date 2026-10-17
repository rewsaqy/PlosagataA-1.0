"""Retrieval sumber internet untuk Sagi (kerja server, bukan model).

Diberi klaim -> cari artikel terkait (Wikipedia ID, gratis tanpa API key)
+ link pencarian situs cek fakta. Gagal jaringan = fallback link manual.
"""
import html
import json
import re
import urllib.parse
import urllib.request

from hoaxcore.tokenizer import STOPWORDS

STOP = STOPWORDS  # alias lama, dipakai kata_kunci()

WIKI_API = "https://id.wikipedia.org/w/api.php"


def bersihkan_cuplikan(raw: str) -> str:
    """Bersihkan entitas HTML dan artefak sitasi Wikipedia dari cuplikan."""
    if not raw:
        return ""
    # 1. Unescape HTML entity (&quot; -> ", &#039; -> ', &amp; -> &, dsb)
    s = html.unescape(raw)
    # 2. Hapus tag HTML
    s = re.sub(r"<[^>]+>", "", s)
    # 3. Hapus noise sitasi metadata Wikipedia
    s = re.sub(r"(?i)\baslinya tanggal\s+[\d\-]+[.,]?\s*", "", s)
    s = re.sub(r"(?i)\bdiakses tanggal\s+[\w\s\d]+[.,]?\s*", "", s)
    s = re.sub(r"(?i)\barsip dari versi asli\s+[\w\s\d]+[.,]?\s*", "", s)
    s = re.sub(r"\{\{[^}]+\}\}", "", s)
    s = re.sub(r"\[\d+\]", "", s)
    # 4. Rapikan kutipan dan spasi berlebih
    s = re.sub(r"[\"']\s*[\"']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:240]


def saring_relevan(kueri: str, hasil: list[dict]) -> list[dict]:
    """Buang hasil live yg tak berkaitan dengan kueri.

    Syarat lolos ketat:
    - Judul harus memuat kata kunci kueri, ATAU
    - Cuplikan harus memiliki kecocokan substantif ganda (>= 2 kata kunci unik)
      sehingga artikel acak (seperti seni/tari/kartun) tidak lolos.
    """
    toks = [w for w in kueri.lower().split() if len(w) >= 3 and w not in STOP]
    if not toks:
        return []
    dinilai = []
    for h in hasil:
        judul = (h.get("judul") or "").lower()
        cupl = (h.get("cuplikan") or "").lower()
        j_hits = sum(1 for w in toks if w in judul)
        c_hits = sum(1 for w in toks if w in cupl)
        unik_hits = sum(1 for w in toks if (w in judul or w in cupl))
        
        # Aturan seleksi ketat:
        lolos = False
        if len(toks) == 1:
            if j_hits >= 1 or c_hits >= 2:
                lolos = True
        else:
            # Multi-kata: wajib cocok minimal 2 kata kunci unik
            if j_hits >= 2:
                lolos = True
            elif j_hits >= 1 and unik_hits >= 2 and c_hits >= 1:
                # Periksa agar tidak terjebak demo penerbangan vs demo aksi massa
                if "demo" in toks and "demo" in cupl:
                    if re.search(r"\bdemo\s+penerbangan\b|\bdemo\s+aplikasi\b", cupl):
                        lolos = False
                    else:
                        lolos = True
                else:
                    lolos = True
            elif unik_hits >= 3 and c_hits >= 3:
                lolos = True
            
        if lolos:
            skor = (j_hits * 6) + c_hits + (unik_hits * 3)
            dinilai.append((skor, h))
            
    dinilai.sort(key=lambda p: -p[0])
    return [h for _, h in dinilai]


def kata_kunci(teks: str, maks: int = 6) -> str:
    words = re.findall(r"[a-zA-Z]{3,}", teks.lower())
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
            "srlimit": jumlah * 2, "format": "json",
        })
        req = urllib.request.Request(WIKI_API + "?" + qs,
                                     headers={"User-Agent": "SagiHoaxBot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        hasil = []
        for item in data.get("query", {}).get("search", []):
            judul = item["title"]
            url = "https://id.wikipedia.org/wiki/" + urllib.parse.quote(judul.replace(" ", "_"))
            cuplikan = bersihkan_cuplikan(item.get("snippet", ""))
            if cuplikan:
                hasil.append({"judul": judul, "url": url, "cuplikan": cuplikan})
        return saring_relevan(kueri, hasil)[:jumlah]
    except Exception:
        return []


def link_cek_fakta(teks: str) -> list[dict]:
    q = urllib.parse.quote_plus(teks[:120])
    return [
        {"judul": "Cari di TurnBackHoax", "url": f"https://turnbackhoax.id/?s={q}"},
        {"judul": "Cari di CekFakta", "url": f"https://cekfakta.com/?s={q}"},
    ]


def cari_web_live(kueri: str, jumlah: int = 3, timeout: int = 4) -> list[dict]:
    """Pencarian live web publik / real-time internet (DuckDuckGo instant answer / API)."""
    if not kueri.strip():
        return []
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": kueri,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        req = urllib.request.Request(url, headers={"User-Agent": "SagiBrowserBot/2.0 (OTN-2026)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="ignore"))

        results = []
        if data.get("AbstractText"):
            results.append({
                "judul": data.get("Heading") or kueri.title(),
                "url": data.get("AbstractURL") or f"https://duckduckgo.com/?q={urllib.parse.quote(kueri)}",
                "cuplikan": data.get("AbstractText")[:240],
                "sumber": data.get("AbstractSource") or "Live Web",
            })
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict) and item.get("Text") and item.get("FirstURL"):
                results.append({
                    "judul": item.get("Text").split(" - ")[0][:60],
                    "url": item.get("FirstURL"),
                    "cuplikan": item.get("Text")[:240],
                    "sumber": "Web Search",
                })
            if len(results) >= jumlah:
                break
        return results[:jumlah]
    except Exception:
        return []


def cari_sumber(teks: str) -> dict:
    """Return {'kueri', 'wikipedia': [...], 'web': [...], 'cek_fakta': [...]}."""
    from plosagata.slang import normalize_slang
    clean_teks = normalize_slang(teks)
    kueri = kata_kunci(clean_teks)
    wiki = []
    try:
        from hoaxcore.rag import rag_search
        lokal = rag_search(clean_teks, k=3)
        for h in lokal:
            wiki.append({
                "judul": h["judul"],
                "url": "",
                "cuplikan": bersihkan_cuplikan(h.get("cuplikan", ""))
            })
    except Exception:
        lokal = []
    if not wiki and kueri:
        # Coba kueri penuh dulu
        wiki = cari_wikipedia(kueri)
        if not wiki:
            # Mundur progresif bila kueri terlalu spesifik
            toks = kueri.split()
            for n in (3, 2):
                if len(toks) > n:
                    wiki = cari_wikipedia(" ".join(toks[:n]))
                    if wiki:
                        break

    # Live Web Search
    web_res = []
    if kueri:
        try:
            web_res = cari_web_live(kueri, jumlah=2)
        except Exception:
            web_res = []

    return {
        "kueri": kueri,
        "wikipedia": wiki,
        "web": web_res,
        "cek_fakta": link_cek_fakta(teks),
    }
