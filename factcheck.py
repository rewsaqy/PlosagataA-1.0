"""Fact-check silang ke LLM / search API.

Default: heuristic lokal (tanpa API key) supaya demo lomba tetap jalan.
Kalau ada OPENAI_API_KEY / GEMINI_API_KEY, tinggal aktifkan fungsi di bawah.
Untuk lomba: jelaskan ini sebagai modul RAG — model 12M kasih skor cepat,
LLM kasih verifikasi + sumber.
"""
import os
import re

HOAX_KEYWORDS = [
    "bagikan ke", "sebarkan", "viralkan", "jangan sampai dihapus",
    "info bocoran", "katanya", "konon", "tanpa sumber",
    "klik link ini", "ambil hadiah", "giveaway", "transfer dulu",
    "chip", "microchip", "vaksin menyebabkan", "bumbu dapur menyembuhkan kanker",
    "wajib sebarkan", "jangan diam saja", "sebelum terlambat",
    "terbukti", "100% ampuh", "dijamin", "garansi",
    "obat herbal", "resep dokter terkenal", "rahasia",
    "pemerintah menutupi", "media tidak memberitakan", "disembunyikan",
    "forward ke", "kirim ke", "share ke",
    "dapat uang", "dapat saldo", "dapat pulsa",
    "daftar sekarang", "terbatas", "buruan",
    "menurut profesor", "menurut pakar", "menurut ahli",
    "sumber terpercaya",
    "awas", "waspada", "hati-hati",
    "cukup dengan", "hanya dengan", "tanpa efek samping",
    "nomor wa", "hubungi", "chat admin",
]


def heuristic_signals(text: str) -> dict:
    t = text.lower()
    hits = [k for k in HOAX_KEYWORDS if k in t]
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl = text.count("!") + text.count("?")
    many_caps = caps_ratio > 0.3 and len(text) > 30
    no_source = not any(s in t for s in ["kompas", "detik", "tempo", "antaranews", "cnn", "bbc", "situs resmi", ".go.id"])
    short_url = bool(re.search(r'(?:\b|https?://)(?:bit\.ly|tinyurl(?:\.com)?|s\.id)(?:/\S*)?', t))
    has_phone = bool(re.search(r'(?:(?:\+?62\s?|0)8[0-9\s\-\.]{7,14}[0-9])', text))
    emojis = re.findall(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', text)
    excess_emoji = len(emojis) >= 3
    return {
        "keyword_hits": hits,
        "many_caps": many_caps,
        "excess_punct": excl > 5,
        "no_source": no_source,
        "shortened_url": short_url,
        "short_urls": short_url,
        "phone_number": has_phone,
        "has_phone": has_phone,
        "excess_emoji": excess_emoji,
    }


def local_verdict(text: str, model_result: dict) -> dict:
    sig = heuristic_signals(text)
    skor = 0
    skor += len(sig["keyword_hits"]) * 20
    skor += 15 if sig["many_caps"] else 0
    skor += 10 if sig["excess_punct"] else 0
    skor += 10 if sig["no_source"] else 0
    skor += 15 if sig.get("shortened_url") or sig.get("short_urls") else 0
    skor += 10 if sig.get("phone_number") or sig.get("has_phone") else 0
    if model_result.get("label") == "hoaks":
        skor += 25
    elif model_result.get("label") == "meragukan":
        skor += 10
    skor = min(skor, 100)

    if skor >= 60:
        kesimpulan = "Berpotensi HOAKS — perlu verifikasi ke sumber resmi."
    elif skor >= 30:
        kesimpulan = "MERAGUKAN — cek sumber resmi sebelum menyebarkan."
    else:
        kesimpulan = "Tampak VALID — tetap cek sumber resmi."

    saran = []
    if sig.get("shortened_url") or sig.get("short_urls"):
        saran.append("Waspadai tautan singkat (bit.ly/tinyurl/s.id); berpotensi phising atau penipuan daring.")
    if sig.get("phone_number") or sig.get("has_phone"):
        saran.append("Jangan sembarangan menghubungi nomor telepon/WhatsApp yang tertera untuk menghindari modus rekayasa sosial atau penipuan.")
    if sig.get("keyword_hits"):
        saran.append(f"Ditemukan frasa mencurigakan ({', '.join(sig['keyword_hits'][:3])}); cek kebenarannya di cekfakta.com.")
    if sig.get("many_caps") or sig.get("excess_punct"):
        saran.append("Penggunaan huruf kapital dan tanda baca berlebihan merupakan indikasi pesan provokatif atau clickbait.")
    if sig.get("excess_emoji"):
        saran.append("Penggunaan emoji berlebihan umum ditemukan pada pesan broadcast berantai palsu.")
    if sig.get("no_source"):
        saran.append("Cek situs resmi instansi pemerintah (.go.id) atau media arus utama yang kredibel.")

    saran.append("Lakukan reverse image search untuk poster/gambar.")
    saran.append("Jangan sebarkan sebelum terverifikasi (Saring sebelum Sharing).")

    return {
        "skor_hoaks": skor,
        "kesimpulan": kesimpulan,
        "sinyal": sig,
        "saran": saran,
        "sumber_disarankan": ["turnbackhoax.id", "cekfakta.com", "kominfo.go.id"],
        "catatan": "Ini cek heuristic lokal. Sambungkan ke Gemini/OpenAI API untuk verifikasi LLM + web search.",
    }


def llm_crosscheck(text: str, api: str = "auto") -> dict:
    """Template integrasi LLM. Isi dengan API call kalau sudah punya key."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return {"ok": False, "msg": "Tidak ada API key. Pakai local_verdict untuk demo."}
    # TODO lomba: panggil OpenAI/Gemini di sini, kembalikan ringkasan + sumber URL.
    # Contoh prompt:
    # "Apakah klaim berikut hoaks? Beri verdict, alasan, dan 3 sumber tepercaya: {text}"
    return {"ok": False, "msg": "Belum diimplementasikan — contoh template di factcheck.py"}
