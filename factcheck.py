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
]


def heuristic_signals(text: str) -> dict:
    t = text.lower()
    hits = [k for k in HOAX_KEYWORDS if k in t]
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl = text.count("!") + text.count("?")
    many_caps = caps_ratio > 0.3 and len(text) > 30
    no_source = not any(s in t for s in ["kompas", "detik", "tempo", "antaranews", "cnn", "bbc", "situs resmi", ".go.id"])
    return {
        "keyword_hits": hits,
        "many_caps": many_caps,
        "excess_punct": excl > 5,
        "no_source": no_source,
    }


def local_verdict(text: str, model_result: dict) -> dict:
    sig = heuristic_signals(text)
    skor = 0
    skor += len(sig["keyword_hits"]) * 20
    skor += 15 if sig["many_caps"] else 0
    skor += 10 if sig["excess_punct"] else 0
    skor += 10 if sig["no_source"] else 0
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

    saran = [
        "Cek situs resmi (.go.id) atau media arus utama.",
        "Lakukan reverse image search untuk poster/gambar.",
        "Jangan sebarkan sebelum terverifikasi (Saring sebelum Sharing).",
    ]
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
