"""Kamus slang modern Indonesia -> bahasa formal.

Tujuan: model kecil (12M/b1) tidak perlu menghafal tiap varian slang.
Varian dinormalisasi SEBELUM masuk model, baik saat training (data)
maupun inferensi (chat). BPE tetap jadi jaring pengaman untuk kata baru.
"""

SLANG = {
    # sapaan / panggilan
    "ngab": "bang", "cuy": "kawan", "coy": "kawan", "bestie": "kawan",
    "maszeh": "mas", "p": "apa",
    # reaksi / tertawa
    "wkwk": "haha", "wkwkwk": "haha", "ngakak": "tertawa", "kocak": "lucu",
    "anjir": "astaga", "anjay": "astaga", "bjir": "astaga", "jir": "astaga",
    "buset": "astaga", "edan": "gila", "gila": "gila",
    # malas / santai
    "gabut": "tidak ada kegiatan", "mager": "malas bergerak",
    "rebahan": "berbaring santai", "santuy": "santai", "healing": "liburan",
    # ajakan / aksi
    "gaskeun": "lanjutkan", "gas": "ayo", "sikat": "ambil",
    "spill": "bocorkan",
    # sifat / penilaian
    "gacor": "bagus sekali", "hoki": "beruntung", "ampas": "jelek",
    "cringe": "memalukan", "sus": "mencurigakan",
    "red flag": "tanda bahaya", "green flag": "tanda baik",
    "baper": "terbawa perasaan", "galau": "gelisah",
    # singkatan internet
    "fomo": "takut ketinggalan", "fyp": "beranda viral",
    "cmiiw": "koreksi jika salah", "tbh": "sejujurnya",
    "fr": "jujur", "no cap": "jujur", "cap": "bohong",
    "imo": "menurut saya", "vibe": "suasana", "slay": "keren",
    # persetujuan / penolakan
    "sabi": "bisa", "gaspol": "lanjut terus", "ogah": "tidak mau",
    "skip": "lewati", "deal": "setuju",
    # uang / judi online (sering muncul di hoaks)
    "maxwin": "kemenangan besar", "wd": "tarik dana", "depo": "setoran",
    # tambahan slang modern
    "kepo": "ingin tahu", "stalking": "mengintai", "julid": "iri dengki",
    "nyinyir": "mencibir", "bacot": "omong kosong", "lebay": "berlebihan",
    "alay": "berlebihan", "norak": "kampungan", "katrok": "kampungan",
    "salfok": "salah fokus", "gagal fokus": "salah fokus",
    "ovt": "berpikir berlebihan", "overthinking": "berpikir berlebihan",
    "insecure": "tidak percaya diri", "gabisa": "tidak bisa",
    "gajelas": "tidak jelas", "gapaham": "tidak paham", "gatahan": "tidak tahan",
    "ngabers": "kawan-kawan", "bacok": "serang", "hajar": "serang",
    "cuan": "untung", "boncos": "rugi", "nombok": "rugi",
    "flexing": "pamer", "pansos": "panjat sosial", "clout": "popularitas",
    "cancel": "boikot", "di-cancel": "dibatalkan",
    "spill the tea": "bocorkan gosip", "tea": "gosip",
    "era": "zaman", "vibes": "suasana", "aesthetic": "indah",
    "staycation": "liburan", "ngadem": "berteduh",
    "mabar": "main bersama", "push rank": "naik peringkat",
    "afk": "tidak aktif", "gg": "bagus", "ez": "mudah",
    "noob": "pemula", "pro": "ahli", "suhu": "ahli",
    # campuran Jawa santai (sering di chat Indonesia)
    "rek": "kawan", "cuk": "kawan", "sam": "kawan",
    "opo": "apa", "piye": "bagaimana", "wes": "sudah",
    "durung": "belum", "ojo": "jangan", "ayo": "ayo",
    "tenan": "sungguh", "temenan": "sungguh", "lho": "lho",
    # campuran Sunda santai
    "atuh": "dong", "teh": "kak", "akang": "kakak",
    "kumaha": "bagaimana", "tos": "sudah", "acan": "belum",
    "ulah": "jangan", "hayu": "ayo",
    # singkatan chat
    "w": "saya", "gw": "saya", "gue": "saya", "gua": "saya",
    "lo": "kamu", "lu": "kamu", "elu": "kamu", "ente": "kamu",
    "org": "orang", "yg": "yang", "dgn": "dengan", "utk": "untuk",
    "bgt": "sekali", "bngt": "sekali", "jg": "juga", "aja": "saja",
    "doang": "saja", "kalo": "kalau", "kl": "kalau",
    "trs": "terus", "tapi": "tetapi", "tp": "tetapi",
    "krn": "karena", "soalnya": "karena", "pdhl": "padahal",
    "emg": "memang", "emang": "memang", "knp": "kenapa",
    "gmn": "bagaimana", "gimana": "bagaimana", "dmn": "di mana",
    "kpn": "kapan", "brp": "berapa", "kmrn": "kemarin",
    "bsk": "besok", "skrg": "sekarang", "skrng": "sekarang",
    "td": "tadi", "udh": "sudah", "udah": "sudah", "blm": "belum",
    "blum": "belum", "jgn": "jangan", "jngn": "jangan",
    "dpt": "dapat", "dpet": "dapat",
    "bs": "bisa", "bisa": "bisa", "gak": "tidak", "ga": "tidak",
    "nggak": "tidak", "engga": "tidak", "kagak": "tidak",
    "tdk": "tidak", "tak": "tidak",
    # hoaks-domain tambahan
    "wajib share": "wajib sebarkan", "wajib sv": "wajib sebarkan",
    "up": "naikkan", "upkan": "naikkan", "viralkan": "viralkan",
    "bantu up": "bantu sebarkan", "bantu share": "bantu sebarkan",
    "amanah": "terpercaya", "real": "asli", "ori": "asli",
    "kw": "palsu", "abal-abal": "palsu", "bodong": "palsu",
    "testimoni": "kesaksian", "testi": "kesaksian",
    "promo": "promosi", "diskon": "potongan harga", "murce": "murah",
    "jastip": "jasa titip", "po": "pesan", "ready": "tersedia",
    "tf": "transfer", "trf": "transfer", "dp": "uang muka",
    "cod": "bayar di tempat", "ovo": "dompet digital", "gopay": "dompet digital",
    # istilah negara / luar negeri
    "russia": "rusia", "ukraine": "ukraina", "america": "amerika",
    "england": "inggris", "china": "tiongkok",
    # istilah validasi
    "hoax": "hoaks", "beneran": "benar", "benerkah": "benarkah",
}

import json as _json
import re as _re
from pathlib import Path as _Path

# Muat kamus slang komprehensif dari JSON
_json_path = _Path(__file__).parent / "kamus_slang.json"
if _json_path.exists():
    try:
        with open(_json_path, encoding="utf-8") as _f:
            _tambahan = _json.load(_f)
            SLANG.update(_tambahan)
    except Exception:
        pass

_PATTERN = None


def _get_pattern():
    global _PATTERN
    if _PATTERN is None:
        # Prioritaskan frasa multi-kata dan token panjang lebih dulu
        valid_keys = [k for k in SLANG if k and _re.match(r"^[a-zA-Z0-9_\-\s]+$", k)]
        valid_keys.sort(key=len, reverse=True)
        _PATTERN = _re.compile(r"\b(" + "|".join(map(_re.escape, valid_keys)) + r")\b",
                               flags=_re.IGNORECASE)
    return _PATTERN


def normalize_slang(text: str) -> str:
    """Ubah slang jadi formal. 'Gaskeun ngab!' -> 'lanjutkan bang!'."""
    if not text:
        return text
    return _get_pattern().sub(lambda m: SLANG.get(m.group(0).lower(), m.group(0)), text)


if __name__ == "__main__":
    for s in ["Gaskeun ngab, spill info giveawaydong wkwk",
              "Sagi, bener gak info maxwin gacor ini? fr!",
              "Aku gabut banget, healing yuk bestie",
              "abis boker gw lgsg otw k ktr, satset bgt nasgornya bro"]:
        print(f"{s}\n  -> {normalize_slang(s)}\n")
