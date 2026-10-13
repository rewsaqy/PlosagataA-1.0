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
}

import re as _re

_PATTERN = None


def _get_pattern():
    global _PATTERN
    if _PATTERN is None:
        keys = sorted(SLANG.keys(), key=len, reverse=True)
        _PATTERN = _re.compile(r"\b(" + "|".join(map(_re.escape, keys)) + r")\b",
                               flags=_re.IGNORECASE)
    return _PATTERN


def normalize_slang(text: str) -> str:
    """Ubah slang jadi formal. 'Gaskeun ngab!' -> 'lanjutkan bang!'."""
    if not text:
        return text
    return _get_pattern().sub(lambda m: SLANG[m.group(0).lower()], text)


if __name__ == "__main__":
    for s in ["Gaskeun ngab, spill info giveawaydong wkwk",
              "Sagi, bener gak info maxwin gacor ini? fr!",
              "Aku gabut banget, healing yuk bestie"]:
        print(f"{s}\n  -> {normalize_slang(s)}\n")
