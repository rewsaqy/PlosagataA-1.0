"""Topic guard Plosagata — hanya bahas berita, tolak topik lain.

Tinggal di paket plosagata (karya sendiri) agar jalur murni juga terlindungi.
"""
import hashlib as _hl
import re as _re2

# Kata kunci topik berita — di luar ini = off-topic
NEWS_KEYWORDS = [
    # Topik umum & cek fakta
    "berita", "hoaks", "hoax", "fakta", "cek", "valid", "bener", "benar",
    "viral", "klaim", "artikel", "poster", "broadcast", "pesan berantai",
    "sumber", "koran", "media", "wartawan", "pemilu", "presiden", "menteri",
    "bencana", "gempa", "banjir", "vaksin", "covid", "giveaway", "penipuan",
    "lowongan", "bantuan", "bansos", "isu", "kabar", "informasi", "sebar",
    "turnbackhoax", "cekfakta", "mafindo", "kominfo",
    # Tokoh politik
    "jokowi", "prabowo", "anies", "ganjar", "gibran", "megawati", "puan",
    "ahok", "ridwan", "kamil",
    # Lembaga & figur pemerintah
    "pemerintah", "dpr", "mpr", "polisi", "tni", "gubernur", "bupati",
    "walikota", "camat", "lurah",
    # Ekonomi & moneter
    "harga", "rupiah", "dollar", "inflasi", "pajak", "bbm", "sembako",
    "beras", "minyak", "listrik", "tarif", "subsidi", "uang", "bank", "ojk",
    # Kesehatan
    "penyakit", "obat", "rumah sakit", "dokter", "kesehatan", "wabah",
    "pandemi", "demam", "flu", "dbd",
    # Bencana & insiden
    "tsunami", "longsor", "erupsi", "gunung", "kebakaran", "kecelakaan",
    # Kriminal & hukum
    "pembunuhan", "pencurian", "narkoba", "korupsi", "tangkap", "ditahan",
    "tersangka",
    # Isu sosial & aksi
    "demo", "mogok", "protes", "bentrok", "tawuran", "kerusuhan",
    # Pendidikan & seleksi
    "sekolah", "universitas", "ujian", "cpns", "pppk", "seleksi",
    # Penanda klaim / kebenaran
    "benarkah", "apa benar", "betul", "beneran", "masa", "emang",
    "ngga mungkin", "mustahil", "percaya", "serius", "yakin", "bukti", "buktinya",
    # Kata kerja klaim / peristiwa
    "mundur", "meninggal", "tewas", "gratis", "ditangkap", "resmi",
    "dilarang", "dihapus", "naik", "turun",
]
GREETING_ONLY = ["halo", "halo halo", "hallo", "helo", "hai", "hi", "hello",
                 "hey", "hei", "pagi", "siang", "sore", "malam",
                 "assalamu", "assalamualaikum", "tes", "permisi", "permisi",
                 "bang", "kak", "bro", "sis", "kawan", "kawan", "mas",
                 "mbak", "mba", "pak", "bu", "loh", "p", "oy", "woy",
                 "haha", "hehe", "hihi"]

OFFTOPIC_KEYWORDS = [
    # Coding / IT teknis
    "python", "javascript", "html", "css", "coding", "koding", "script",
    "program", "code", "algoritma", "database", "sql", "git",
    # Tugas sekolah / akademik
    "tugas sekolah", "pekerjaan rumah", "rumus", "hitung", "matematika",
    "biologi", "kimia", "fisika",
    # Resep masakan
    "resep", "cara masak", "cara membuat makanan",
    # Curhat & relasi personal
    "curhat", "pacar", "jodoh", "galau", "baper", "mantan",
    # Hiburan
    "rekomendasi lagu", "rekomendasi film", "chord", "lirik",
    "lelucon", "tebak-tebakan", "pantun", "puisi", "cerpen",
    # Obrolan identitas bot kasual
    "kamu siapa", "siapa kamu", "siapa namamu", "lagi apa", "sedang apa",
]

NEWS_QUERY_MARKERS = [
    "apakah", "kenapa", "mengapa", "bagaimana", "gimana", "siapa", "kapan",
    "dimana", "di mana", "ada apa", "apa betul", "apa iya",
]

EVENT_MARKERS = [
    "kejadian", "peristiwa", "terjadi", "korban", "warga", "masyarakat",
    "daerah", "wilayah", "negara", "kota", "jalan", "kantor", "darurat",
    "update", "terkini", "terbaru", "hari ini", "kemarin", "besok",
    "tadi", "barusan", "siang ini", "malam ini", "pagi ini",
    "kebijakan", "aturan", "pengumuman", "putus", "tutup", "buka",
    "batal", "rusak", "ambruk", "amblas", "ledakan", "hancur", "waspada", "siaga",
    "hilang", "ditemukan", "razia", "sidak",
]

# Penanda rujukan ke berita yg sedang dibahas (follow-up multi-turn).
# Dipakai is_follow_up(): pesan yg merujuk topik aktif JANGAN ditolak
# sebagai offtopic walau tanpa keyword berita. Regex word-boundary —
# JANGAN pakai substring ("nya" ada di "hanya"/"tanya").
RUJUKAN = [
    "itu", "tersebut", "tadi", "dia", "mereka", "beliau",
    "ini", "nya", "kan",  # "...kan?" ("bener kan?"), "nya" ("sumbernya mana?")
    "kok", "sih", "dong",  # partikel lanjutan obrolan
]

PERINTAH_LANJUT = [
    "jelaskan", "jelasin", "jabar", "kasih", "berikan", "tunjukkan",
    "tunjukin", "lanjutkan", "lanjut", "detail", "sumbernya", "buktinya",
]

import re as _re

_RUJUK_PAT = _re.compile(r"\b(" + "|".join(map(_re.escape, RUJUKAN)) + r")\b")
_LANJUT_PAT = _re.compile(r"\b(" + "|".join(map(_re.escape, PERINTAH_LANJUT)) + r")")


def is_follow_up(text: str) -> bool:
    """True bila pesan tampak merujuk pembahasan berjalan.

    Syarat: pertanyaan pendek, kata rujukan (itu/tersebut/nya/...),
    atau perintah lanjutan (jelaskan/kasih sumbernya/...).
    Penelepon WAJIB cek sesi punya topik aktif dulu — tanpa topik,
    ini BUKAN follow-up.
    """
    t = (text or "").lower().strip()
    if not t:
        return False
    if "?" in t and len(t) < 200:
        return True
    if _RUJUK_PAT.search(t) or _LANJUT_PAT.search(t):
        return True
    return False


def is_follow_up_safe(text: str) -> bool:
    """Follow-up yang aman: merujuk obrolan TAPI bukan topik terlarang.

    Mencegah bypass "Siapa pacarmu?" — ada "?" tapi mengandung kata
    off-topic ("pacar") → tetap off-topic, bukan follow-up.
    """
    return bool(is_follow_up(text) and not ada_kata_offtopic(text))


WAKE_WORD = "sagi"

SAGI_HELLO = (
    "Halo! Saya Sagi, asisten validator berita Plosagata. "
    "Tempel berita atau klaim yang mau dicek kebenarannya untuk mulai."
)

SAGI_HELLO_VARIANTS = [
    SAGI_HELLO,
    "Halo! Sagi di sini. Ada berita yang meragukan? Tempel saja, saya cek.",
    "Hai! Saya Sagi, penjaga berita biar kamu tidak termakan hoaks. Mana beritanya?",
]

SAGI_REDIRECT_VARIANTS = [
    "Maaf, saya hanya bisa membahas seputar berita dan cek fakta. "
    "Ada berita atau klaim yang mau dicek kebenarannya? Tempel saja di sini.",
    "Itu di luar tugas saya — saya Sagi, khusus validasi berita. "
    "Coba tempel klaim atau berita yang mau dicek.",
    "Saya tolak dengan sopan ya — Sagi hanya melayani cek fakta berita. "
    "Ada berita yang mau diverifikasi?",
]


def sagi_hello(prompt: str = "") -> str:
    """Sapaan bervariasi tapi deterministik (stabil untuk demo)."""
    p = prompt.lower()
    if any(k in p for k in ["siapa", "fungsi", "tugas", "bisa apa", "namamu", "apa aja"]):
        return (
            "Halo! Saya Sagi, asisten AI verifikasi fakta dan pelurus berita hoaks. "
            "Tugas saya membantu kamu memeriksa kebenaran isu viral, broadcast WhatsApp, kabar bencana, "
            "penipuan digital, atau berita apa pun yang kamu ragukan. Cukup tanyakan atau tempel naskah "
            "berita di sini, nanti saya bantu analisis dan luruskan faktanya!"
        )
    n = int(_hl.md5(prompt.encode()).hexdigest(), 16)
    return SAGI_HELLO_VARIANTS[n % len(SAGI_HELLO_VARIANTS)]


def sagi_redirect(prompt: str = "") -> str:
    n = int(_hl.md5(prompt.encode()).hexdigest(), 16)
    return SAGI_REDIRECT_VARIANTS[n % len(SAGI_REDIRECT_VARIANTS)]


# Akar penting yg gagal diurai aturan imbuhan (dicek DULUAN).
PENGECUALIAN = {
    "memberitakan": "berita", "diberitakan": "berita",
    "pemberitaan": "berita",
    "diketahui": "tahu", "mengetahui": "tahu", "diketahuinya": "tahu",
    "menyatakan": "nyata", "dinyatakan": "nyata", "pernyataan": "nyata",
    "dilaporkan": "lapor", "melaporkan": "lapor", "laporan": "lapor",
    "kebakaran": "bakar", "terbakar": "bakar", "membakar": "bakar",
}


def stem_id(kata: str) -> str:
    """Stemmer mini Indonesia: buang imbuhan agar cocok kata utuh.

    Aturan imbuhan BUTA akar ("memberitakan" akan jadi "mberita"),
    jadi akar penting didaftarkan eksplisit di PENGECUALIAN.
    "buang"->"buang" (bukan "uang") tetap dijamin.
    """
    w = kata.lower()
    if w in PENGECUALIAN:
        return PENGECUALIAN[w]
    for pre, ganti in (("menge", ""), ("meny", "s"), ("meng", ""), ("mem", "m"),
                       ("men", "n"), ("penge", ""), ("peny", "s"), ("peng", ""),
                       ("pem", "m"), ("pen", "n"), ("ber", ""), ("ter", ""),
                       ("per", ""), ("pel", ""), ("di", ""), ("ke", ""),
                       ("se", ""), ("pe", "")):
        if w.startswith(pre) and len(w) - len(pre) >= 4:
            w = ganti + w[len(pre):]
            break
    for suf in ("kan", "an", "i", "mu", "ku", "nya", "lah", "kah", "tah", "pun"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[:-len(suf)]
            break
    return w


def _bagi(kata_list: list[str]) -> tuple[set[str], list[str]]:
    """Pisah kata tunggal (stem) vs frasa (substring)."""
    single = {stem_id(k) for k in kata_list if " " not in k}
    phrase = [k.lower() for k in kata_list if " " in k]
    return single, phrase


NEWS_SINGLE, NEWS_PHRASE = _bagi(NEWS_KEYWORDS)
OFFTOPIC_SINGLE, OFFTOPIC_PHRASE = _bagi(OFFTOPIC_KEYWORDS)


def ada_kata_berita(teks: str) -> bool:
    """Cocok kata-per-kata (stem), bukan substring — 'buang' != 'uang'."""
    t = teks.lower()
    if any(p in t for p in NEWS_PHRASE):
        return True
    return any(stem_id(w) in NEWS_SINGLE for w in _re2.findall(r"[a-z]+", t))


def ada_kata_offtopic(teks: str) -> bool:
    t = teks.lower()
    if any(p in t for p in OFFTOPIC_PHRASE):
        return True
    return any(stem_id(w) in OFFTOPIC_SINGLE for w in _re2.findall(r"[a-z]+", t))


def strip_wake(text: str) -> tuple[bool, str]:
    """Deteksi panggilan 'Sagi'. Return (disebut, sisa_pesan)."""
    t = text.lower().strip()
    if WAKE_WORD in t:
        rest = " ".join(t.replace(WAKE_WORD, " ").strip(" ,.!?").split())
        return True, rest
    return False, t


def _is_greeting(t: str) -> bool:
    # cocok utuh atau kata pertama ("halo bang", "pagi kak").
    # TIDAK pakai substring agar "nih" tidak dikira "hi".
    if len(t) > 30:
        return False
    words = t.split()
    return t in GREETING_ONLY or (bool(words) and words[0] in GREETING_ONLY)


def is_news_topic(text: str) -> str:
    """Return 'greeting' | 'news' | 'offtopic'."""
    mentioned, rest = strip_wake(text)
    if mentioned and not rest:
        return "greeting"  # cuma manggil "Sagi!" -> sapa balik
    t = rest if mentioned else text.lower().strip()
    is_intro = any(k in t for k in [
        "kamu siapa", "siapa kamu", "kamu ini siapa", "siapa namamu", "siapa anda",
        "fungsimu apa", "fungsinya apa", "bisa apa", "kamu bisa apa", "bisa bantu apa",
        "apa fungsimu", "apa tugasmu", "tugas kamu apa", "tugasmu apa", "siapa sagi"
    ])
    if is_intro:
        return "greeting"
    if _is_greeting(t) and not ada_kata_berita(t):
        return "greeting"
    if ada_kata_berita(t):
        return "news"
    if ada_kata_offtopic(t):
        return "offtopic"

    # Pertanyaan di bawah 200 karakter: periksa apakah potensi query berita/klaim
    if "?" in text and len(t) < 200:
        if any(m in t for m in NEWS_QUERY_MARKERS) or any(e in t for e in EVENT_MARKERS):
            return "news"
        if len(t.split()) >= 4:
            return "news"
        return "offtopic"

    # Pernyataan pendek: periksa indikasi peristiwa/klaim berita
    if len(t) < 80:
        if any(e in t for e in EVENT_MARKERS):
            return "news"
        return "offtopic"

    # Teks panjang tanpa sinyal berita (mis. tempelan artikel) tetap diproses
    return "news"


def redirect_reply() -> str:
    return (
        "Maaf, saya hanya bisa membahas seputar berita dan cek fakta. "
        "Ada berita atau klaim yang mau dicek kebenarannya? Tempel saja di sini."
    )
