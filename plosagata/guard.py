"""Topic guard Plosagata — hanya bahas berita, tolak topik lain.

Dipindah dari chat.py (karya sendiri) agar jalur murni juga terlindungi.
"""

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
                 "mbak", "mba", "pak", "bu", "loh", "p", "oy", "woy"]

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


def _is_greeting(t: str) -> bool:
    # cocok utuh atau kata pertama ("halo bang", "pagi kak").
    # TIDAK pakai substring agar "nih" tidak dikira "hi".
    if len(t) > 30:
        return False
    words = t.split()
    return t in GREETING_ONLY or (bool(words) and words[0] in GREETING_ONLY)


def is_news_topic(text: str) -> str:
    """Return 'greeting' | 'news' | 'offtopic'."""
    t = text.lower().strip()
    if _is_greeting(t) and not any(k in t for k in NEWS_KEYWORDS):
        return "greeting"
    if any(k in t for k in NEWS_KEYWORDS):
        return "news"
    if any(o in t for o in OFFTOPIC_KEYWORDS):
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
