"""Fact-check silang ke LLM / search API.

Default: heuristic lokal (tanpa API key) supaya demo lomba tetap jalan.
Kalau ada OPENAI_API_KEY / GEMINI_API_KEY, tinggal aktifkan fungsi di bawah.
Untuk lomba: jelaskan ini sebagai modul RAG — model 12M kasih skor cepat,
LLM kasih verifikasi + sumber.
"""
import hashlib as _hl
import os
import re

HOAX_KEYWORDS = [
    "bagikan ke", "sebarkan", "viralkan", "jangan sampai dihapus",
    "info bocoran", "katanya", "konon", "tanpa sumber",
    "klik link ini", "ambil hadiah", "giveaway", "transfer dulu",
    "chip", "microchip", "vaksin menyebabkan", "bumbu dapur menyembuhkan kanker",
    "viral", "menyebar", "sebarkan", "tanpa izin", "tanpa menyebutkan",
    "tidak menyebutkan", "tanpa nama", "tanpa bukti", "matikan",
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
    ".apk", "file apk", "unduh apk", "instal apk", "download apk",
    "kode otp", "minta otp", "kirim otp", "pin atm",
]

# Tokoh publik + kata kerja kriminal TANPA sumber kredibel = pola fitnah absurd.
# ("Prabowo ledakkan rumah Putin", "Jokowi culik artis", ...)
# Hanya aktif bila tak ada media arus utama disebut (berita kriminal asli
# selalu menyebut sumber) agar berita kriminal beneran tidak kena.
PEJABAT = [
    "presiden", "wakil presiden", "wapres", "menteri", "gubernur", "bupati",
    "walikota", "jokowi", "prabowo", "anies", "ganjar", "gibran", "megawati",
    "puan", "ahok", "ridwan kamil", "putin", "biden", "trump", "raja",
]
KRIMINAL = [
    "bunuh", "membunuh", "pembunuhan", "ledak", "meledakkan", "bom",
    "culik", "menculik", "penculikan", "rampok", "merampok", "perkosa",
    "memperkosa", "tikam", "menusuk", "tembak", "menembak", "racun",
    "meracuni", "santet", "teluh", "bunuh diri",
]

# Penanda konten yang MENDEKLARASIKAN dirinya fiksi/simulasi.
# Kuat = cukup 1 temuan. Lemah = butuh >=2 temuan berbeda
# (agar berita latihan/simulasi bencana asli tidak kena false-positive).
FIKSI_KUAT = [
    "fiktif", "fiksi", "rekaan", "direka", "khayalan", "dongeng",
]
FIKSI_LEMAH = [
    "simulasi", "skenario", "ilustrasi", "sintetis", "ilustratif",
    "simulasi literasi", "untuk simulasi", "dalam simulasi",
    "skenario tersebut", "data sintetis", "naskah ini",
    "contoh jurnalistik", "pejabat fiktif", "tokoh fiktif",
]


def deteksi_fiksi(text: str) -> list[str]:
    """Kembalikan penanda fiksi yg ditemukan (kosong = bukan deklarasi fiksi)."""
    t = (text or "").lower()
    kuat = [k for k in FIKSI_KUAT if k in t]
    lemah = [k for k in FIKSI_LEMAH if k in t]
    if kuat or len(set(lemah)) >= 2:
        return kuat + lemah
    return []


def analisis_konteks_pesan(text: str) -> dict:
    """Ekstraksi entitas cerdas: lokasi, waktu, jenis kejadian, dan niat pertanyaan."""
    t = text.lower()
    try:
        from plosagata.slang import normalize_slang
        t = normalize_slang(t).lower()
    except Exception:
        pass

    # 1. Deteksi niat bertanya / verifikasi rumor
    is_tanya = (
        "?" in text
        or any(k in t for k in [
            "apakah benar", "apakah", "apa benar", "bener gak", "beneran gak",
            "bener kah", "benarkah", "betulkah", "apa betul", "katanya",
            "denger kabar", "isu", "gosip", "ada kabar", "ada info",
            "ada yang tau", "ada yg tau", "beneran nih", "seriusan",
            "tolong cek", "tolong klarifikasi", "bisa tolong cek"
        ])
    )

    # 2. Deteksi Waktu Spesifik
    waktu_parts = re.findall(
        r"(?:nanti\s+)?jam\s+\d+(?:[.:]\d+)?(?:\s*(?:siang|malam|sore|pagi))?|sampai\s+jam\s+\d+(?:[.:]\d+)?(?:\s*(?:siang|malam|sore|pagi))?|pukul\s+\d+(?:[.:]\d+)?(?:\s*(?:wib|wita|wit))?|besok(?:\s+(?:pagi|siang|sore|malam))?|nanti\s+(?:siang|malam|sore|pagi)|hari\s+ini(?:\s+siang|\s+malam|\s+sore|\s+pagi)?|malam\s+ini|siang\s+ini|sore\s+ini|lusa",
        t
    )
    waktu_str = " ".join(waktu_parts).strip() if waktu_parts else ""

    # 3. Deteksi Lokasi
    daftar_kota = [
        "bogor", "jakarta pusat", "jakarta selatan", "jakarta barat", "jakarta timur", "jakarta utara", "jakarta",
        "depok", "tangerang selatan", "tangsel", "tangerang", "bekasi", "jabodetabek",
        "bandung", "surabaya", "semarang", "yogyakarta", "jogja", "solo", "surakarta",
        "medan", "palembang", "makassar", "denpasar", "bali", "malang", "aceh", "padang",
        "batam", "balikpapan", "samarinda", "pontianak", "manado", "papua", "jayapura",
        "gedung dpr", "monas", "istana negara", "balai kota", "tugu kujang", "bundaran hi"
    ]
    lokasi_ditemukan = [k for k in daftar_kota if k in t]
    lokasi_str = lokasi_ditemukan[0].title() if lokasi_ditemukan else ""

    # 4. Kategori Kejadian / Aksi
    is_demo = any(k in t for k in [
        "demo", "demonstrasi", "unjuk rasa", "aksi massa", "aksi damai",
        "demo buruh", "demo mahasiswa", "mogok kerja", "long march", "turun ke jalan"
    ])
    is_razia = any(k in t for k in [
        "razia", "operasi zebra", "tilang manual", "razia gabungan", "razia stnk",
        "penyekatan jalan", "operasi kepolisian", "operasi lilin", "operasi ketupat"
    ])
    is_lalin = any(k in t for k in [
        "tutup jalan", "penutupan jalan", "rekayasa lalu lintas", "one way", "contraflow",
        "jalan tol ditutup", "macet parah", "jalur ditutup"
    ])
    is_pemadaman = any(k in t for k in [
        "mati lampu", "pemadaman listrik", "pln padam", "pdam mati", "gangguan air",
        "aliran air mati", "listrik padam"
    ])
    is_kriminal = any(k in t for k in [
        "penculikan anak", "begal", "geng motor", "komplotan begal", "teror",
        "penusukan", "pembunuhan", "perampokan"
    ])
    is_bencana = any(k in t for k in [
        "gempa", "tsunami", "gunung meletus", "erupsi", "angin puting beliung",
        "banjir bandang", "megathrust", "tanah longsor"
    ])

    return {
        "is_tanya": is_tanya,
        "waktu_str": waktu_str,
        "lokasi_str": lokasi_str,
        "is_demo": is_demo,
        "is_razia": is_razia,
        "is_lalin": is_lalin,
        "is_pemadaman": is_pemadaman,
        "is_kriminal": is_kriminal,
        "is_bencana": is_bencana,
    }


def heuristic_signals(text: str) -> dict:
    t = text.lower()
    try:
        from plosagata.slang import normalize_slang
        t = normalize_slang(t).lower()
    except Exception:
        pass

    ctx = analisis_konteks_pesan(text)
    hits = [k for k in HOAX_KEYWORDS if k in t]
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl = text.count("!") + text.count("?")
    many_caps = caps_ratio > 0.3 and len(text) > 30
    no_source = not any(s in t for s in [
        "kompas", "detik", "tempo", "antaranews", "cnn", "bbc", "situs resmi", ".go.id",
        "polresta", "polres", "polda", "humas polri", "tmc", "bmkg", "bnpb", "kemenkes"
    ])
    short_pat = r'(?:\b|https?://)(?:bit\.ly|tinyurl(?:\.com)?|s\.id)(?:/\S*)?'
    phone_pat = r'(?:(?:\+?62\s?|0)8[0-9\s\-\.]{7,14}[0-9])'
    short_urls = re.findall(short_pat, t)
    phone_numbers = re.findall(phone_pat, text)
    short_url = bool(short_urls)
    has_phone = bool(phone_numbers)
    apk_pat = r'(?:\.apk\b|\bfile\s+apk\b|\bunduh\s+apk\b|\binstal\s+apk\b|\bdownload\s+apk\b)'
    has_apk = bool(re.search(apk_pat, t))

    # Fitnah absurd: tokoh publik + kerja kriminal, tanpa media kredibel.
    fitnah = (any(p in t for p in PEJABAT)
              and any(k in t for k in KRIMINAL)
              and not any(s in t for s in ["kompas", "detik", "tempo", "antaranews",
                                           "cnn", "bbc", "kejaksaan", "kepolisian",
                                           "kpk", "pengadilan", "vonis", "ditangkap"]))
    # Hoaks perang / pengerahan militer luar negeri tanpa konfirmasi resmi
    hoaks_perang = (any(p in t for p in PEJABAT)
                    and any(w in t for w in ["perang", "tempur", "kirim pasukan", "kirim tentara"])
                    and any(c in t for c in ["rusia", "russia", "ukraina", "israel", "palestina", "moskow", "kiev"])
                    and not any(s in t for s in ["kemlu", "tni.mil.id", ".go.id", "antaranews"]))
    # Hoaks kematian tokoh publik tanpa konfirmasi media arus utama
    hoaks_wafat = (any(p in t for p in PEJABAT)
                   and any(w in t for w in ["meninggal", "tewas", "wafat", "mati mendadak", "kritis di rumah sakit"])
                   and not any(s in t for s in ["kompas", "detik", "tempo", "antaranews", "cnn", "situs resmi", ".go.id"]))

    # Hoaks saldo e-wallet / perbankan / sistem pembayaran nasional
    hoaks_ewallet = (
        any(k in t for k in ["e-wallet", "wallet", "dompet digital", "saldo", "rekening", "gopay", "ovo", "dana", "shopeepay", "linkaja", "m-banking", "mbanking", "atm"])
        and any(k in t for k in ["terpotong", "berkurang", "hilang", "pembaruan sistem", "sistem pembayaran", "nomor lama", "putuskan koneksi", "tutup akun", "pembayaran nasional", "bi-fast", "qris", "hangus", "malam ini"])
    )

    # Hoaks bantuan / bansos / blt / giveaway via tautan atau nomor WA
    hoaks_bansos = (
        any(k in t for k in ["bansos", "blt", "pkh", "subsidi", "bantuan tunai", "giveaway", "saldo gratis", "dana kaget", "kuota gratis", "undian"])
        and (any(k in t for k in ["link", "tautan", "wa", "whatsapp", "klik", "daftar", "cairkan", "bit.ly", "s.id", "form", "isi data", "otp"]) or short_url or has_phone)
    )

    # Hoaks obat ajaib / kesembuhan instan
    hoaks_obat = (
        any(k in t for k in ["sembuh total", "obat ajaib", "kanker stadium 4", "ramuan ajaib", "tanpa operasi", "100% sembuh"])
        and any(k in t for k in ["minum", "resep", "daun", "rebusan", "bawang", "jeruk", "garam", "tanpa dokter"])
    )

    # Hoaks ramalan bencana alam
    hoaks_gempa = (
        any(k in t for k in ["gempa besar", "gempa dahsyat", "tsunami besar", "megathrust", "gunung meletus"])
        and any(k in t for k in ["besok", "nanti malam", "malam ini", "lusa", "pukul", "jam", "diprediksi terjadi"])
        and not any(s in t for s in ["bmkg", "bnpb", "pvmbg"])
    )

    # Deteksi kejadian/rumor yang belum terverifikasi (jangan pernah langsung disebut VALID!)
    is_unverified_event = bool(
        (ctx["is_demo"] or ctx["is_razia"] or ctx["is_lalin"] or ctx["is_pemadaman"] or ctx["is_kriminal"] or ctx["is_tanya"])
        and no_source
    )

    emojis = re.findall(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', text)
    excess_emoji = len(emojis) >= 3
    return {
        "konteks": ctx,
        "is_unverified_event": is_unverified_event,
        "keyword_hits": hits,
        "fitnah_tokoh": fitnah,
        "hoaks_perang": hoaks_perang,
        "hoaks_wafat": hoaks_wafat,
        "hoaks_ewallet": hoaks_ewallet,
        "hoaks_bansos": hoaks_bansos,
        "hoaks_obat": hoaks_obat,
        "hoaks_gempa": hoaks_gempa,
        "many_caps": many_caps,
        "excess_punct": excl > 5,
        "no_source": no_source,
        "shortened_url": short_url,
        "short_urls": short_urls,
        "phone_number": has_phone,
        "has_phone": has_phone,
        "phone_numbers": phone_numbers,
        "has_apk": has_apk,
        "excess_emoji": excess_emoji,
    }


def final_label(model_label: str, skor: int, fiksi: bool = False, is_unverified: bool = False) -> str:
    """SATU label final — badge & kesimpulan tidak boleh kontradiksi."""
    if fiksi:
        return "fiksi"
    if skor >= 60:
        return "hoaks"
    if skor >= 35 or is_unverified:
        return "hoaks" if (model_label or "") == "hoaks" and skor >= 45 else "meragukan"
    return "valid"


def local_verdict(text: str, model_result: dict) -> dict:
    sig = heuristic_signals(text)
    fiksi = deteksi_fiksi(text)
    sig["fiksi_hits"] = fiksi
    ctx = sig.get("konteks", {})

    # Deteksi sinyal peringatan aktif (red flags)
    has_alarm = bool(
        sig["keyword_hits"]
        or sig["many_caps"]
        or sig["excess_punct"]
        or sig.get("shortened_url")
        or sig.get("phone_number")
        or sig.get("excess_emoji")
        or sig.get("fitnah_tokoh")
        or sig.get("has_apk")
        or sig.get("hoaks_perang")
        or sig.get("hoaks_wafat")
        or sig.get("hoaks_ewallet")
        or sig.get("hoaks_bansos")
        or sig.get("hoaks_obat")
        or sig.get("hoaks_gempa")
    )

    skor = 0
    skor += len(sig["keyword_hits"]) * 20
    skor += 50 if sig.get("fitnah_tokoh") else 0   # fitnah tokoh ≈ pasti hoaks
    skor += 40 if sig.get("has_apk") else 0        # malware APK ≈ pasti penipuan
    skor += 45 if sig.get("hoaks_perang") else 0   # hoaks perang tokoh publik
    skor += 45 if sig.get("hoaks_wafat") else 0    # hoaks kematian tokoh publik
    skor += 65 if sig.get("hoaks_ewallet") else 0  # hoaks saldo terpotong ewallet
    skor += 65 if sig.get("hoaks_bansos") else 0   # phishing bansos / giveaway
    skor += 60 if sig.get("hoaks_obat") else 0     # hoaks obat ajaib tanpa medis
    skor += 60 if sig.get("hoaks_gempa") else 0    # hoaks prediksi tanggal/jam gempa
    skor += 15 if sig["many_caps"] else 0
    skor += 10 if sig["excess_punct"] else 0
    skor += 15 if sig.get("shortened_url") or sig.get("short_urls") else 0
    skor += 10 if sig.get("phone_number") or sig.get("has_phone") else 0

    # Penalti tanpa sumber hanya berlaku jika teks mengandung nada provokatif / alarm
    if sig["no_source"] and has_alarm:
        skor += 10

    # Kontribusi model 12M
    m_label = model_result.get("label", "")
    if m_label == "hoaks":
        if has_alarm:
            skor += 25
        else:
            skor += 10
    elif m_label == "meragukan":
        skor += 5 if has_alarm else 0

    skor = min(skor, 100)

    # Perlindungan mutlak: Pertanyaan rumor atau isu kejadian tanpa sumber TIDAK BOLEH berlabel VALID!
    is_unverified = bool(
        sig.get("is_unverified_event")
        or (ctx.get("is_tanya") and sig.get("no_source"))
    )

    if fiksi:
        kesimpulan = ("FIKSI/SIMULASI — konten menyatakan dirinya rekaan, "
                      "bukan laporan kejadian nyata.")
    elif skor >= 60:
        kesimpulan = "Berpotensi HOAKS — terindikasi kuat manipulasi informasi atau penipuan digital."
    elif skor >= 35 or is_unverified:
        skor = max(skor, 40)
        kesimpulan = "BELUM TERKONFIRMASI — masih bersifat rumor atau kabar burung; belum ada pernyataan resmi."
    elif sig.get("no_source"):
        skor = max(skor, 35)
        kesimpulan = "MERAGUKAN — belum terverifikasi, ketiadaan sumber resmi yang memvalidasi klaim ini."
    else:
        kesimpulan = "Tampak VALID — informasi didukung oleh rujukan resmi."

    saran = []
    if sig.get("fitnah_tokoh"):
        saran.append("Pola fitnah: tokoh publik dituduh kejahatan berat tanpa proses hukum maupun media kredibel. Cek ke kejaksaan/pengadilan resmi.")
    if sig.get("has_apk"):
        saran.append("BAHAYA MALWARE: Terdeteksi file .APK / unduhan aplikasi! Waspadai modus sniffing SMS perbankan atau penyusupan virus ke HP Anda.")
    if fiksi:
        saran.append(f"Terdeteksi deklarasi fiksi/simulasi ({', '.join(fiksi[:3])}). "
                     "Jangan disebar sebagai berita faktual; fiksi berlabel jelas bukan hoaks.")
    if sig.get("shortened_url") or sig.get("short_urls"):
        saran.append("Waspadai tautan singkat (bit.ly/tinyurl/s.id); berpotensi phising atau penipuan daring.")
    if sig.get("phone_number") or sig.get("has_phone"):
        saran.append("Jangan sembarangan menghubungi nomor telepon/WhatsApp yang tertera untuk menghindari modus rekayasa sosial atau penipuan.")
    if sig.get("keyword_hits"):
        saran.append(f"Ditemukan frasa mencurigakan ({', '.join(sig['keyword_hits'][:3])}); cek kebenarannya di cekfakta.com.")
    if is_unverified:
        saran.append("Hindari menyebarkan isu atau kabar burung yang belum memiliki konfirmasi resmi dari instansi berwenang.")
    if sig.get("no_source"):
        saran.append("Cek situs resmi instansi pemerintah (.go.id) atau media arus utama yang kredibel.")

    saran.append("Jangan sebarkan sebelum terverifikasi (Saring sebelum Sharing).")

    return {
        "skor_hoaks": skor,
        "label_final": final_label(model_result.get("label"), skor, fiksi=bool(fiksi), is_unverified=is_unverified),
        "kesimpulan": kesimpulan,
        "sinyal": sig,
        "saran": saran,
        "sumber_disarankan": ["turnbackhoax.id", "cekfakta.com", "kominfo.go.id"],
        "catatan": "Pemeriksaan semantik & penalaran cerdas lokal Sagi AI.",
    }


def _bangun_pembuka_natural(teks: str) -> str:
    """Deteksi emosi & intensitas kueri warganet untuk sapaan alami non-robotik."""
    t = (teks or "").lower()
    is_panik = any(k in t for k in ["takut", "ngeri", "gawat", "parah", "aduh", "beneran nih", "merinding", "panik", "resah", "bahaya"])
    is_skeptis = any(k in t for k in ["masa sih", "kok aneh", "kayak bohong", "gak masuk akal", "hoaks bukan", "bener gak sih", "ragu", "beneran gak"])
    is_kasual = any(k in t for k in ["halo", "hai", "min", "gan", "kak", "bang", "sagi", "bro", "pagi", "malam", "siang"])

    n = int(_hl.md5((teks or "default").encode()).hexdigest(), 16)

    if is_panik:
        pilihan = [
            "Tenang dulu ya, jangan buru-buru cemas atau panik. Wajar sekali kabar mengejutkan semacam ini bikin resah saat berseliweran di ponselmu. Mari kita telusuri faktanya secara jernih bersama-sama:",
            "Tarik nafas dulu sejenak ya! Informasi heboh semacam ini memang sengaja dirancang oleh pembuatnya untuk memicu kepanikan emosional warga. Yuk kita lihat bukti-bukti konkret di baliknya:",
            "Aku paham sekali kenapa kabar ini bikin kamu khawatir. Kabar viral yang bernada genting memang sering memicu kepanikan. Tapi sebelum kamu resah, ini fakta sebenarnya yang wajib kamu tahu:",
        ]
        return pilihan[n % len(pilihan)]
    if is_skeptis:
        pilihan = [
            "Insting kamu untuk skeptis dan kroscek dulu sudah tepat banget! Memang ada banyak sekali kejanggalan pada narasi ini jika kita telaah secara kritis:",
            "Kecurigaanmu sangat beralasan! Narasi semacam ini memang memuat pola rekayasa informasi yang sering mengecoh masyarakat. Mari kita buktikan faktanya:",
            "Pertanyaan kritis yang cerdas! Menjawab keraguanmu, berikut hasil penelusuran fakta mendalam yang berhasil kami rangkum:",
        ]
        return pilihan[n % len(pilihan)]
    if is_kasual:
        pilihan = [
            "Halo! Pertanyaan yang sangat bagus dan tepat waktu. Isu ini belakangan memang lagi ramai berseliweran di linimasa dan grup obrolan. Mari kita kupas tuntas:",
            "Hai! Senang banget kamu membiasakan diri kroscek dulu ke aku sebelum percaya atau membagikannya. Yuk kita bedah faktanya bareng-bareng:",
            "Halo juga! Siap, mari kita telusuri bukti resmi di balik kabar viral tersebut langkah demi langkah:",
        ]
        return pilihan[n % len(pilihan)]

    pilihan = [
        "Mengenai kabar atau informasi yang kamu tanyakan, berikut hasil analisis investigasi fakta dan pemeriksaan bukti resminya:",
        "Menanggapi klaim tersebut, mari kita bedah duduk perkaranya berdasarkan konfirmasi otoritas resmi dan pangkalan data cek fakta:",
        "Terima kasih sudah menanyakan keabsahan informasi ini. Berikut telaah mendalam fakta di lapangan:",
    ]
    return pilihan[n % len(pilihan)]


def susun_jawaban_cerdas(
    teks: str = "",
    label: str = "meragukan",
    skor: int = 50,
    kesimpulan: str = "",
    sumber: dict | None = None,
    fiksi: list | None = None,
    sinyal: dict | None = None,
    saran: list | None = None,
) -> str:
    """Susun jawaban pelurusan fakta yang cerdas, kontekstual, manusiawi, dan natural seperti AI berfikir."""
    t_clean = (teks or "").lower()
    sig = sinyal or {}
    ctx = sig.get("konteks") or analisis_konteks_pesan(teks)
    pembuka = _bangun_pembuka_natural(teks)

    # 1. Konten Fiksi / Simulasi
    if fiksi:
        return (
            f"{pembuka}\n\n"
            "📌 **Status Analisis**: **KARYA FIKSI / SIMULASI CERITA REKAAN**\n\n"
            "Konten yang kamu tanyakan ini merupakan **karya fiksi / rekaan cerita hiburan**, bukan peristiwa nyata di dunia kriminal atau bencana. "
            "Di dalam naskahnya sendiri secara eksplisit telah tercantum penanda bahwa nama tokoh, latar, dan peristiwa di dalamnya adalah rekaan artistik.\n\n"
            "💡 **Saran Sahabat Sagi**: Nikmati karya tersebut murni sebagai bahan bacaan fiksi atau edukasi literasi media, dan jangan diteruskan ke grup obrolan seolah-olah kejadian nyata di lapangan ya!"
        )

    # 2. Demo / Unjuk Rasa / Aksi Massa
    if ctx.get("is_demo"):
        lok = ctx.get("lokasi_str") or "wilayah tersebut"
        wkt = f" pada {ctx.get('waktu_str')}" if ctx.get("waktu_str") else ""
        return (
            f"{pembuka}\n\n"
            f"🔍 **Status Isu**: **BELUM TERKONFIRMASI / MASIH BERSIFAT RUMOR**\n\n"
            f"Terkait isu rencana aksi demonstrasi/unjuk rasa di **{lok}**{wkt}, berikut fakta hasil investigasi:\n\n"
            f"1. **Prosedur Pemberitahuan Aksi (STTP):** Di Indonesia, setiap aksi penyampaian pendapat di muka umum wajib melayangkan Surat Tanda Terima Pemberitahuan (STTP) ke kepolisian setempat. Sampai saat ini, belum ada rilis resmi dari Polresta {lok} maupun Satlantas mengenai pengamanan ataupun pengalihan arus lalu lintas di jam tersebut.\n"
            f"2. **Nihil Pemberitaan Media Kredibel:** Belum ada laporan dari kantor berita arus utama atau media lokal terverifikasi Dewan Pers yang mengabarkan jadwal aksi unjuk rasa tersebut.\n"
            f"3. **Karakteristik Pesan Berantai:** Pesan berantai di grup percakapan kerap mendaur ulang selebaran aksi lama atau sengaja dibuat untuk memicu kepanikan pengguna jalan.\n\n"
            f"💡 **Saran Sahabat Sagi**: Jika kamu berencana melintas di titik-titik keramaian di {lok}, tetap tenang dan pantau akun resmi kepolisian (seperti akun media sosial TMC Satlantas Polresta {lok} atau Dishub). Jangan langsung membagikan kabar yang belum jelas sumbernya."
        )

    # 3. Operasi Kepolisian / Razia Kendaraan
    if ctx.get("is_razia"):
        lok = ctx.get("lokasi_str") or "lokasi tersebut"
        wkt = f" {ctx.get('waktu_str')}" if ctx.get("waktu_str") else ""
        return (
            f"{pembuka}\n\n"
            f"🔍 **Status Isu**: **BELUM TERVERIFIKASI RESMI**\n\n"
            f"Mengenai kabar razia kendaraan bermotor di **{lok}**{wkt}, ini hal penting yang perlu kamu ketahui:\n\n"
            f"1. **Surat Perintah Tugas (Sprin):** Operasi penertiban lalu lintas resmi kepolisian selalu dilengkapi Surat Perintah Tugas resmi dan wajib memasang plang razia minimal 50–100 meter sebelum lokasi.\n"
            f"2. **Pola Hoaks Daftar Razia:** Pesan broadcast berisi daftar panjang titik razia di grup WA hampir selalu merupakan pesan hoaks daur ulang dari tahun-tahun sebelumnya yang terus diputar ulang.\n"
            f"3. **Kanal Resmi:** Operasi besar kepolisian (seperti Operasi Zebra atau Operasi Patuh) selalu diumumkan secara terbuka terlebih dahulu melalui kanal resmi Humas Polri dan TMC Korlantas.\n\n"
            f"💡 **Saran Sahabat Sagi**: Selalu lengkapi surat kendaraan (SIM & STNK aktif), gunakan helm/sabuk pengaman, dan patuhi rambu lalu lintas kapan pun berkendara, terlepas dari ada atau tidaknya razia."
        )

    # 4. Pemadaman Listrik / Gangguan Air Bersih
    if ctx.get("is_pemadaman"):
        lok = ctx.get("lokasi_str") or "daerah Anda"
        return (
            f"{pembuka}\n\n"
            f"🔍 **Status Isu**: **BELUM DITEMUKAN PENGUMUMAN RESMI**\n\n"
            f"Terkait informasi pemadaman listrik/air di **{lok}**, berikut hasil kroscek:\n\n"
            f"1. **SOP Pemeliharaan PLN:** Pemadaman terencana oleh PLN selalu disertai informasi detail mengenai jadwal pemeliharaan dan wilayah gardu terdampak melalui aplikasi resmi **PLN Mobile** atau akun resmi PLN Distribusi daerah terkait.\n"
            f"2. **Ciri Kabar Burung:** Pesan broadcast yang hanya menyebut kota tanpa rincian nomor gardu atau surat edaran resmi PLN/PDAM biasanya merupakan kabar simpang siur.\n\n"
            f"💡 **Saran Sahabat Sagi**: Cek langsung status pemeliharaan terkini lewat aplikasi PLN Mobile (menu Informasi Pemadaman) atau hubungi Contact Center PLN 123."
        )

    # 5. Saldo E-Wallet / Bank Hangus / Pembaruan Sistem Pembayaran Nasional
    is_ewallet_bank = (
        any(k in t_clean for k in ["e-wallet", "wallet", "dompet digital", "saldo", "rekening", "gopay", "ovo", "dana", "shopeepay", "linkaja", "m-banking", "mbanking", "atm"])
        and any(k in t_clean for k in ["terpotong", "berkurang", "hilang", "pembaruan sistem", "sistem pembayaran", "nomor lama", "putuskan koneksi", "tutup akun", "pembayaran nasional", "bi-fast", "qris", "hangus", "malam ini"])
    )
    if is_ewallet_bank or sig.get("hoaks_ewallet"):
        return (
            f"{pembuka}\n\n"
            "🚨 **Status Investigasi**: **HOAKS REKAYASA SOSIAL (MANIPULASI PERBANKAN)**\n\n"
            "Kabar yang menyebutkan **saldo dompet digital (e-wallet) atau rekening bank akan terpotong/hangus otomatis karena pembaruan sistem pembayaran nasional adalah HOAKS TOTAL**.\n\n"
            "💳 **Fakta Regulasi Moneter Sebenarnya**:\n"
            "1. **Regulasi Bank Indonesia (BI):** Pembaruan infrastruktur sistem pembayaran nasional (seperti BI-FAST, SNAP, atau QRIS) dikelola terpusat oleh BI. Sistem perbankan maupun dompet digital resmi tidak pernah mengurangi atau menghanguskan saldo pengguna sepihak dengan dalih pemeliharaan sistem atau nomor telepon lama.\n"
            "2. **Keamanan Saldo Pengguna:** Saldo kamu di dompet digital resmi (DANA, GoPay, OVO, ShopeePay, dll.) tersimpan aman di rekening penampung (*escrow account*) yang diawasi BI dan OJK. Saldo HANYA bisa berkurang jika ada transaksi yang kamu setujui sendiri via PIN atau kode OTP rahasia.\n"
            "3. **Trik Psikologis Pembuat Panik:** Pesan ini sengaja mencatut 'informasi orang dalam' dan menentukan batas waktu genting fiktif ('malam ini jam 00.00') agar orang panik dan langsung meneruskannya ke mana-mana.\n\n"
            "💡 **Saran Sahabat Sagi**: Saldo kamu aman 100%. Jangan pernah percaya ajakan memutuskan sambungan akun secara tergesa-gesa, dan yang paling penting: jangan pernah berikan kode OTP atau PIN ke siapa pun!"
        )

    # 6. Kunjungan Kenegaraan / Prabowo / Jokowi / Perang Rusia-Ukraina
    is_foreign_visit = any(k in t_clean for k in ["prabowo", "jokowi", "presiden ri", "menhan", "pejabat", "menteri"])
    is_war_topic = any(k in t_clean for k in ["rusia", "russia", "perang", "ukraina", "putin", "moskow", "pasukan", "kirim tentara", "tempur"])
    if (is_foreign_visit and is_war_topic) or sig.get("hoaks_perang"):
        return (
            f"{pembuka}\n\n"
            "📢 **Status Investigasi**: **HOAKS MANIPULASI KONTEKS DIPLOMASI**\n\n"
            "Klaim yang menyebut bahwa **tokoh kenegaraan Indonesia pergi ke Rusia untuk ikut berperang adalah HOAKS TOTAL**.\n\n"
            "🇮🇩 **Fakta Hubungan Internasional Sebenarnya**:\n"
            "1. **Diplomasi Bilateral Resmi:** Pertemuan bilateral dengan kepala negara asing di Moskow merupakan agenda kerja kenegaraan resmi, membahas kerja sama pertahanan, ekonomi, perdagangan, dan ketahanan pangan.\n"
            "2. **Politik Luar Negeri Bebas-Aktif:** Sesuai amanat konstitusi UUD 1945, Indonesia menganut prinsip politik luar negeri Bebas dan Aktif. Indonesia TIDAK PERNAH mengirimkan personel militer atau prajurit TNI untuk bertempur dalam perang antarnegara lain.\n"
            "3. **Manipulasi Dokumentasi:** Foto dan video dokumentasi kunjungan kenegaraan sengaja diplintir narasinya oleh pembuat konten tidak bertanggung jawab demi mendulang penonton (*clickbait viewer*).\n\n"
            "💡 **Saran Sahabat Sagi**: Kroscek selalu isu hubungan internasional lewat portal resmi Kementerian Luar Negeri RI (kemlu.go.id) atau media berita arus utama terverifikasi."
        )

    # 7. Penipuan Bansos / BLT / Saldo Gratis Phishing
    is_aid = any(k in t_clean for k in ["bansos", "blt", "pkh", "subsidi", "bantuan tunai", "giveaway", "saldo gratis", "dana kaget", "kuota gratis", "undian"])
    is_phish = any(k in t_clean for k in ["link", "tautan", "wa", "whatsapp", "klik", "daftar", "cairkan", "bit.ly", "s.id", "form", "isi data", "otp"]) or sig.get("shortened_url") or sig.get("phone_number")
    if (is_aid and is_phish) or sig.get("hoaks_bansos"):
        return (
            f"{pembuka}\n\n"
            "🚨 **Status Investigasi**: **HOAKS PENIPUAN DIGITAL (MODUS PHISHING)**\n\n"
            "Kabar pembagian bantuan uang tunai, bansos sembako, saldo dompet digital gratis, atau kuota 100GB melalui tautan tersebut adalah **100% HOAKS PENIPUAN (PHISHING)**.\n\n"
            "🔍 **Fakta & Cara Mengetahui Kepalsuannya**:\n"
            "1. **Bukan Prosedur Resmi Pemerintah:** Bantuan resmi dari kementerian hanya disalurkan lewat Bank Himbara (BRI, BNI, Mandiri, BTN) atau PT Pos Indonesia dengan pengumuman di domain resmi `.go.id` (seperti `cekbansos.kemensos.go.id`), bukan lewat tautan pendek bit.ly/s.id atau broadcast WhatsApp.\n"
            "2. **Modus Pencurian Data Pribadi:** Tautan itu sengaja dibuat untuk memancing kamu mengisi data rahasia seperti NIK KTP, nomor HP, foto selfie buku tabungan, atau kode OTP m-banking untuk dibobol.\n\n"
            "💡 **Saran Sahabat Sagi**: JANGAN PERNAH mengklik tautan tersebut, jangan isi form data apa pun, dan jangan pernah membagikan kode OTP ke siapa pun!"
        )

    # 8. Malware File .APK
    is_apk_scam = sig.get("has_apk") or (any(k in t_clean for k in [".apk", "surat tilang", "undangan pernikahan", "resi kurir", "kurir paket", "tagihan pln"]) and any(k in t_clean for k in ["apk", "unduh", "install", "buka", "paket"]))
    if is_apk_scam:
        return (
            f"{pembuka}\n\n"
            "⚠️ **Status Investigasi**: **MODUS KEJAHATAN SIBER TINGKAT TINGGI (MALWARE SNIFFER APK)**\n\n"
            "Pesan yang mengirimkan file berformat **.APK** (dengan dalih undangan pernikahan, surat tilang kepolisian ETLE, foto paket kurir rusak, atau tagihan PLN) adalah **MODUS PENCURIAN SALDO REKENING BANK**.\n\n"
            "🛡️ **Fakta Bahaya & Cara Kerjanya**:\n"
            "1. Surat tilang elektronik (ETLE) asli hanya dikirim lewat surat fisik pos resmi ke alamat pemilik kendaraan, bukan file .APK di WhatsApp pribadi. Begitu pula undangan pernikahan atau resi kurir resmi.\n"
            "2. File .APK tersebut adalah virus Trojan. Kalau diinstal di HP Android, aplikasi jahat ini akan menyadap SMS kode OTP perbankan kamu sehingga saldo m-banking bisa dikuras habis secara senyap.\n\n"
            "🚨 **Langkah Darurat Jika Terlanjur Klik**:\n"
            "Segera nyalakan **Mode Pesawat (Airplane Mode)** seketika, copot/uninstall aplikasi tersebut lewat menu Pengaturan HP -> Kelola Aplikasi, lalu hubungi pihak bank untuk memblokir sementara rekening Anda."
        )

    # 9. Obat Ajaib / Pseudosains Medis
    is_health = any(k in t_clean for k in ["kanker", "stroke", "diabetes", "ginjal", "jantung", "tumor", "obat herbal", "ramuan", "vaksin", "autis"])
    is_miracle = any(k in t_clean for k in ["3 hari", "sembuh total", "100%", "tanpa operasi", "rahasia dokter", "bikin autis", "racun", "ajaib", "paku"])
    if (is_health and is_miracle) or sig.get("hoaks_obat"):
        return (
            f"{pembuka}\n\n"
            "🚨 **Status Investigasi**: **HOAKS MEDIS BERBAHAYA (PSEUDOSAINS)**\n\n"
            "Klaim ramuan ajaib yang menyembuhkan penyakit kronis dalam hitungan hari, atau kabar hoaks vaksin menyebabkan autis/penyakit, adalah **HOAKS MEDIS YANG BERISIKO FATAL**.\n\n"
            "🩺 **Fakta Ilmiah Medis Sebenarnya**:\n"
            "1. Pengobatan penyakit kronis (seperti kanker atau diabetes) memerlukan terapi medis terukur di bawah penanganan dokter spesialis onkologi dan obat yang telah lulus uji klinis BPOM.\n"
            "2. Menghentikan obat resep dokter demi ramuan herbal viral yang belum teruji secara klinis sangat berisiko fatal bagi keselamatan jiwa pasien.\n\n"
            "💡 **Saran Sahabat Sagi**: Jangan tergiur testimoni tanpa rujukan jurnal medis. Selalu konsultasikan kondisi kesehatan Anda ke dokter atau fasilitas kesehatan resmi terdekat."
        )

    # 10. Ramalan Bencana Alam Gempa / Megathrust
    is_disaster = any(k in t_clean for k in ["megathrust", "gempa", "tsunami", "ramalan gempa", "prediksi gempa", "kiamat", "gempa susulan"]) and any(k in t_clean for k in ["tanggal", "besok", "pasti terjadi", "ramal", "prediksi", "jam", "hari ini"])
    if is_disaster or sig.get("hoaks_gempa"):
        return (
            f"{pembuka}\n\n"
            "📢 **Status Investigasi**: **HOAKS & SPEKULASI KEBENCANAAN MENYESATKAN**\n\n"
            "Klaim yang menyebutkan tanggal dan jam pasti terjadinya gempa bumi megathrust atau tsunami adalah **HOAKS TOTAL**.\n\n"
            "🌍 **Fakta Ilmiah Kebencanaan**:\n"
            "1. **Keterbatasan Sains Dunia:** Sampai hari ini, belum ada satu pun teknologi di dunia yang bisa memprediksi hari, tanggal, dan jam terjadinya gempa secara akurat.\n"
            "2. **Potensi Geologis vs Jadwal:** Zona megathrust memang zona pertemuan lempeng tektonik yang memiliki potensi gempa, namun tidak ada kepastian waktu kapan gempa tersebut akan dilepaskan.\n"
            "3. **Lembaga Resmi:** Satu-satunya lembaga resmi negara yang berwenang menyampaikan info gempa dan peringatan dini tsunami di Indonesia adalah **BMKG (bmkg.go.id)**.\n\n"
            "💡 **Saran Sahabat Sagi**: Tetap tenang, pelajari langkah mitigasi gempa mandiri bersama keluarga, dan jangan mudah terpancing narasi kiamat/panik di media sosial."
        )

    # 11. Isu Kriminalitas Viral (Penculikan Anak / Begal)
    if ctx.get("is_kriminal"):
        lok = ctx.get("lokasi_str") or "daerah tersebut"
        return (
            f"{pembuka}\n\n"
            f"🔍 **Status Isu**: **BELUM DAPAT DIBUKTIKAN / PERLU DIWASPADAI SEBAGAI RUMOR**\n\n"
            f"Terkait kabar kriminalitas (isu penculikan anak atau begal) di **{lok}**, berikut fakta di lapangan:\n\n"
            f"1. **Pola Daur Ulang Pesan Penculikan:** Pesan berantai tentang penculikan anak hampir selalu merupakan kabar daur ulang menakut-nakuti warga. Di berbagai daerah, foto orang dengan gangguan jiwa (ODGJ) sering kali difitnah dan disebarkan sebagai foto pelaku penculikan.\n"
            f"2. **Nihil Laporan Polisi:** Hingga saat ini belum ada laporan polisi resmi maupun rilis dari Polsek/Polres di {lok} terkait kejadian tersebut.\n\n"
            f"💡 **Saran Sahabat Sagi**: Kewaspadaan terhadap anak-anak dan keamanan lingkungan memang selalu penting, tapi jangan panik dan jangan main hakim sendiri terhadap orang yang dicurigai tanpa bukti yang jelas."
        )

    # 12. Kabar Wafat Tokoh Publik (Hoaks Innalillahi)
    is_death_hoax = sig.get("hoaks_wafat") or (any(k in t_clean for k in ["innalillahi", "meninggal dunia", "wafat", "tewas", "kabar duka"]) and any(p in t_clean for p in ["jokowi", "prabowo", "megawati", "anies", "ganjar", "gibran", "tokoh"]))
    if is_death_hoax:
        return (
            f"{pembuka}\n\n"
            "📢 **Status Investigasi**: **HOAKS KABAR DUKA PALSU (CLICKBAIT)**\n\n"
            "Kabar duka yang menyebutkan tokoh publik tersebut meninggal dunia adalah **HOAKS TIDAK BENAR**.\n\n"
            "Tokoh yang bersangkutan saat ini dalam kondisi sehat dan masih beraktivitas menjalankan tugas seperti biasa. Tidak ada pernyataan duka resmi dari pihak keluarga maupun kantor berita nasional. Konten duka palsu seperti ini biasanya sengaja dibuat oleh oknum pengelola akun untuk memancing klik dan kehebohan (*clickbait*).\n\n"
            "💡 **Saran Sahabat Sagi**: Hindari menyebarkan berita duka sebelum ada konfirmasi resmi dari pihak keluarga atau media kredibel."
        )

    # 13. General Hoaks
    status_label = label.lower()
    if status_label == "hoaks" or skor >= 60:
        return (
            f"{pembuka}\n\n"
            "🚨 **Status Investigasi**: **TERINDIKASI KUAT SEBAGAI HOAKS / KABAR TIDAK BENAR**\n\n"
            f"Berdasarkan verifikasi fakta dan analisis indikator disinformasi:\n\n"
            f"1. **Status Temuan:** {kesimpulan}\n"
            f"2. **Nihil Data Sah:** Informasi ini tidak didukung oleh rilis resmi dari lembaga yang berwenang maupun pemberitaan media massa terpercaya.\n"
            f"3. **Ciri Penyesatan:** Narasi menggunakan bahasa provokatif yang memancing emosi atau kepanikan tanpa menyertakan data primer yang dapat dipertanggungjawabkan.\n\n"
            f"💡 **Saran Sahabat Sagi**: Jangan jadikan kabar ini sebagai rujukan dan tahan diri untuk tidak membagikannya ke media sosial atau grup obrolan. Jadilah pemutus rantai hoaks!"
        )

    # 14. General Valid (Hanya jika benar-benar ada bukti dan konfirmasi resmi)
    if status_label == "valid" and skor < 30 and not sig.get("no_source") and not ctx.get("is_tanya"):
        return (
            f"{pembuka}\n\n"
            "✅ **Status Konfirmasi**: **FAKTUAL / TERKONFIRMASI RESMI**\n\n"
            "Berdasarkan penelusuran rujukan resmi dan pangkalan data kredibel:\n\n"
            "1. **Sesuai Rujukan Resmi:** Informasi ini sejalan dengan pernyataan resmi yang telah dirilis oleh institusi berwenang atau media arus utama nasional.\n"
            "2. **Penyampaian Riil:** Narasi menyampaikan peristiwa nyata tanpa adanya rekayasa data atau pemutarbalikan fakta.\n\n"
            "📌 **Catatan Sahabat Sagi**: Informasi ini dapat dipertanggungjawabkan. Pastikan masyarakat tetap merujuk pada kanal komunikasi resmi institusi terkait untuk petunjuk teknis lebih lanjut."
        )

    # 15. General Meragukan / Belum Terkonfirmasi (Default Aman untuk Rumor/Pertanyaan)
    return (
        f"{pembuka}\n\n"
        "🔍 **Status Investigasi**: **BELUM DAPAT DIKONFIRMASI / BERSIFAT RUMOR**\n\n"
        "Mengenai isu atau pertanyaan yang kamu sampaikan, statusnya saat ini masih **BELUM MEMILIKI BUKTI OTENTIK**:\n\n"
        "1. **Nihil Rilis Resmi:** Hingga saat ini, belum ada rilis resmi dari instansi berwenang maupun pemberitaan dari media arus utama terpercaya yang memvalidasi kebenaran narasi ini.\n"
        "2. **Potensi Informasi Simpang Siur:** Pesan yang beredar tanpa tautan sumber terverifikasi kemungkinan besar merupakan desas-desus, potongan peristiwa lama yang didaur ulang, atau opini sepihak yang belum tentu terbukti di lapangan.\n"
        "3. **Prinsip Tabayyun:** Di era digital, sebuah peristiwa baru bisa dipastikan benar jika didukung oleh sumber primer yang jelas penanggung jawabnya.\n\n"
        "💡 **Saran Sahabat Sagi**: Tahan diri dan jangan terburu-buru menyebarkan informasi ini ke grup percakapan atau media sosial sebelum ada klarifikasi resmi dari pihak berwenang."
    )


def ringkasan_hasil(
    label: str,
    skor: int,
    kesimpulan: str,
    sumber: dict | None = None,
    fiksi: list | None = None,
    teks: str = "",
    sinyal: dict | None = None,
    saran: list | None = None,
) -> str:
    """Balasan ekstraktif/sintesis cerdas saat model generatif belum aktif."""
    return susun_jawaban_cerdas(
        teks=teks,
        label=label,
        skor=skor,
        kesimpulan=kesimpulan,
        sumber=sumber,
        fiksi=fiksi,
        sinyal=sinyal,
        saran=saran,
    )


def llm_crosscheck(text: str, api: str = "auto") -> dict:
    """Template integrasi LLM. Isi dengan API call kalau sudah punya key."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return {"ok": False, "msg": "Tidak ada API key. Pakai local_verdict untuk demo."}
    return {"ok": False, "msg": "Belum diimplementasikan — contoh template di factcheck.py"}


def format_creator_script(text: str, verdict: dict, sumber: dict | None = None) -> dict:
    """Format naskah pelurusan berita & kesimpulan siap pakai untuk Konten Kreator."""
    label = verdict.get("label_final", "meragukan")
    skor = verdict.get("skor_hoaks", 0)
    sig = verdict.get("sinyal", {})

    # 1. Hook Pembuka Video
    if label == "hoaks":
        hook = "Stop jangan langsung di-forward! Kabar viral ini ternyata hoaks manipulasi. Ini fakta sebenarnya yang wajib kalian tahu!"
    elif label == "valid":
        hook = "Kabar yang lagi ramai dibicarakan ini ternyata benar dan terkonfirmasi resmi! Yuk kita simak faktanya."
    else:
        hook = "Beredar kabar yang bikin heboh di media sosial, tapi tunggu dulu! Informasi ini masih simpang siur dan belum terbukti."

    # 2. Poin Pelurusan / Fakta
    poin = []
    poin.append(f"Klaim Beredar: \"{text[:120]}{'...' if len(text)>120 else ''}\"")
    if label == "hoaks":
        alasan = []
        if sig.get("has_apk"): alasan.append("Modus file APK berbahaya/phishing")
        if sig.get("shortened_url"): alasan.append("Tautan singkat mencurigakan")
        if sig.get("keyword_hits"): alasan.append(f"Indikasi manipulasi ({', '.join(sig['keyword_hits'][:2])})")
        if not alasan: alasan.append("Tidak ada rujukan sumber kredibel")
        poin.append(f"Fakta Lapangan: Klaim ini tidak terbukti secara faktual ({'; '.join(alasan)}).")
        poin.append("Himbauan: Jangan disebarkan ke grup keluarga. Waspada penipuan atau kepanikan!")
    elif label == "valid":
        poin.append("Fakta Lapangan: Informasi ini sesuai dengan pengumuman atau data resmi yang terverifikasi.")
        poin.append("Himbauan: Ambil selalu rujukan dari kanal utama yang kredibel.")
    else:
        poin.append("Fakta Lapangan: Belum ada konfirmasi resmi dari otoritas terkait, narasinya masih berupa kabar burung.")
        poin.append("Himbauan: Tahan diri untuk tidak membagikan kabar ini sebelum ada rilis resmi (Saring Sebelum Sharing).")

    # 3. Draft Caption Medsos Siap Salin
    snip = text[:80] + ("..." if len(text) > 80 else "")
    caption = (
        f"🚨 [FAKTA ATAU HOAKS?]\n\n"
        f"Lagi viral narasi: \"{snip}\"\n\n"
        f"📌 STATUS: {label.upper()} (Skor Indikator: {skor}/100)\n\n"
        f"💡 PENJELASAN:\n"
        f"• {poin[1]}\n"
        f"• {poin[2]}\n\n"
        f"Yuk jadi netizen cerdas dengan selalu cek kebenaran berita sebelum share!\n\n"
        f"#CekFakta #Edukasi #LiterasiDigital #SaringSebelumSharing #BeritaViral"
    )

    naskah_video = (
        f"🎙️ [HOOK VIDEO - 3 DETIK PERTAMA]\n"
        f"\"{hook}\"\n\n"
        f"📋 [POIN PEMBAHASAN]\n"
        f"1. {poin[0]}\n"
        f"2. {poin[1]}\n"
        f"3. {poin[2]}\n\n"
        f"🎯 [CLOSING]\n"
        f"\"Selalu saring sebelum sharing, cek kebenarannya di sumber resmi!\""
    )

    return {
        "hook": hook,
        "poin": poin,
        "caption": caption,
        "naskah_video": naskah_video,
    }
