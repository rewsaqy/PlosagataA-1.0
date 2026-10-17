"""Generator Dataset Hoaks Modern & Kekinian Indonesia (OTN 2026).

Menghasilkan ribuan variasi kasus hoaks terkini (APK sniffing, AI deepfake,
phishing bansos, megathrust, penipuan giveaway, loker palsu, dan berita valid)
dalam format percakapan multi-turn yang siap dilatih ke model Plosagata_Browser (25M).

Penggunaan:
  python scripts/generate_modern_hoaks.py --target 5000
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Variasi pembuka pertanyaan warganet
PROMPT_TEMPLATES = [
    "Bener gak sih info ini: {klaim}?",
    "Min, tolong cek apakah ini asli atau hoaks: '{klaim}'",
    "Ada broadcast di grup keluarga: '{klaim}'. Benar gak ya?",
    "Lagi viral di TikTok dan WA: {klaim}. Valid gak min?",
    "Tolong periksa berita ini: '{klaim}'. Saya ragu mau forward.",
    "Beneran gak sih gan: {klaim}?",
    "Apakah benar kabar: '{klaim}'?",
    "Mohon konfirmasinya min: {klaim}",
    "Dapet kiriman pesan kayak gini: '{klaim}'. Ini penipuan atau fakta ya?",
    "Halo Sagi, mau tanya apakah berita ini valid: '{klaim}'?",
]

# Kategori 1: APK Sniffing & Malware (Phishing 2024-2026)
KASUS_APK = [
    ("Pemberitahuan tilang elektronik ETLE, kendaraan Anda melanggar marka jalan. Unduh file Surat_Tilang_ETLE.apk berikut untuk melihat bukti tilang.",
     "Surat tilang ETLE berformat .apk",
     "AWAS! Itu 100% HOAKS PENIPUAN sekaligus modus pencurian data rekening perbankan (sniffing APK). Pihak Kepolisian (Korlantas Polri) TIDAK PERNAH mengirimkan surat konfirmasi tilang dalam format file .apk lewat pesan WhatsApp pribadi. Surat tilang resmi selalu dikirim lewat pos ke alamat pemilik kendaraan atau dapat dicek langsung di situs resmi etle-korlantas.info. Jangan unduh atau pasang aplikasi tersebut!"),

    ("Paket Anda tertahan di gudang karena alamat tidak lengkap. Silakan instal aplikasi Kurir_JNE_Lacak.apk untuk verifikasi pengiriman ulang.",
     "Kurir ekspedisi mengirim file .apk untuk lacak paket",
     "WASPADA! Itu adalah HOAKS PENIPUAN MODUS APK KURIR. Ekspedisi resmi (JNE, J&T, SiCepat, Pos Indonesia) melacak resi melalui situs resmi atau aplikasi resmi di Google Play Store / App Store, bukan membagikan file instalasi .apk mentah lewat chat. File .apk tersebut adalah malware yang bisa menyadap SMS kode OTP perbankan Anda. Segera blokir pengirimnya!"),

    ("Undangan Pernikahan Digital Kami: Kami mengundang Bapak/Ibu hadir ke pernikahan kami. Mohon instal Surat_Undangan_Digital.apk untuk membuka lokasi acara.",
     "Undangan pernikahan berformat .apk",
     "PERINGATAN KERAS! Itu adalah MODUS PENIPUAN KLASIK SNIFFING APK. Undangan pernikahan digital yang asli berupa tautan website resmi (seperti domain .com atau platform undangan web), bukan file aplikasi (.apk) yang meminta izin akses SMS. Jika diinstal, pelaku dapat menguras saldo m-banking Anda tanpa sepengetahuan Anda. Jangan pernah klik atau buka file tersebut!"),

    ("Pemberitahuan resmi PLN: Tagihan listrik rumah Anda menunggak 2 bulan dan aliran listrik akan diputus nanti malam. Unduh Info_Tagihan_PLN.apk untuk melunasi.",
     "Surat pemutusan listrik PLN dalam bentuk file .apk",
     "Ini adalah HOAKS PENIPUAN mengatasnamakan PT PLN (Persero). PLN memiliki aplikasi resmi 'PLN Mobile' di Google Play Store dan tidak pernah menagih tunggakan dengan mengirim file .apk mentah lewat nomor WhatsApp pribadi. Abaikan dan jangan unduh aplikasi tersebut!"),

    ("Bansos tunai Kemensos Rp 600.000 sudah cair hari ini! Unduh aplikasi BANSOS-CAIR-2026.APK ini untuk mencairkan saldo ke dompet digital Anda sekarang.",
     "Aplikasi pencairan bansos Kemensos berformat .apk",
     "Itu adalah HOAKS BERBAHAYA BERKEDOK BANSOS! Kementerian Sosial tidak pernah merilis aplikasi bansos lewat file kiriman WhatsApp personal. Seluruh pengecekan penerima bansos hanya dilakukan melalui portal resmi cekbansos.kemensos.go.id. Menginstal file APK ini berisiko membuat ponsel Anda disusupi virus perbankan."),
]

# Kategori 2: Phishing Tautan, Bansos & Giveaway Fiktif
KASUS_PHISHING = [
    ("Klaim saldo DANA Kaget Rp 2.500.000 spesial hari ini! Buka tautan https://s.id/dana-kaget-resmi masukkan nomor ponsel dan PIN Anda sekarang!",
     "DANA Kaget jutaan rupiah lewat tautan s.id",
     "Klaim tersebut dipastikan HOAKS PHISHING. Dompet digital DANA tidak pernah membagikan saldo jutaan secara cuma-cuma melalui shortlink pihak ketiga seperti s.id. Halaman palsu tersebut dirancang untuk mencuri nomor telepon, PIN, dan kode OTP akun Anda. Jangan pernah memasukkan data rahasia perbankan pada situs mencurigakan!"),

    ("Bagi-bagi kuota internet gratis 100GB edisi perayaan kemerdekaan untuk semua operator, klik link http://kuota-gratis-100gb.my.id bagikan ke 10 grup WhatsApp!",
     "Kuota gratis 100GB bagikan ke grup WA",
     "Informasi tersebut 100% HOAKS DAUR ULANG. Operator telekomunikasi (Telkomsel, Indosat, XL, Smartfren) selalu mengumumkan promo resmi lewat aplikasi resmi atau situs korporat mereka, bukan via tautan domain gratisan atau mewajibkan forward berantai ke grup. Tautan tersebut berbahaya dan berisiko mencuri data pribadi Anda."),

    ("Program tebus murah minyak goreng Rp 5.000 per liter dari Alfamart, klik tautan http://promo-alfamart-minyak.site isi formulir pendaftaran sekarang sebelum kehabisan!",
     "Tebus minyak goreng murah via link abal-abal",
     "Klaim itu adalah HOAKS PENIPUAN TAUTAN PALSU. Pihak minimarket resmi telah menegaskan seluruh program promosi hanya diumumkan di akun media sosial resmi bercentang biru atau website korporat resmi. Jangan isi data diri Anda pada formulir di domain tidak dikenal."),

    ("Raffi Ahmad dan Nagita Slavina bagi rezeki uang tunai 50 juta rupiah, hubungi admin via WhatsApp dan transfer biaya pendaftaran Rp 250.000 untuk pencairan hadiah.",
     "Giveaway artis minta transfer biaya admin",
     "Itu jelas PENIPUAN BERKEDOK NAMA ARTIS (IMPERSONASI)! Ciri khas penipuan hadiah sangat nyata: menjanjikan uang besar tetapi meminta korban mentransfer 'uang muka/administrasi' terlebih dahulu. Figur publik asli tidak pernah memungut biaya apa pun dalam pembagian hadiah. Jangan kirim uang sepeser pun!"),

    ("Pendaftaran bantuan modal UMKM non-tunai 15 juta dari kementerian dibuka hari ini, klik tautan https://bantuan-umkm.web.app dan upload foto KTP serta selfie memegang buku tabungan.",
     "Bantuan modal UMKM minta foto KTP dan buku tabungan di link gratisan",
     "WASPADA! Itu adalah HOAKS PENCURIAN IDENTITAS (IDENTITY THEFT). Kementerian Koperasi dan UKM tidak pernah membuka pendaftaran bantuan menggunakan domain gratisan seperti web.app atau blogspot. Foto KTP dan selfie rekening Anda yang dikirim ke tautan tersebut berisiko disalahgunakan untuk pinjaman online (pinjol) ilegal."),
]

# Kategori 3: AI Deepfake & Rekayasa Digital
KASUS_DEEPFAKE = [
    ("Beredar video Presiden mengumumkan investasi trading saham kilat yang bisa melipatgandakan uang 10 kali lipat dalam seminggu di akun TikTok @investasiresmi.",
     "Video Presiden tawarkan trading investasi kilat",
     "KONTEN TERSEBUT ADALAH HOAKS DEEPFAKE AI! Analisis forensik audio-visual membuktikan bahwa video tersebut merupakan manipulasi rekaman pidato kenegaraan yang disinkronisasi gerak bibirnya (lip-sync) dan ditimpa audio tiruan kecerdasan buatan (voice cloning). Kepala Negara tidak pernah mempromosikan platform trading komersial apa pun."),

    ("Viral foto penampakan kota Jakarta tenggelam oleh gelombang tsunami setinggi gedung bertingkat sore kemarin, ngeri banget!",
     "Foto Jakarta tenggelam tsunami buatan AI",
     "Foto tersebut dipastikan HOAKS REKAAN DIGITAL (AI-GENERATED IMAGE). Tidak ada bencana tsunami yang menerjang Jakarta. Foto tersebut memiliki kejanggalan visual khas gambar AI pada tekstur air, bayangan gedung, dan detail arsitektur yang tidak konsisten. BMKG mengonfirmasi perairan Teluk Jakarta dalam status normal."),

    ("Beredar rekaman telepon Menteri Keuangan menyatakan Indonesia resmi bangkrut dan uang tabungan di bank akan disita negara per akhir bulan ini.",
     "Rekaman suara Menkeu klaim Indonesia bangkrut",
     "Informasi tersebut 100% HOAKS MANIPULASI SUARA. Rekaman audio tersebut dibuat menggunakan teknologi kloning suara AI untuk memicu kepanikan moneter. Bank Indonesia dan OJK menegaskan kondisi perbankan nasional stabil dan dana nasabah dijamin aman oleh Lembaga Penjamin Simpanan (LPS)."),
]

# Kategori 4: Hoaks Medis & Pseudosains
KASUS_MEDIS = [
    ("Dokter luar negeri ungkap rahasia: minum rebusan daun sirsak campur soda kue setiap pagi dijamin sembuhkan kanker stadium 4 dalam 3 hari tanpa kemoterapi!",
     "Rebusan daun sirsak sembuhkan kanker 3 hari",
     "Klaim itu adalah HOAKS MEDIS BERBAHAYA. Kanker merupakan mutasi sel ganas yang membutuhkan penanganan medis berbasis bukti ilmiah (onkologi) seperti operasi, radioterapi, atau kemoterapi. Menghentikan pengobatan dokter demi ramuan yang belum teruji klinis justru dapat memperparah kondisi dan berakibat fatal."),

    ("Pertolongan pertama stroke: tusuk 10 ujung jari dan kedua daun telinga dengan jarum jahit sampai keluar darah agar darah mengalir lancar dan sembuh seketika!",
     "Menusuk jari dengan jarum saat stroke",
     "JANGAN PERNAH DILAKUKAN! Itu adalah HOAKS MEDIS MEMATIKAN. Menusuk jari tidak memiliki dasar ilmiah untuk mengatasi sumbatan atau pendarahan pembuluh darah di otak. Tindakan ini membuang waktu emas penyelamatan (golden period 3-4,5 jam) dan memicu risiko infeksi tetanus. Pasien stroke harus segera dibawa ke IGD rumah sakit terdekat!"),

    ("Waspada radiasi kosmik melewati bumi nanti malam pukul 00.30 sampai 03.30, segera matikan semua ponsel dan jauhkan dari tubuh karena bisa memicu kanker dan ledakan baterai!",
     "Radiasi kosmik malam ini disuruh matikan HP",
     "Kabar ini adalah HOAKS DAUR ULANG yang sudah berulang sejak 2008. Lembaga riset astronomi BRIN dan NASA menegaskan tidak ada peristiwa radiasi kosmik terjadwal yang mampu menembus atmosfer bumi dan membahayakan perangkat ponsel pada jam tertentu. Ponsel Anda aman dan tidak perlu dimatikan."),

    ("Vaksin polio dan campak sengaja disuntikkan untuk memasukkan microchip pengendali populasi dan merusak kecerdasan anak Indonesia.",
     "Vaksin mengandung microchip pengendali pikiran",
     "Itu adalah HOAKS TEORI KONSPIRASI tanpa dasar sains. Vaksin mengandung antigen virus yang dilemahkan atau dimatikan untuk membentuk antibodi kekebalan tubuh, bukan microchip elektronik. Seluruh vaksin telah melalui uji klinis ketat dan pengawasan keamanan oleh Badan POM serta WHO."),
]

# Kategori 5: Bencana & Histeria Megathrust
KASUS_BENCANA = [
    ("Info orang dalam BMKG: besok malam jam 23.15 WIB dipastikan terjadi gempa megathrust magnitudo 9,5 di selatan pulau Jawa disertai tsunami 30 meter, segera mengungsi!",
     "Prediksi jam pasti gempa megathrust besok malam",
     "Itu 100% HOAKS KEBENCANAAN YANG MENYESATKAN! Hingga saat ini, belum ada satupun sains atau teknologi di muka bumi yang mampu memprediksi hari, tanggal, dan jam terjadinya gempa secara presisi. BMKG secara resmi menegaskan bahwa potensi gempa megathrust memang ada di zona tektonik, tetapi waktu terjadinya tidak dapat diprediksi. Jangan panik oleh ramalan palsu!"),

    ("Awan berbentuk gelombang aneh di langit Yogyakarta pagi ini, pakar mengonfirmasi pertanda pasti gempa bumi dahsyat dalam waktu kurang dari 24 jam.",
     "Awan aneh tanda pasti gempa bumi 24 jam",
     "Klaim tersebut adalah HOAKS MITOS PSEUDOSAINS. Bentuk awan dipengaruhi oleh dinamika atmosfer, kelembapan, suhu, dan kecepatan angin, bukan dipicu oleh pergeseran lempeng tektonik di kedalaman bumi. BMKG menyatakan tidak ada hubungan ilmiah antara formasi awan dengan kepastian waktu terjadinya gempa."),
]

# Kategori 6: Berita Valid & Faktual (Keseimbangan Ground Truth)
KASUS_VALID = [
    ("BMKG mencatat gempa bumi magnitudo 4,8 di barat daya Bayah Banten dengan kedalaman 10 km, tidak berpotensi tsunami.",
     "Gempa Banten M 4.8 BMKG",
     "Informasi tersebut VALID dan FAKTUAL. Parameter gempa bersesuaian dengan siaran resmi BMKG dan telah disosialisasikan lewat kanal resmi InfoBMKG serta media massa terpercaya."),

    ("Kementerian Kesehatan menyelenggarakan Pekan Imunisasi Nasional (PIN) Polio serentak gratis di seluruh posyandu dan puskesmas.",
     "PIN Polio Kemenkes gratis",
     "Informasi tersebut VALID dan ASLI. Program PIN Polio adalah agenda resmi pemerintah untuk mencegah penularan polio. Panduan dan jadwal resminya dapat dicek di situs kemkes.go.id."),

    ("Badan Pusat Statistik (BPS) merilis pertumbuhan ekonomi Indonesia triwulan II-2024 tumbuh sebesar 5,05 persen secara tahunan (y-on-y).",
     "Pertumbuhan ekonomi BPS 5.05 persen",
     "Berita tersebut BENAR dan FAKTUAL. Data tersebut bersumber dari Berita Resmi Statistik yang dipublikasikan secara transparan melalui portal resmi bps.go.id."),

    ("PT Kereta Api Indonesia (Persero) mengoperasikan rangkaian kereta New Generation pada sejumlah rute antarkota untuk meningkatkan kenyamanan penumpang.",
     "Kereta New Generation KAI",
     "Kabar ini VALID dan RESMI. Pembaruan sarana kereta eksekutif dan ekonomi New Generation telah resmi diluncurkan secara bertahap oleh KAI dan diliput luas oleh media nasional."),

    ("Veddriq Leonardo berhasil mempersembahkan medali emas pertama bagi Indonesia di ajang Olimpiade Paris 2024 nomor panjat tebing speed putra.",
     "Medali emas Veddriq Leonardo Olimpiade",
     "Informasi tersebut ASLI dan MENJADI SEJARAH OLAHRAGA NASIONAL. Kemenangan Veddriq tercatat resmi di situs Olimpiade Internasional dan Komite Olimpiade Indonesia."),
]


def generate_samples(target_count: int = 5000) -> list[dict]:
    all_cases = (
        KASUS_APK * 6 +
        KASUS_PHISHING * 6 +
        KASUS_DEEPFAKE * 5 +
        KASUS_MEDIS * 5 +
        KASUS_BENCANA * 5 +
        KASUS_VALID * 6
    )
    
    samples = []
    # Variasi kata tambahan di klaim warganet agar dataset kaya dan natural
    variasi_klaim = [
        "",
        " viralkan sebelum dihapus",
        " katanya dari sumber orang dalam",
        " ada yang baru ngirim ke grup WA",
        " mohon info kebenarannya min",
        " barusan dapet pesan ini",
        " bener gak ya kira-kira?",
    ]

    while len(samples) < target_count:
        klaim_raw, keyword, penjelasan = random.choice(all_cases)
        imbuhan = random.choice(variasi_klaim)
        klaim = klaim_raw + imbuhan

        template = random.choice(PROMPT_TEMPLATES)
        user_msg = template.format(klaim=klaim)

        # Variasi respons santun dan tajam khas Sagi
        pembuka_sagi = random.choice([
            "Halo! Mari kita telaah klaim ini dengan teliti.",
            "Halo! Terima kasih sudah bertanya sebelum menyebarkannya.",
            "Penting sekali kamu kroscek kabar ini sebelum diteruskan.",
            "Halo! Sagi bantu bedah kebenaran informasinya ya.",
            "Waspada ya, yuk kita periksa bukti resminya bersama-sama.",
        ])

        assistant_msg = f"{pembuka_sagi}\n\n{penjelasan}\n\nIngat prinsip cerdas bermedia: **Saring sebelum Sharing**!"

        sample = {
            "messages": [
                {"role": "system", "content": "Kamu Sagi, asisten AI cerdas pelurus hoaks dan analis disinformasi digital berbasis Plosagata_Browser (25M Core)."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generator Dataset Kasus Hoaks Modern Indonesia")
    parser.add_argument("--target", type=int, default=5000, help="Jumlah sampel dialog yang digenerate")
    parser.add_argument("--out", default="data/train_chat.jsonl", help="File output dataset")
    args = parser.parse_args()

    print(f"Mengenerate {args.target:,} sampel kasus hoaks modern Indonesia...")
    samples = generate_samples(args.target)

    # Baca data yang sudah ada agar tidak terhapus
    existing = []
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(line)

    print(f"Data lama: {len(existing):,} sampel")
    
    # Tulis gabungan data baru
    with open(out_path, "w", encoding="utf-8") as f:
        for line in existing:
            f.write(line + "\n")
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_now = len(existing) + len(samples)
    print(f"✅ Selesai! Total data di {args.out} sekarang: {total_now:,} sampel chat!")


if __name__ == "__main__":
    main()
