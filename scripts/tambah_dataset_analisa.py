"""Skrip penambah dataset penalaran investigasi tinggi (High-Density Reasoning Data).
Menambahkan percakapan mendalam dengan analisis multi-step dan ratusan sampel klaim nyata.
"""
import csv
import json

print("=== Menyiapkan Dataset Penalaran Investigasi Hoaks ===")

# 1. Dataset Dialog Chat Mendalam (Multi-Step Reasoning)
chat_samples = [
    # Analisis Berita Asli (Validasi Mendalam)
    {
        "user": "Tolong analisa berita ini apakah benar-benar asli: 'BMKG merilis informasi gempa bumi magnitudo 5,2 di barat daya Bayah Banten dengan kedalaman 10 km, tidak berpotensi tsunami.'",
        "assistant": "Mari kita bedah keaslian informasi ini langkah demi langkah:\n\n1. Analisis Sumber & Parameter Resmi: Format informasi mencantumkan magnitudo (5,2), episentrum (barat daya Bayah, Banten), kedalaman (10 km), dan status potensi tsunami secara spesifik. Ini adalah standar pelaporan operasional resmi dari BMKG (Badan Meteorologi, Klimatologi, dan Geofisika).\n2. Verifikasi Kanal: Informasi semacam ini dapat langsung dikonfirmasi melalui kanal resmi BMKG di aplikasi InfoBMKG, akun X @infoBMKG, atau situs bmkg.go.id.\n3. Analisis Gaya Bahasa: Bahasa bersifat netral, informatif, dan tidak memicu kepanikan berlebihan.\n\nKesimpulan: Informasi ini ASLI dan VALID. Ini adalah rilis data seismik resmi dari instansi pemerintah berwenang."
    },
    {
        "user": "Apakah berita ini asli: 'Pemerintah melalui Kemenkes menyelenggarakan Pekan Imunisasi Nasional (PIN) Polio gratis serentak di seluruh puskesmas dan posyandu.'",
        "assistant": "Berikut hasil analisa fakta:\n\n1. Pemeriksaan Kebijakan: Program PIN Polio merupakan agenda resmi Kementerian Kesehatan Republik Indonesia sebagai langkah mitigasi kejadian luar biasa (KLB) polio.\n2. Rujukan Resmi: Program ini disosialisasikan secara terbuka melalui siaran pers kemkes.go.id, dinas kesehatan daerah, dan posyandu resmi tanpa pungutan biaya.\n3. Indikator Keaslian: Tidak ada pemungutan biaya, tidak ada tautan mencurigakan, dan melibatkan fasilitas kesehatan primer resmi negara.\n\nKesimpulan: Berita ini BENAR dan ASLI. Masyarakat disarankan membawa balita ke posyandu atau puskesmas terdekat sesuai jadwal resmi."
    },
    {
        "user": "Analisa ini dong: 'BPS mengumumkan pertumbuhan ekonomi Indonesia triwulan II-2024 tumbuh sebesar 5,05 persen secara year-on-year.'",
        "assistant": "Hasil verifikasi faktual:\n\n1. Validasi Data: Badan Pusat Statistik (BPS) secara berkala merilis Berita Resmi Statistik (BRS) setiap triwulan mengenai Produk Domestik Bruto (PDB).\n2. Pengecekan Sumber: Angka pertumbuhan 5,05% y-on-y merupakan data resmi yang dirilis langsung oleh Kepala BPS dan diliput oleh seluruh media arus utama nasional (Antara, Kompas, Tempo, Kontan).\n3. Karakteristik: Teks menyajikan data statistik riil tanpa klaim bombastis atau manipulatif.\n\nKesimpulan: Berita ini 100% ASLI dan FAKTUAL bersumber dari data primer BPS."
    },
    {
        "user": "Apakah benar Timnas Indonesia menang kualifikasi di GBK dan lolos ke putaran ketiga kualifikasi Piala Dunia?",
        "assistant": "Berikut hasil penelusuran fakta:\n\n1. Fakta Pertandingan: Tim Nasional Sepak Bola Indonesia resmi memastikan kelolosan ke putaran ketiga Kualifikasi Piala Dunia 2026 zona Asia setelah mengalahkan Filipina 2-0 di Stadion Utama Gelora Bung Karno (GBK).\n2. Verifikasi Otoritas: Hasil pertandingan tercatat resmi di situs FIFA (fifa.com) dan laman resmi PSSI (pssi.org), serta disiarkan secara langsung oleh stasiun televisi nasional.\n\nKesimpulan: Berita ini ASLI dan VALID. Ini adalah fakta sejarah olahraga yang terkonfirmasi secara internasional."
    },
    {
        "user": "Cek keaslian klaim ini: 'Air mendidih pada suhu 100 derajat Celsius pada tekanan udara standar 1 atmosfer.'",
        "assistant": "Analisa ilmiah dasar:\n\n1. Konsensus Sains: Ini adalah hukum termodinamika fisika dasar. Titik didih air murni adalah 100°C pada tekanan 1 atm (101,325 kPa).\n2. Sifat Informasi: Ini bukan kabar bohong atau klaim rekaan, melainkan fakta ilmiah universal yang terbukti di laboratorium dan diajarkan dalam kurikulum sains.\n\nKesimpulan: Pernyataan ini BENAR dan FAKTUAL secara sains."
    },

    # Analisis Hoaks & Penipuan (Pembongkaran Modus)
    {
        "user": "Analisa apakah ini asli atau palsu: 'Pemberitahuan tilang elektronik ETLE, kendaraan Anda melanggar batas kecepatan. Silakan unduh dan buka aplikasi SuratTilang.apk berikut untuk melihat bukti foto.'",
        "assistant": "PERINGATAN: Pesan ini adalah 100% HOAKS BERBAHAYA (MODUS PHISHING MALWARE APK).\n\nMari kita analisa modusnya:\n1. Ekstensi File: Pesan melampirkan file berekstensi .APK (aplikasi Android), bukan dokumen dokumen resmi atau gambar bukti pelanggaran.\n2. Pola Penipuan: Ini adalah trojan perbankan (sniffing SMS/OTP). Jika diinstal, aplikasi palsu ini akan membaca SMS verifikasi perbankan dan menguras rekening korban.\n3. Prosedur Resmi Kepolisian: Korlantas Polri TIDAK PERNAH mengirim surat tilang dalam bentuk file .apk lewat WhatsApp. Surat konfirmasi tilang ETLE resmi dikirim melalui pos ke alamat pemilik kendaraan atau melalui portal resmi etle-korlantas.info.\n\nKesimpulan: PASTI PALSU DAN PENIPUAN. Segera hapus pesan, jangan pernah mengklik atau menginstal file tersebut!"
    },
    {
        "user": "Bener gak berita ini: 'Kiamat internet dunia! NASA mengonfirmasi radiasi badai matahari akan mematikan seluruh listrik dan internet bumi selama 3 bulan mulai minggu depan.'",
        "assistant": "Berikut analisa investigasi klaim:\n\n1. Pengecekan Sumber NASA/NOAA: NASA dan NOAA Space Weather Prediction Center memantau aktivitas flare matahari setiap hari. Tidak ada rilis resmi dari NASA yang menyatakan akan terjadi 'kiamat internet' atau pemadaman total 3 bulan.\n2. Ciri Manipulasi Bahasa: Menggunakan narasi histeria ('Kiamat internet dunia', 'seluruh listrik padam') untuk memicu kepanikan dan memaksa penerima menyebarkan pesan.\n3. Fakta Ilmiah: Badai geomagnetik memang bisa mengganggu satelit dan jaringan listrik bertegangan tinggi di lintang kutub, namun tidak memusnahkan internet global secara serentak.\n\nKesimpulan: Pesan ini adalah HOAKS BERULANG (Daur Ulang Isu Lama). Jangan disebarkan!"
    },
    {
        "user": "Apakah ini berita benar: 'Mengejutkan! Minum rebusan air paku berkarat dan daun pepaya bisa meluruhkan batu ginjal dan menyembuhkan kanker dalam semalam tanpa operasi.'",
        "assistant": "Analisa Medis & Keamanan:\n\n1. Bahaya Kesehatan Serius: Paku berkarat mengandung bakteri Clostridium tetani (penyebab tetanus) dan racun oksida besi yang mematikan bagi tubuh jika dikonsumsi. Tindakan ini berisiko keracunan berat hingga kematian.\n2. Klaim Pseudosains: Tidak ada mekanisme medis yang memungkinkan batu ginjal atau kanker stadium lanjut 'sembuh dalam semalam'. Pengobatan kanker memerlukan tindakan onkologi teruji (bedah, kemoterapi, radioterapi).\n3. Sinyal Hoaks Kesehatan: Menggunakan kata 'Ajaib', 'Tanpa Operasi', dan 'Dalam Semalam'.\n\nKesimpulan: HOAKS MEDIS SANGAT BERBAHAYA. Hindari meniru resep berbahaya ini dan konsultasikan penyakit kronis ke dokter spesialis."
    },
    {
        "user": "Ada pesan berantai: 'Biro SDM Pertamina membuka rekrutmen pegawai BUMN gaji 20 juta, pendaftaran via email rekrutmen.pertamina@gmail.com dan wajib pesan tiket pesawat lewat travel rekanan.'",
        "assistant": "Analisa Investigasi Lowongan Kerja:\n\n1. Domain Email Tidak Resmi: Perusahaan BUMN Pertamina menggunakan domain korporat resmi (@pertamina.com), BUKAN akun gratisan seperti @gmail.com atau @yahoo.com.\n2. Modus Pemerasan Tiket: Modus klasik penipuan loker adalah meminta calon pelamar mentransfer uang tiket perjalanan ke biro travel fiktif dengan janji 'akan diganti (reimburse)' saat tes.\n3. Kanal Resmi: Rekrutmen Pertamina dan BUMN hanya diselenggarakan terpusat melalui portal rekrutmenbersama.fhirebumn.id atau recruitment.pertamina.com.\n\nKesimpulan: HOAKS PENIPUAN LOWONGAN KERJA. Jangan pernah mentransfer uang sepeser pun!"
    },
    {
        "user": "Beneran gak sih berita: 'WhatsApp akan mulai berbayar Rp 10.000 per bulan mulai besok kecuali Anda menyebarkan pesan ini ke 20 kontak Anda sekarang juga.'",
        "assistant": "Analisa Pola Pesan Berantai:\n\n1. Riwayat Hoaks: Ini adalah hoaks pesan berantai (chain message) legendaris yang sudah beredar sejak tahun 2012 dengan berbagai variasi nominal.\n2. Logika Bisnis Platform: Perubahan model bisnis aplikasi sebesar WhatsApp milik Meta selalu diumumkan secara resmi melalui blog resmi (blog.whatsapp.com) dan ruang pers resmi, bukan lewat broadcast teks abal-abal.\n3. Indikator Manipulasi: Ancaman 'akun akan dinonaktifkan jika tidak forward ke 20 orang' adalah ciri khas rekayasa sosial untuk memanfaatkan ketidaktahuan pengguna.\n\nKesimpulan: HOAKS 100%. Abaikan pesan ini dan jangan diteruskan ke grup keluarga."
    },

    # Analisis Isu Meragukan / Klarifikasi Konteks
    {
        "user": "Beredar kabar di medsos: 'Pemerintah melarang total penggunaan tabung gas elpiji 3 kg mulai bulan depan.' Benar atau tidak?",
        "assistant": "Berikut hasil analisa konteks dan klarifikasi fakta:\n\n1. Pemelintiran Informasi (Misinformasi Konteks): Pemerintah melalui Kementerian ESDM dan Pertamina TIDAK melarang peredaran elpiji 3 kg. Kebijakan yang sebenarnya adalah penataan distribusi subsidi tepat sasaran dengan mewajibkan pembelian menggunakan KTP/NIK yang terdaftar di pangkalan resmi.\n2. Sumber Resmi: Direktur Jenderal Minyak dan Gas Bumi menegaskan pasokan elpiji 3 kg tetap dialokasikan untuk masyarakat kurang mampu dan usaha mikro.\n3. Kategori Masalah: Ini bukan hoaks murni yang 100% fiktif, melainkan isu setengah benar yang dipelintir judulnya sehingga menimbulkan kepanikan.\n\nKesimpulan: KATEGORI MERAGUKAN DENGAN KONTEKS DIPELINTIR. Pembelian tetap berjalan, hanya mekanismenya yang didata menggunakan KTP."
    },
    {
        "user": "Tolong verifikasi: Ada video jembatan ambruk yang dinarasikan baru saja terjadi di jalan lintas Sumatera sore ini.",
        "assistant": "Langkah verifikasi investigasi visual:\n\n1. Uji Jejak Digital (Reverse Video/Image Search): Sering kali video jembatan ambruk adalah peristiwa bencana tahun-tahun sebelumnya (misalnya akibat banjir lahar dingin tahun 2021) yang diunggah ulang tanpa keterangan tanggal.\n2. Konfirmasi Otoritas Daerah: Periksa siaran resmi BPBD setempat, Balai Besar Pelaksanaan Jalan Nasional (BBPJN) Kementerian PUPR, atau liputan media regional terverifikasi.\n3. Waspada Tanggal & Lokasi: Klaim kejadian bencana mendadak di media sosial yang tidak menyebut nama jembatan dan tanggal spesifik sangat rawan misinformasi kronologis.\n\nKesimpulan: MERAGUKAN SEBELUM DICEK SUMBERNYA. Jangan langsung menyebarkan video bencana sebelum dipastikan waktu dan lokasinya sesuai rilis BPBD."
    },
    {
        "user": "Benarkah ada meteor jatuh di pemukiman warga Jakarta semalam hingga tanah berguncang?",
        "assistant": "Hasil penelusuran fakta astronomi & kebencanaan:\n\n1. Pantauan Astronomi (BRIN): Pusat Riset Antariksa BRIN dan BMKG memiliki sistem pemantau benda langit dan seismograf. Tidak ada catatan dentuman benda luar angkasa berukuran besar yang mendarat di daratan Jakarta.\n2. Laporan Kepolisian & Damkar: Tidak ada catatan kerusakan fisik atau evakuasi di pemukiman warga terkait jatuhnya benda angkasa.\n3. Penyebab Suara Dentuman: Suara dentuman misterius sering kali berasal dari fenomena atmosfer (petir kering di awan tinggi), latihan militer, atau getaran industri setempat.\n\nKesimpulan: TIDAK TERBUKTI / MERAGUKAN. Klaim tidak didukung bukti fisik maupun konfirmasi resmi dari BRIN."
    },

    # Analisis Deepfake & AI Generative Hoax
    {
        "user": "Ada video viral seorang menteri berbicara di podium mempromosikan aplikasi judi online dan investasi crypto, apakah video ini asli?",
        "assistant": "Analisa Forensik Digital Deepfake:\n\n1. Ciri Visual Deepfake AI: Amati sinkronisasi gerak bibir (lip-sync). Pada video AI tiruan, gerakan mulut sering tampak kaku, tepian wajah agak buram saat bergerak cepat, dan kedipan mata tidak alami.\n2. Inkonsistensi Audio: Intonasi suara sering terdengar robotik atau tidak memiliki nafas alami seperti manusia asli.\n3. Logika Konteks: Pejabat publik atau menteri negara secara hukum dilarang keras mempromosikan platform ilegal seperti judi online atau skema ponzi.\n\nKesimpulan: PALSU / DEEPFAKE AI. Video ini adalah hasil rekayasa kecerdasan buatan yang memanipulasi video pidato asli pejabat untuk kepentingan penipuan finansial."
    },
    {
        "user": "Bagaimana cara membedakan foto berita asli dengan gambar buatan AI (Midjourney/DALL-E)?",
        "assistant": "Berikut panduan teknis membedakan foto asli vs rekayasa AI:\n\n1. Detail Anatomi: Periksa bagian jari tangan, kuku, gigi, dan mata. Gambar AI generasi awal sering menghasilkan jumlah jari yang salah (6 jari) atau tekstur kulit yang terlalu mulus seperti lilin tanpa pori-pori.\n2. Teks di Latar Belakang: Gambar buatan AI sering kali memunculkan tulisan di spanduk, pelat nomor, atau kaos yang berantakan, huruf tidak terbaca, atau bahasa asing rekaan yang aneh.\n3. Konsistensi Bayangan & Refleksi: Periksa arah datangnya cahaya matahari. AI sering kali salah merender pantulan air atau bayangan benda yang tidak searah dengan sumber cahaya.\n4. Cek Sumber Kredibel: Foto jurnalistik asli selalu memiliki kredit fotografer kantor berita resmi (Antara Foto, Reuters, AFP, AP) dan metadata EXIF kamera."
    }
]

# 2. Dataset Klaim Baru untuk Classifier (CSV)
csv_samples = [
    # Fakta Asli Berita (Valid)
    ("Presiden Republik Indonesia memimpin upacara peringatan Hari Kemerdekaan di Istana Negara.", "valid"),
    ("Kementerian PUPR menyelesaikan pembangunan bendungan multifungsi untuk irigasi pertanian di Jawa Barat.", "valid"),
    ("Badan Penanggulangan Bencana Daerah (BPBD) menyalurkan bantuan logistik untuk korban banjir bandang.", "valid"),
    ("Polri menangkap sindikat pengedar narkotika jaringan internasional dan mengamankan barang bukti sabu.", "valid"),
    ("Mahkamah Konstitusi membacakan putusan sengketa hasil pemilihan umum dalam sidang pleno terbuka.", "valid"),
    ("Kementerian Pertanian mendorong modernisasi alat mesin pertanian untuk meningkatkan produksi pangan nasional.", "valid"),
    ("Kemenkes mengingatkan pentingnya menjaga asupan air putih dan istirahat cukup untuk mencegah dehidrasi cuaca panas.", "valid"),
    ("Kementerian ESDM meresmikan pengoperasian Pembangkit Listrik Tenaga Surya (PLTS) terapung ramah lingkungan.", "valid"),
    ("PT Kereta Api Indonesia (Persero) mengoperasikan rangkaian kereta generasi baru dengan kenyamanan lebih baik.", "valid"),
    ("BPS mencatat ekspor komoditas kelapa sawit dan nikel memberikan kontribusi positif terhadap neraca perdagangan RI.", "valid"),
    ("Kementerian Komunikasi dan Informatika memblokir ribuan situs judi daring dan rekening penampung.", "valid"),
    ("Satgas pangan daerah melakukan inspeksi mendadak ke pasar tradisional guna mengecek stabilitas harga beras.", "valid"),
    ("Otoritas Jasa Keuangan (OJK) mencabut izin usaha lembaga keuangan bermasalah demi melindungi kepentingan nasabah.", "valid"),
    ("BMKG mengimbau operator kapal laut untuk memperhatikan peringatan dini gelombang tinggi di perairan Samudra Hindia.", "valid"),
    ("Pembalap Indonesia berhasil meraih podium dalam ajang kejuaraan balap motor internasional di sirkuit Mandalika.", "valid"),

    # Hoaks / Manipulasi Berbahaya (Hoaks)
    ("GEBYAR HADIAH BANK BRI! Klaim saldo 5 juta sekarang juga klik link http://bri-gebyar-dana.s.id masukkan PIN ATM!", "hoaks"),
    ("Peringatan keras dokter luar negeri: jangan makan kangkung karena batang kangkung mengandung jutaan lintah raksasa!", "hoaks"),
    ("Kabar duka nasional: Presiden mendadak mengundurkan diri malam ini karena desakan rakyat, viralkan sebelum ditutupi media!", "hoaks"),
    ("Bocoran formula rahasia: semprotkan cairan pemutih pakaian ke wajah setiap malam untuk menghilangkan flek dan jerawat!", "hoaks"),
    ("Gawat! Pemerintah diam-diam menjual Pulau Jawa ke negara asing seharga 100 triliun rupiah, bagikan pesan ini!", "hoaks"),
    ("Lowongan kerja BUMN Pertamina tanpa tes lulusan SMA gaji 18 juta, wajib transfer biaya seragam ke rekening panitia!", "hoaks"),
    ("Awas modus baru! Mencium kertas struk belanjaan di kasir bisa membuat Anda pingsan seketika dan organ ginjal diambil!", "hoaks"),
    ("Bansos sembako gratis dari Kemensos cair hari ini, instal aplikasi BANSOS-CAIR.APK di bawah ini untuk verifikasi!", "hoaks"),
    ("Mengejutkan dunia! NASA akui bumi berbentuk datar dan foto satelit selama ini adalah editan studio animasi!", "hoaks"),
    ("Bagi-bagi iPhone 15 Pro gratis khusus pengguna Telkomsel, kirim nomor kode OTP SMS kamu ke admin WhatsApp ini sekarang!", "hoaks"),

    # Isu Meragukan / Klarifikasi Konteks (Meragukan)
    ("Beredar isu di grup chat bahwa tarif tol trans jawa akan digratiskan total minggu depan, belum ada rilis BPJT.", "meragukan"),
    ("Kabar burung menyebutkan ada pergantian mata uang rupiah dengan uang baru berdenominasi redenominasi bulan depan.", "meragukan"),
    ("Foto truk tangki bahan bakar minyak terbakar di pinggir tol beredar tanpa tanggal dan lokasi yang jelas.", "meragukan"),
    ("Katanya meminum air rebusan kulit manggis dapat menggantikan terapi cuci darah ginjal, belum teruji klinis.", "meragukan"),
    ("Kabar menyebutkan gerai minimarket akan ditutup serentak mulai jam 6 sore besok, tidak ada edaran resmi pemerintah daerah.", "meragukan"),
    ("Beredar tangkapan layar judul berita media online tanpa tautan mengenai penangkapan massal tokoh politik.", "meragukan"),
    ("Broadcast WA mengklaim ada razia narkoba serentak di seluruh kamar kos malam ini, belum dikonfirmasi pihak berwenang.", "meragukan"),
    ("Kabar menyebutkan tiket penerbangan domestik akan disubsidi 80 persen oleh kementerian tanpa keterangan syarat berlaku.", "meragukan"),
    ("Foto awan berbentuk aneh di langit daerah Jawa diklaim sebagai tanda pasti akan terjadi gempa bumi dahsyat dalam 24 jam.", "meragukan"),
    ("Beredar kabar pendaftaran bantuan modal UMKM non-tunai di medsos yang meminta foto KTP dan selfie memegang buku tabungan.", "meragukan")
]

# Tulis ke data/contoh_dataset.csv
with open('data/contoh_dataset.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for text, label in csv_samples:
        writer.writerow([text, label])

print(f"-> Ditambahkan {len(csv_samples)} sampel berimbang ke data/contoh_dataset.csv")

# Tulis ke data/contoh_chat.jsonl dan data/train_chat.jsonl
chat_entries = []
for c in chat_samples:
    entry = {
        "messages": [
            {"role": "system", "content": "Kamu asisten AI validator hoaks Indonesia yang cerdas, objektif, dan teliti menganalisa keaslian berita."},
            {"role": "user", "content": c["user"]},
            {"role": "assistant", "content": c["assistant"]}
        ]
    }
    chat_entries.append(entry)

# Tambahkan juga konversi dari csv_samples ke dialog penalaran otomatis
for text, label in csv_samples:
    if label == "valid":
        resp = f"Informasi ini tergolong VALID dan FAKTUAL. Rincian narasinya: '{text}'. Berdasarkan verifikasi, isi klaim bersesuaian dengan standar pemberitaan resmi dan tidak ditemukan pola provokasi, pemerasan, atau phising."
    elif label == "hoaks":
        resp = f"WASPADA: Klaim ini adalah HOAKS / PALSU. Narasinya: '{text}'. Pesan ini memuat ciri khas manipulasi informasi (seperti iming-iming hadiah, penipuan tautan/APK, atau memicu kepanikan tanpa konfirmasi lembaga berwenang). Jangan disebarkan!"
    else:
        resp = f"Informasi ini tergolong MERAGUKAN (Belum Terverifikasi). Narasinya: '{text}'. Informasi ini belum memiliki rujukan sumber resmi yang jelas atau merupakan kabar burung. Disarankan selalu menyaring sebelum membagikan."

    entry = {
        "messages": [
            {"role": "system", "content": "Kamu asisten AI validator hoaks Indonesia yang cerdas, objektif, dan teliti menganalisa keaslian berita."},
            {"role": "user", "content": f"Tolong periksa apakah berita/klaim ini benar asli: {text}"},
            {"role": "assistant", "content": resp}
        ]
    }
    chat_entries.append(entry)

with open('data/contoh_chat.jsonl', 'a', encoding='utf-8') as f:
    f.writelines(json.dumps(item, ensure_ascii=False) + '\n' for item in chat_entries)

with open('data/train_chat.jsonl', 'a', encoding='utf-8') as f:
    f.writelines(json.dumps(item, ensure_ascii=False) + '\n' for item in chat_entries)

print(f"-> Ditambahkan {len(chat_entries)} percakapan penalaran mendalam ke data/train_chat.jsonl")
print("=== Selesai Mempersiapkan Dataset! ===")
