"""Sitasi sumber di jawaban: normalisasi + blok teks utk API/CLI.

UI web punya kartu verdict + pill sumber sendiri; blok teks ini memastikan
jawaban tetap bersitasi lewat API biasa / CLI / salinan teks.
"""
JENIS_URUT = (("cek_fakta", "CekFakta"), ("wikipedia", "Wikipedia"), ("web", "Web"))


def kumpulkan_sumber(sumber: dict | None, maks: int = 3) -> list[dict]:
    """Normalisasi struktur cari_sumber() menjadi daftar ringkas & stabil."""
    hasil: list[dict] = []
    for kunci, jenis in JENIS_URUT:
        for s in (sumber or {}).get(kunci, []) or []:
            if not isinstance(s, dict):
                continue
            judul = str(s.get("judul") or s.get("title") or "").strip()
            url = str(s.get("url") or "").strip()
            if not (judul or url):
                continue
            hasil.append({"jenis": jenis, "judul": judul or url, "url": url})
            if len(hasil) >= maks:
                return hasil
    return hasil


def blok_sitasi(verdict: dict | None, sumber: dict | None, maks: int = 3) -> str:
    """Blok teks ringkas (verdict + sumber) utk ditempel di akhir jawaban.

    Kosong bila tak ada info — jangan menempel blok kosong.
    """
    baris: list[str] = []
    label = (verdict or {}).get("label") or ""
    if label:
        skor = (verdict or {}).get("skor")
        if (verdict or {}).get("lantai"):
            # Skor "lantai" (mis. 40 saat belum ada bukti) BUKAN hasil ukur —
            # jangan tampilkan angka palsu, tulis keadaannya apa adanya.
            ekor = " · belum terkonfirmasi (bukti belum cukup)"
        else:
            ekor = f" · skor {int(skor)}/100" if isinstance(skor, (int, float)) else ""
        baris.append(f"🔎 Verdict: {label.upper()}{ekor}")
    nama = [s["judul"] for s in kumpulkan_sumber(sumber, maks=maks)]
    if nama:
        baris.append("📚 Sumber: " + " · ".join(nama))
    return ("\n\n---\n" + "\n".join(baris)) if baris else ""
