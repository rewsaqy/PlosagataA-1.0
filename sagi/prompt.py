"""Pembangun prompt & pembersih output chat Sagi.

Format WAJIB sama persis dengan data training (lihat scripts/train_12m.py):
    normalize_slang("\\n".join(f"{role}: {content}"))
Perhatikan: normalisasi diterapkan ke SELURUH transkrip — termasuk label peran.
Akibatnya label training bukan "user:" melainkan "konsumen:" (dan "system:" ->
"sistem:"). Prompt inferensi harus mengikuti hasil normalisasi yang sama, kalau
tidak model melihat teks di luar distribusi latihan.

Semua baris training diawali baris `system` (92% memakai SISTEM_DEFAULT di
bawah), jadi prompt inferensi juga menyertakannya. Prompt diakhiri
"assistant:" (tanpa <sep>) supaya model melanjutkan balasan.

Catatan B-35: sebelumnya web/chatbot.py & scripts/pure_chat.py mengirim teks
mentah tanpa bingkai peran — model hasil training tak akan tahu ia harus
menjawab sebagai assistant. Modul ini memperbaikinya; <sep> juga dipastikan
TIDAK ikut masuk prompt (lih. engine.SagiEngine._tokenisasi).
"""
import re

from plosagata import normalize_slang

PERAN_USER = "user"
PERAN_ASSISTANT = "assistant"

# Baris `system` yang dipakai 92% baris training (data/train_chat.jsonl).
SISTEM_DEFAULT = (
    "Kamu Sagi, asisten AI cerdas pelurus hoaks dan analis disinformasi digital "
    "berbasis Plosagata_Browser (25M Core)."
)

# Label peran SETELAH normalisasi (dipakai pembersih output).
LABEL_KONSUMEN = normalize_slang(PERAN_USER)      # "konsumen"
LABEL_SISTEM = normalize_slang("system")          # "sistem"

MAKS_KATA_PESAN = 150    # cap kata utk pesan user (kepala, konsisten truncation training)
MAKS_KATA_BALASAN = 100  # cap kata utk balasan lama di riwayat
ELIPSIS = "…"


def rapikan_isi(teks: str, maks_kata: int) -> str:
    """Ratakan newline -> spasi (jaga struktur peran) + potong kepala bila kepanjangan."""
    kata = str(teks or "").split()
    if not kata:
        return ""
    if len(kata) > maks_kata:
        kata = kata[:maks_kata] + [ELIPSIS]
    return " ".join(kata)


def bangun_prompt(pesan_user: str, riwayat=None, sistem: str | None = SISTEM_DEFAULT) -> str:
    """Transkrip siap-generate untuk model (dialiri normalize_slang seperti training).

    riwayat: list[(pesan_user, balasan_assistant)] urut lama -> baru.
    sistem: None = tanpa baris system (khusus eksperimen).
    Hasil selalu berakhir dengan "assistant:" supaya model melanjutkan balasan.
    """
    baris = []
    if sistem:
        baris.append(f"system: {rapikan_isi(sistem, MAKS_KATA_BALASAN)}")
    for giliran in (riwayat or []):
        try:
            u, a = giliran
        except (TypeError, ValueError):
            continue
        isi_u = rapikan_isi(u, MAKS_KATA_PESAN)
        if isi_u:
            baris.append(f"{PERAN_USER}: {isi_u}")
        isi_a = rapikan_isi(a, MAKS_KATA_BALASAN)
        if isi_a:
            baris.append(f"{PERAN_ASSISTANT}: {isi_a}")
    pesan = rapikan_isi(pesan_user, MAKS_KATA_PESAN)
    baris.append(f"{PERAN_USER}: {pesan}")
    teks = "\n".join(baris) + f"\n{PERAN_ASSISTANT}:"
    return normalize_slang(teks)


_PENANDA_BOCOR = (
    "\nuser:", "\nassistant:", f"\n{LABEL_KONSUMEN}:", f"\n{LABEL_SISTEM}:",
    " user:", " assistant:", f" {LABEL_KONSUMEN}:", f" {LABEL_SISTEM}:",
)


def bersihkan_output(teks: str) -> str:
    """Bersihkan keluaran model: potong penanda peran bocor, rapikan spasi."""
    if not teks:
        return ""
    t = str(teks).replace("\r", "\n")
    potong = len(t)
    for penanda in _PENANDA_BOCOR:
        i = t.find(penanda)
        if 0 <= i < potong:
            potong = i
    t = t[:potong]
    t = t.replace("\ufffd", "")  # sisa byte utf-8 yang terpotong di perbatasan token
    t = re.sub(rf"^(?:user|assistant|{LABEL_KONSUMEN}|{LABEL_SISTEM})\s*:\s*", "",
               t.strip(), flags=re.I)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t.strip()
