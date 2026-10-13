"""Uji penerimaan topik multi-turn: tempel berita -> tanya lanjutan.

Syarat lolos:
1. Follow-up dalam sesi -> topic followup + balasan "Merujuk berita tadi".
2. Follow-up TANPA sesi -> perilaku lama (news standalone, bukan followup).
3. Bukan rujukan walau ada topik ("Buatkan kode python") -> tetap offtopic.
4. Berita ber-keyword ("Gimana cara bedakan berita hoaks?") -> news (topik baru).
5. Ganti topik -> follow-up merujuk topik BARU.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
os.chdir(ROOT)

from chatbot import SagiEngine


def main():
    e = SagiEngine()
    s = "uji-topik"

    r1 = e.reply("Bener gak BLT cair bulan ini?", session_id=s)
    assert r1["topic"] == "news", r1["topic"]
    assert r1["verdict"], "berita harus ada verdict"

    r2 = e.reply("kapan cairnya?", session_id=s)
    assert r2["topic"] == "followup", r2["topic"]
    assert "Merujuk berita tadi" in r2["reply"], r2["reply"][:120]
    assert r2["verdict"], "follow-up harus ada verdict klaim"
    print("1-2 OK: follow-up merujuk topik,", r2["topic"])

    r3 = e.reply("kapan cairnya?")
    assert r3["topic"] != "followup", r3["topic"]
    print("3 OK: tanpa sesi bukan follow-up,", r3["topic"])

    r4 = e.reply("Buatkan kode python", session_id=s)
    assert r4["topic"] == "offtopic", r4["topic"]
    print("4 OK: non-rujukan tetap ditolak")

    r5 = e.reply("Gimana cara bedakan berita hoaks?", session_id=s)
    assert r5["topic"] == "news", r5["topic"]
    r6 = e.reply("kasih contohnya", session_id=s)
    assert r6["topic"] == "followup", r6["topic"]
    assert "Merujuk berita tadi" in r6["reply"]
    print("5-6 OK: topik baru + follow-up mengikutinya")

    r7 = e.reply("Sagi", session_id=s)
    assert r7["topic"] == "greeting", r7["topic"]
    print("7 OK: sapaan tak merusak topik")

    print("EVAL-TOPIK LULUS")


if __name__ == "__main__":
    main()
