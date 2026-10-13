"""Client Sagi jarak jauh — ngobrol dengan Sagi di komputer teman via internet.

Setup sekali (lihat README bagian Sagi Remote):
  export SAGI_URL=http://IP-TEMAN:5000 SAGI_TOKEN=rahasia-kalian

Pakai:
  python scripts/sagi_remote.py                  (interaktif)
  python scripts/sagi_remote.py "Sagi, halo!"    (sekali)
"""
import argparse
import json
import os
import sys
import urllib.request

URL = os.getenv("SAGI_URL", "http://127.0.0.1:5000")
TOKEN = os.getenv("SAGI_TOKEN", "")


def tanya(pesan: str) -> dict:
    req = urllib.request.Request(
        URL.rstrip("/") + "/api/chat",
        data=json.dumps({"message": pesan}).encode(),
        headers={"Content-Type": "application/json", "X-Sagi-Token": TOKEN},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesan", nargs="?", default="")
    a = ap.parse_args()

    if a.pesan:
        j = tanya(a.pesan)
        print(j.get("reply", j.get("error")) if j.get("ok") else f"Gagal: {j}")
        return

    print(f"Sagi remote ({URL}). Ketik 'keluar' untuk berhenti.")
    while True:
        try:
            u = input("kamu> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("keluar", "exit", "quit"):
            break
        if not u:
            continue
        try:
            j = tanya(u)
            if not j.get("ok"):
                print("Gagal:", j)
                continue
            print("sagi> " + j["reply"])
            if j.get("verdict"):
                v = j["verdict"]
                print(f"      [12M: {v['label']} skor {v['skor']}]")
        except Exception as e:
            print(f"Tak tersambung ({e}). Cek SAGI_URL / internet / server teman.")


if __name__ == "__main__":
    sys.exit(main())
