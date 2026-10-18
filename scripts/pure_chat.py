"""Chat CLI Sagi — mesin yang SAMA dengan web (paket `sagi`).

  python3 scripts/pure_chat.py --size Plosagata_Go --prompt "Halo bang!"
  python3 scripts/pure_chat.py --size Plosagata_12M --ckpt checkpoints/pure-Plosagata_12M-best.pt

Di terminal: /help, /models, /model <nama>, /status — ketik 'keluar' untuk berhenti.
Format prompt & memori multi-turn kini identik dengan web (sagi.prompt/session).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from plosagata import CONFIGS, PARAMS
from sagi import SagiEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="Plosagata_More", choices=list(CONFIGS.keys()))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--vocab-file", default="", help="Wajib jika ckpt dari train_pure --out X.pt (pasangannya X.pt.vocab.pt). Tanpa ini output jadi sampah.")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.7, help="<1 fokus, >1 liar")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--no-repeat", type=int, default=3, help="blokir frasa berulang (0=mati)")
    ap.add_argument("--pure-think", action="store_true",
                    help="Tanpa template: SEMUA pesan (termasuk halo/off-topic) "
                         "lewat tokenizer -> model -> decode. Butuh model TERLATIH, "
                         "kalau tidak output ngawur.")
    a = ap.parse_args()

    if a.pure_think:
        os.environ["SAGI_PURE_THINK"] = "1"
    os.environ["SAGI_TEMP"] = str(a.temp)
    os.environ["SAGI_TOPK"] = str(a.top_k)
    os.environ["SAGI_NOREPEAT"] = str(a.no_repeat)

    if a.size in ("Plosagata_More", "Plosagata_Next"):
        print("[model besar di CPU lambat; untuk serius pakai GPU + checkpoint hasil training]")

    eng = SagiEngine(size=a.size, ckpt=a.ckpt)

    # --vocab-file: vocab mentah tanpa backend BPE (format lama train_pure --out X.pt)
    if a.vocab_file and os.path.exists(a.vocab_file):
        import torch
        from hoaxcore import WordTokenizer
        vocab = torch.load(a.vocab_file, map_location="cpu")
        eng.tok = WordTokenizer(vocab, max_len=eng.cfg.max_len)
        eng.use_pieces = True
        eng.ign = {i for t, i in eng.tok.vocab.items()
                   if t.startswith("<pad") or t in ("<unk>", "<cls>")}
        eng.eos = {eng.tok.sep_id}
        print(f"[vocab dimuat: {a.vocab_file}]")

    if eng.trained:
        print(f"[ckpt: {a.ckpt} — bobot dimuat saat balasan pertama]")
    else:
        print("[tanpa ckpt] mode template heuristik (bobot saraf belum tersedia).")

    sess = "cli"

    def cmd_terminal(text: str) -> str:
        parts = text.split()
        cmd = parts[0].lower()
        if cmd in ("/help", "/bantuan"):
            return ("Perintah: /models (daftar), /model <nama> "
                    "(restart dengan --size itu), /status")
        if cmd == "/models":
            return "\n".join(
                f"{'●' if n == a.size else '○'} {n} "
                f"({PARAMS.get(n,0)/1e6:.1f} jt)" for n in CONFIGS)
        if cmd in ("/model", "/pakai", "/ganti"):
            mau = " ".join(parts[1:])
            if mau in CONFIGS and mau != a.size:
                return (f"Restart dengan: --size {mau} "
                        f"(proses ini jalan {a.size})")
            return f"Sudah pakai {a.size}. Lihat: /models"
        if cmd == "/status":
            return f"Model: {a.size} | topik sesi: {eng.sesi.topik(sess) or '-'}"
        return f"Perintah tak dikenal: {cmd}. Ketik /help."

    def answer(prompt: str) -> str:
        if prompt.strip().startswith("/"):
            return cmd_terminal(prompt.strip())
        r = eng.reply(prompt, max_new=a.max_new, session_id=sess)
        return r.get("reply", "")

    if a.prompt:
        print(answer(a.prompt))
        return
    print(f"Plosagata-{a.size} siap (murni kodinganmu). Ketik 'keluar' untuk berhenti.")
    while True:
        try:
            u = input("kamu> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("keluar", "exit", "quit"):
            break
        if u:
            print("ai> " + answer(u) + "\n")


if __name__ == "__main__":
    main()
