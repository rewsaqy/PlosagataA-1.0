"""Chat dengan GPT murni milikmu (bobot acak -> awalnya ngawur, wajar).

  python3 pure_chat.py --size tiny --prompt "Halo bang!"
  python3 pure_chat.py --size tiny   (interaktif)

Setelah training (lihat train_pure.py), muat checkpoint:
  python3 pure_chat.py --size tiny --ckpt pure-tiny.pt --prompt "Halo!"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from gpt_pure import CONFIGS, PureGPT
from tokenizer import WordTokenizer


def build_tokenizer(vocab_size: int):
    # tokenizer kata kecil untuk demo murni (tanpa internet)
    seeds = ["halo", "hai", "pagi", "bang", "berita", "hoaks", "bener", "gak",
             "giveaway", "klik", "link", "covid", "vaksin", "pemerintah", "cek"]
    tok = WordTokenizer.build(seeds, vocab_size=vocab_size, max_len=256)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(CONFIGS.keys()))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=40)
    a = ap.parse_args()

    cfg = CONFIGS[a.size]
    if a.size == "b1":
        print("size b1 (~1B) jangan di-load di CPU sini — pindah ke device GPU. "
              "Gunakan --size tiny untuk buktikan arsitektur nyala.")
        return

    tok = build_tokenizer(cfg["vocab"])
    model = PureGPT(**cfg)
    if a.ckpt and os.path.exists(a.ckpt):
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
        print(f"[ckpt dimuat: {a.ckpt}]")
    else:
        print("[bobot ACAK — output awal ngawur itu normal sebelum training]")

    def answer(prompt: str) -> str:
        ids, _ = tok.encode(prompt)
        # buang padding untuk generate
        real = [i for i in ids if i != tok.pad_id][:64]
        inp = torch.tensor([real])
        out = model.generate(inp, max_new=a.max_new)[0].tolist()
        words = [tok.inv.get(i, "?") for i in out[len(real):]]
        return " ".join(words)

    if a.prompt:
        print(answer(a.prompt))
        return
    print(f"PureGPT-{a.size} siap (murni kodinganmu). Ketik 'keluar' untuk berhenti.")
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
