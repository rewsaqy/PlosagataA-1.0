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

from plosagata import (CONFIGS, SAGI_HELLO, PureGPT, is_news_topic,
                       redirect_reply)
from tokenizer import WordTokenizer


def build_tokenizer(vocab_size: int):
    seeds = ["halo", "hai", "pagi", "bang", "berita", "hoaks", "bener", "gak",
             "giveaway", "klik", "link", "covid", "vaksin", "pemerintah", "cek"]
    return WordTokenizer.build(seeds, vocab_size=vocab_size, max_len=256)


def ignore_ids_for(tok: WordTokenizer) -> set[int]:
    # jangan pernah generate token sampah padding/unknown
    return {i for t, i in tok.vocab.items()
            if t.startswith("<pad") or t in ("<unk>", "<cls>", "<sep>")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(CONFIGS.keys()))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=40)
    a = ap.parse_args()

    cfg = CONFIGS[a.size]
    if a.size == "b1":
        print("size b1 (~900M) jangan di-load di CPU sini — pindah ke device GPU. "
              "Gunakan --size tiny untuk buktikan arsitektur nyala.")
        return

    tok = build_tokenizer(cfg.vocab)
    ign = ignore_ids_for(tok)
    model = PureGPT(cfg)
    if a.ckpt and os.path.exists(a.ckpt):
        try:
            model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
            print(f"[ckpt dimuat: {a.ckpt}]")
        except RuntimeError as e:
            print(f"[ckpt lama tak cocok dengan arsitektur baru, pakai bobot acak: {e}]")
    else:
        print("[bobot ACAK — output awal ngawur itu normal sebelum training]")

    def answer(prompt: str) -> str:
        topic = is_news_topic(prompt)
        if topic == "offtopic":
            return redirect_reply()
        if topic == "greeting":
            return SAGI_HELLO
        ids, _ = tok.encode(prompt)
        real = [i for i in ids if i != tok.pad_id][:64]
        out = model.generate(torch.tensor([real]), max_new=a.max_new,
                             ignore_ids=ign)[0].tolist()
        # BPE: decode via tokenizer (mapping id->kata mentah menghasilkan byte sampah)
        return tok.decode(out[len(real):])

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
