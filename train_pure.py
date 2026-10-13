"""Training GPT murni Plosagata (causal LM) dengan data chat + berita.

CPU sini (tanpa GPU):
  python3 train_pure.py --size tiny --epochs 3
  -> menghasilkan pure-tiny.pt (bukti training jalan)

Device GPU lain (untuk small / b1 ~900M):
  copy folder ini ke sana, lalu:
  python train_pure.py --size small --epochs 5 --batch 8
  python train_pure.py --size b1 --epochs 3 --batch 1 --grad-accum 16
  butuh GPU 24GB+ untuk b1, pakai fp16 + gradient checkpointing.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset

from plosagata import CONFIGS, PureGPT, count_params
from tokenizer import WordTokenizer


class ChatLMDataset(Dataset):
    def __init__(self, rows, tok: WordTokenizer, max_len: int):
        self.s, self.tok, self.L = rows, tok, max_len

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        msgs = self.s[i]["messages"]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        ids, _ = self.tok.encode(text)
        ids = ids[:self.L]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(CONFIGS.keys()))
    ap.add_argument("--data", default="data/contoh_chat.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = CONFIGS[a.size]
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    print(f"Data: {len(rows)} contoh chat")

    texts = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"]) for r in rows]
    tok = WordTokenizer.build(texts, vocab_size=cfg.vocab, max_len=cfg.max_len)
    print(f"Vocab: {len(tok.vocab)}")

    ld = DataLoader(ChatLMDataset(rows, tok, cfg.max_len), batch_size=a.batch, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = PureGPT(cfg).to(device)
    print(f"Parameter: {count_params(model):,}")
    torch.save(tok.vocab, (a.out or f"pure-{a.size}") + ".vocab.pt")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    model.train()
    for ep in range(1, a.epochs + 1):
        tot = 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(ld, 1):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / a.grad_accum).backward()
            if step % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            tot += loss.item()
        print(f"epoch {ep}/{a.epochs} loss={tot/len(ld):.4f}")
    out = a.out or f"pure-{a.size}.pt"
    torch.save(model.state_dict(), out)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
