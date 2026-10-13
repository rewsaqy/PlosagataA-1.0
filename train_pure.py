"""Training GPT murni Plosagata (causal LM).

CPU sini (tanpa GPU):
  python train_pure.py --size tiny --epochs 3

GPU teman RTX 2060 6GB — RENCANA 5 HARI untuk b1:
  Hari 1: setup + build korpus (lihat build_corpus.py)
  Hari 1-4: pretrain di teks Indonesia (korpus besar)
    python train_pure.py --size b1 --corpus data/korpus.txt --epochs 1 \\
      --batch 1 --grad-accum 32 --fp16 --grad-checkpoint --optim adam8bit --max-tokens 0
  Hari 5: instruction-tune di data chat (317+ contoh)
    python train_pure.py --size b1 --data data/train_chat.jsonl --epochs 5 \\
      --batch 1 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \\
      --ckpt pure-b1-pretrain.pt --out pure-b1.pt

Catatan VRAM 6GB: full training fp32 TIDAK muat (~20GB).
Kombinasi fp16 + grad-checkpoint + adam8bit + batch 1 ≈ 4.5-5.5GB: muat mepet.
Jika OOM: turunkan --max-len 1024 -> 512 -> 256.
"""
import argparse
import json
import time

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


class CorpusDataset(Dataset):
    """Dokumen teks mentah (dipisah baris kosong) -> blok causal LM."""

    def __init__(self, path, tok: WordTokenizer, max_len: int):
        raw = open(path, encoding="utf-8").read()
        docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
        self.blocks = []
        for d in docs:
            ids, _ = tok.encode(d)
            ids = [i for i in ids if i != tok.pad_id]
            for s in range(0, max(1, len(ids) - 1), max_len):
                chunk = ids[s:s + max_len]
                if len(chunk) > 16:
                    self.blocks.append(chunk)
        print(f"Korpus: {len(docs)} dokumen -> {len(self.blocks)} blok")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        b = self.blocks[i]
        return torch.tensor(b[:-1]), torch.tensor(b[1:])


def collate_pad(batch, pad_id: int = 0):
    xs, ys = zip(*batch)
    L = max(x.size(0) for x in xs)
    import torch.nn.functional as F
    xb = torch.stack([F.pad(x, (0, L - x.size(0)), value=pad_id) for x in xs])
    yb = torch.stack([F.pad(y, (0, L - y.size(0)), value=-100) for y in ys])
    return xb, yb


def make_optimizer(name, params, lr):
    if name == "adam8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise SystemExit("pip install bitsandbytes untuk --optim adam8bit")
        return bnb.optim.Adam8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(CONFIGS.keys()))
    ap.add_argument("--data", default="data/contoh_chat.jsonl", help="JSONL chat (instruction)")
    ap.add_argument("--corpus", default="", help="TXT korpus (pretrain). Menang atas --data.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", help="Wajib di GPU kecil.")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Hemat VRAM besar.")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adam8bit"])
    ap.add_argument("--max-len", type=int, default=0, help="Override CFG.max_len (kecilkan jika OOM).")
    ap.add_argument("--ckpt", default="", help="Lanjutkan dari checkpoint.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = CONFIGS[a.size]
    if a.max_len:
        cfg.max_len = a.max_len

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    use_fp16 = a.fp16 and device == "cuda"
    if a.fp16 and device != "cuda":
        print("[fp16 diabaikan: tidak ada GPU]")

    if a.corpus:
        texts = [d.strip() for d in open(a.corpus, encoding="utf-8").read().split("\n\n") if d.strip()]
        mode = "pretrain"
    else:
        rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
        texts = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"]) for r in rows]
        mode = "instruction"
        print(f"Data chat: {len(rows)} contoh")

    tok = WordTokenizer.build(texts if not a.corpus else
                              [t[:2000] for t in texts[:20000]],  # vocab dari sampel korpus
                              vocab_size=cfg.vocab, max_len=cfg.max_len)
    print(f"Vocab: {len(tok.vocab)}")

    if a.corpus:
        ds = CorpusDataset(a.corpus, tok, cfg.max_len)
    else:
        ds = ChatLMDataset(rows, tok, cfg.max_len)
    if len(ds) == 0:
        raise SystemExit("Dataset kosong — korpus terlalu kecil / semua dokumen < 16 token.")
    ld = DataLoader(ds, batch_size=a.batch, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, tok.pad_id))

    model = PureGPT(cfg).to(device)
    if a.grad_checkpoint:
        model.use_checkpoint = True
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=device))
        print(f"[lanjut dari {a.ckpt}]")
    print(f"Parameter: {count_params(model):,} | mode: {mode}")
    torch.save(tok.vocab, (a.out or f"pure-{a.size}") + ".vocab.pt")

    opt = make_optimizer(a.optim, model.parameters(), a.lr)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    model.train()
    t0, seen = time.time(), 0
    for ep in range(1, a.epochs + 1):
        tot = 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(ld, 1):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                _, loss = model(x, y)
            loss = loss / a.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
            tot += loss.item() * a.grad_accum
            seen += x.numel()
        dt = time.time() - t0
        print(f"epoch {ep}/{a.epochs} loss={tot/len(ld):.4f} "
              f"tok={seen:,} ({seen/max(dt,1):,.0f} tok/s, {dt/3600:.1f} jam)")
    out = a.out or f"pure-{a.size}.pt"
    torch.save(model.state_dict(), out)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
