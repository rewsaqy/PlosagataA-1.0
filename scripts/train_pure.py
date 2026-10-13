"""Training GPT murni Plosagata (causal LM).

CPU sini (tanpa GPU):
  python train_pure.py --size Plosagata_Go --epochs 3

GPU teman RTX 2060 6GB — RENCANA 5 HARI untuk b1:
  Hari 1: setup + build korpus (lihat build_corpus.py)
  Hari 1-4: pretrain di teks Indonesia (korpus besar)
    python train_pure.py --size Plosagata_More --corpus data/korpus.txt --epochs 1 \\
      --batch 1 --grad-accum 32 --fp16 --grad-checkpoint --optim adam8bit --max-tokens 0
  Hari 5: instruction-tune di data chat (317+ contoh)
    python train_pure.py --size Plosagata_More --data data/train_chat.jsonl --epochs 5 \\
      --batch 1 --grad-accum 16 --fp16 --grad-checkpoint --optim adam8bit \\
      --ckpt pure-Plosagata_More-pretrain.pt --out pure-Plosagata_More.pt

Catatan VRAM 6GB: full training fp32 TIDAK muat (~20GB).
Kombinasi fp16 + grad-checkpoint + adam8bit + batch 1 ≈ 4.5-5.5GB: muat mepet.
Jika OOM: turunkan --max-len 1024 -> 512 -> 256.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from torch.utils.data import DataLoader, Dataset

from plosagata import CONFIGS, PureGPT, count_params, normalize_slang
from hoaxcore import WordTokenizer


class ChatLMDataset(Dataset):
    def __init__(self, rows, tok: WordTokenizer, max_len: int):
        self.s, self.tok, self.L = rows, tok, max_len

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        msgs = self.s[i]["messages"]
        text = normalize_slang("\n".join(f"{m['role']}: {m['content']}" for m in msgs))
        ids, _ = self.tok.encode(text)
        ids = ids[:self.L]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


class CorpusDataset(Dataset):
    """Dokumen teks mentah (dipisah baris kosong) -> blok causal LM."""

    def __init__(self, path, tok: WordTokenizer, max_len: int):
        raw = open(path, encoding="utf-8").read()
        docs = [normalize_slang(d.strip()) for d in raw.split("\n\n") if d.strip()]
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
    ap.add_argument("--size", default="Plosagata_Go", choices=list(CONFIGS.keys()))
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
    ap.add_argument("--val-split", type=float, default=0.1, help="Fraksi data untuk validasi (0 = tanpa val).")
    ap.add_argument("--vocab-file", default="", help="Pakai vocab .pt yang sudah ada (wajib saat lanjut pretrain agar embedding cocok).")
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
        raw_docs = [d.strip() for d in open(a.corpus, encoding="utf-8").read().split("\n\n") if d.strip()]
        texts = [normalize_slang(d) for d in raw_docs]
        mode = "pretrain"
    else:
        rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
        texts = [normalize_slang("\n".join(f"{m['role']}: {m['content']}" for m in r["messages"])) for r in rows]
        mode = "instruction"
        print(f"Data chat: {len(rows)} contoh")

    tok = None
    if a.vocab_file:
        vocab = torch.load(a.vocab_file, map_location="cpu")
        tok = WordTokenizer(vocab, max_len=cfg.max_len)
        print(f"Vocab dimuat dari {a.vocab_file}: {len(tok.vocab)}")
    else:
        tok = WordTokenizer.build(texts if not a.corpus else
                                  [t[:2000] for t in texts[:20000]],  # vocab dari sampel korpus
                                  vocab_size=cfg.vocab, max_len=cfg.max_len,
                                  seed_file="data/vocab_seed.txt")
        print(f"Vocab: {len(tok.vocab)}")

    if a.corpus:
        ds = CorpusDataset(a.corpus, tok, cfg.max_len)
    else:
        ds = ChatLMDataset(rows, tok, cfg.max_len)
    if len(ds) == 0:
        raise SystemExit("Dataset kosong — korpus terlalu kecil / semua dokumen < 16 token.")
    import random as _random
    if a.val_split > 0 and len(ds) > 10:
        n_val = max(1, int(len(ds) * a.val_split))
        n_train = len(ds) - n_val
        tr_ds, va_ds = torch.utils.data.random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        print(f"Split: train {n_train} | val {n_val} (seed 42)")
    else:
        tr_ds, va_ds = ds, None
        print(f"Tanpa val (val-split=0 atau data kecil). Train: {len(ds)}")
    tr_ld = DataLoader(tr_ds, batch_size=a.batch, shuffle=True,
                       collate_fn=lambda b: collate_pad(b, tok.pad_id))
    va_ld = None
    if va_ds is not None:
        va_ld = DataLoader(va_ds, batch_size=a.batch, shuffle=False,
                           collate_fn=lambda b: collate_pad(b, tok.pad_id))

    model = PureGPT(cfg).to(device)
    if a.grad_checkpoint:
        model.use_checkpoint = True
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=device))
        print(f"[lanjut dari {a.ckpt}]")
    print(f"Parameter: {count_params(model):,} | mode: {mode}")
    vocab_path = (a.out or f"pure-{a.size}") + ".vocab.pt"
    torch.save(tok.vocab, vocab_path)

    opt = make_optimizer(a.optim, model.parameters(), a.lr)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    t0, seen = time.time(), 0
    out = a.out or f"pure-{a.size}.pt"
    best_out = out.replace(".pt", "-best.pt") if out.endswith(".pt") else out + "-best.pt"
    best_val = float("inf")
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(tr_ld, 1):
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
        # validasi
        val_loss = None
        if va_ld is not None:
            model.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for x, y in va_ld:
                    x, y = x.to(device), y.to(device)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        _, loss = model(x, y)
                    vtot += loss.item()
                    vn += 1
            val_loss = vtot / max(vn, 1)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_out)
                # pasangan vocab best: jangan pisahkan .pt dari .vocab.pt
                import shutil as _shutil
                _shutil.copy(vocab_path, best_out + ".vocab.pt")
                improved = " -> best disimpan"
            else:
                improved = ""
        dt = time.time() - t0
        msg = (f"epoch {ep}/{a.epochs} train_loss={tot/len(tr_ld):.4f} "
               f"tok={seen:,} ({seen/max(dt,1):,.0f} tok/s, {dt/3600:.1f} jam)")
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f} (best {best_val:.4f}){improved}"
        print(msg)
    torch.save(model.state_dict(), out)
    print(f"tersimpan: {out} (best val: {best_out} loss={best_val:.4f})" if va_ld is not None else f"tersimpan: {out}")


if __name__ == "__main__":
    main()
