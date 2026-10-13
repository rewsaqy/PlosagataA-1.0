"""Training / fine-tune model hoaks 12M."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import HoaxDataset, load_csv
from model import TinyHoaxTransformer, count_params
from tokenizer import WordTokenizer


def train(args):
    texts, labels = load_csv(args.data)
    print(f"Data: {len(texts)} sampel")

    tok = WordTokenizer.build(texts, vocab_size=args.vocab_size, max_len=args.max_len)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tok.save(f"{args.out}/vocab.json")

    tr_t, va_t, tr_l, va_l = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    tr_ds = HoaxDataset(tr_t, tr_l, tok)
    va_ds = HoaxDataset(va_t, va_l, tok)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyHoaxTransformer(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        num_classes=3,
    ).to(device)
    print(f"Parameter: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for ids, mask, y in tr_ld:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(ids, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()

        # validasi
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for ids, mask, y in va_ld:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                pred = model(ids, mask).argmax(-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {ep}/{args.epochs} loss={tot/len(tr_ld):.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), f"{args.out}/best.pt")
            print(f"  -> saved ({acc:.4f})")

    torch.save(model.state_dict(), f"{args.out}/last.pt")
    print(f"Selesai. Best acc: {best:.4f}. Model di {args.out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/contoh_dataset.csv")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--max-len", type=int, default=128)
    train(p.parse_args())
