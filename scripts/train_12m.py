"""Training pipeline resmi Plosagata_12M (12.66M Parameters).

Dirancang khusus untuk pelatihan mandiri (self-training) di CPU laptop / GPU lokal.
Causal Language Model (Decoder-Only Transformer) murni tanpa ketergantungan API luar.

Contoh Penggunaan:
  # Latihan mandiri cepat di CPU (default 5 epoch):
  python scripts/train_12m.py

  # Latihan dengan batch lebih besar / epoch lebih banyak:
  python scripts/train_12m.py --epochs 10 --batch 16 --lr 3e-4

  # Uji coba chat interaktif setelah training:
  python scripts/pure_chat.py --size Plosagata_12M --ckpt checkpoints/pure-Plosagata_12M-best.pt
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hoaxcore import WordTokenizer
from plosagata import CONFIGS, PureGPT, count_params, normalize_slang


class ChatLMDataset(Dataset):
    """Dataset format chat multi-turn untuk causal language modeling."""
    def __init__(self, rows, tok: WordTokenizer, max_len: int = 256):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i].get("messages", [])
        text = normalize_slang("\n".join(f"{m['role']}: {m['content']}" for m in msgs))
        ids, _ = self.tok.encode(text)
        pad = self.tok.pad_id

        # Potong trailing pad: komputasi dinamis sesuai panjang aktual teks
        # (sama seperti train_browser.py — tanpa ini tiap sample diproses 256 tok).
        real_len = len(ids)
        while real_len > 0 and ids[real_len - 1] == pad:
            real_len -= 1
        ids = ids[:max(real_len, 3)]
        ids = ids[:self.max_len]

        # Target token padding bernilai -100 agar diabaikan dalam cross-entropy
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor([t if t != pad else -100 for t in ids[1:]], dtype=torch.long)
        return x, y


def collate_pad(batch, pad_id: int = 0):
    xs, ys = zip(*batch)
    max_l = max(x.size(0) for x in xs)
    xb = torch.stack([F.pad(x, (0, max_l - x.size(0)), value=pad_id) for x in xs])
    yb = torch.stack([F.pad(y, (0, max_l - y.size(0)), value=-100) for y in ys])
    return xb, yb


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    """Learning rate scheduler cosine dengan linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_smoke_test(model, tok, device, max_new: int = 45):
    """Uji coba respon generasi model setelah selesai latihan."""
    print("\n" + "=" * 65)
    print("🤖 SMOKE TEST GENERASI MODEL PLOSAGATA 12M")
    print("=" * 65)
    model.eval()
    test_prompts = [
        "user: Halo Sagi!\nassistant:",
        "user: Kamu bisa bantu apa aja?\nassistant:",
        "user: Bener gak vaksin covid mengandung microchip?\nassistant:",
    ]
    ign = {i for t, i in tok.vocab.items() if t.startswith("<pad") or t in ("<unk>", "<cls>")}
    eos = {tok.sep_id}

    with torch.no_grad():
        for p in test_prompts:
            ids, _ = tok.encode(p)
            real = [i for i in ids if i != tok.pad_id][:64]
            t0 = time.time()
            out = model.generate(
                torch.tensor([real]).to(device),
                max_new=max_new,
                temperature=0.7,
                top_k=40,
                no_repeat_ngram=3,
                eos_ids=eos,
                ignore_ids=ign
            )[0].tolist()
            dt = max(time.time() - t0, 1e-4)
            reply = tok.decode(out[len(real):]).strip()
            speed = len(out[len(real):]) / dt
            print(f"\n[Prompt]: {p.strip()}")
            print(f"[Jawaban]: {reply if reply else '(model menghasilkan EOS seketika)'}")
            print(f"[Kecepatan]: {speed:.1f} token/detik")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Trainer Mandiri Plosagata_12M Causal LM")
    parser.add_argument("--data", default="data/train_chat.jsonl", help="Dataset jsonl format chat")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch training (default: 5)")
    parser.add_argument("--batch", type=int, default=8, help="Ukuran batch per step (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate awal (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Bobot L2 regularization")
    parser.add_argument("--val-split", type=float, default=0.1, help="Rasio data validasi (default: 0.1)")
    parser.add_argument("--max-len", type=int, default=256, help="Panjang maksimum konteks token")
    parser.add_argument("--out-dir", default="checkpoints", help="Folder output checkpoint")
    parser.add_argument("--name", default="pure-Plosagata_12M", help="Nama basis checkpoint")
    parser.add_argument("--resume", default="", help="Path checkpoint untuk melanjutkan latihan")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Perangkat komputasi")
    parser.add_argument("--fp16", action="store_true", help="Aktifkan mixed precision FP16 (khusus CUDA)")
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Jalankan smoke test setelah training")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    use_fp16 = args.fp16 and device == "cuda"

    print("=" * 65)
    print("🚀 PLOSAGATA 12M — PELATIHAN MANDIRI (SELF-TRAINING)")
    print("=" * 65)
    print(f"Perangkat Komputasi : {device.upper()}" + (" (FP16 AMP)" if use_fp16 else ""))
    print(f"File Dataset        : {args.data}")
    print(f"Epochs              : {args.epochs}")
    print(f"Batch Size          : {args.batch}")
    print(f"Learning Rate       : {args.lr}")

    # Konfigurasi Plosagata_12M
    cfg = CONFIGS["Plosagata_12M"]
    if args.max_len:
        cfg.max_len = args.max_len

    # Load dataset
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset {args.data} tidak ditemukan!")

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Jumlah Contoh Chat  : {len(rows):,} sampel")

    # Siapkan Tokenizer BPE / Subword
    texts = [normalize_slang("\n".join(f"{m['role']}: {m['content']}" for m in r.get("messages", []))) for r in rows]
    tok = WordTokenizer.build(texts, vocab_size=cfg.vocab, max_len=cfg.max_len, seed_file="data/vocab_seed.txt")
    print(f"Ukuran Kosakata     : {len(tok.vocab):,} token")

    # Inisialisasi Model
    model = PureGPT(cfg).to(device)
    total_params = count_params(model)
    print(f"Arsitektur Model    : {cfg.layers} layer, {cfg.heads} head, d_model={cfg.dim}, ff={cfg.ff}")
    print(f"Total Parameter     : {total_params:,} (12.66M)")

    # Resume dari checkpoint jika diberikan
    if args.resume and Path(args.resume).exists():
        print(f"Melanjutkan dari    : {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # Split train & val
    full_ds = ChatLMDataset(rows, tok, max_len=cfg.max_len)
    if args.val_split > 0 and len(full_ds) > 10:
        n_val = max(1, int(len(full_ds) * args.val_split))
        n_train = len(full_ds) - n_val
        tr_ds, va_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        print(f"Pembagian Data      : {n_train:,} Train | {n_val:,} Validasi (10%)")
    else:
        tr_ds, va_ds = full_ds, None
        print(f"Pembagian Data      : {len(full_ds):,} Train (tanpa validasi)")

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_pad(b, tok.pad_id)
    )
    va_loader = None
    if va_ds is not None:
        va_loader = DataLoader(
            va_ds, batch_size=args.batch, shuffle=False,
            collate_fn=lambda b: collate_pad(b, tok.pad_id)
        )

    # Output paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.name}-best.pt"
    latest_path = out_dir / f"{args.name}.pt"
    vocab_path = out_dir / f"{args.name}.vocab.pt"
    best_vocab_path = out_dir / f"{args.name}-best.pt.vocab.pt"
    tok_json_path = out_dir / f"{args.name}.tokenizer.json"
    best_tok_json_path = out_dir / f"{args.name}-best.tokenizer.json"

    # Simpan vocab awal
    torch.save(tok.vocab, vocab_path)
    torch.save(tok.vocab, best_vocab_path)
    try:
        tok.save(str(tok_json_path))
        tok.save(str(best_tok_json_path))
    except Exception:
        pass

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(tr_loader) * args.epochs
    warmup_steps = max(10, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    print("-" * 65)
    print("Mulai proses pelatihan...\n")

    t_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        tokens_seen = 0
        t_epoch_start = time.time()

        for step, (x, y) in enumerate(tr_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                _, loss = model(x, y)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_train_loss += loss.item()
            tokens_seen += (y != -100).sum().item()

        avg_train_loss = total_train_loss / len(tr_loader)

        # Evaluasi Validasi
        val_loss = None
        if va_loader is not None:
            model.eval()
            total_val_loss, val_count = 0.0, 0
            with torch.no_grad():
                for vx, vy in va_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                        _, vloss = model(vx, vy)
                    total_val_loss += vloss.item()
                    val_count += 1
            val_loss = total_val_loss / max(val_count, 1)

        t_epoch = max(time.time() - t_epoch_start, 1e-4)
        current_lr = scheduler.get_last_lr()[0]
        tok_speed = tokens_seen / t_epoch

        status_msg = f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_train_loss:.4f}"
        if val_loss is not None:
            ppl = math.exp(min(val_loss, 20))
            status_msg += f" | Val Loss: {val_loss:.4f} (PPL: {ppl:.2f})"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                status_msg += " 🌟 BEST"
        status_msg += f" | {tok_speed:,.0f} tok/s | LR: {current_lr:.2e}"
        print(status_msg)

    # Simpan checkpoint akhir
    torch.save(model.state_dict(), latest_path)

    total_time = time.time() - t_start
    print("-" * 65)
    print(f"🎉 Pelatihan Selesai dalam {total_time:.1f} detik ({total_time/60:.1f} menit)!")
    print(f"💾 Checkpoint Terbaik : {best_path} (Val Loss: {best_val_loss:.4f})")
    print(f"💾 Checkpoint Terakhir: {latest_path}")
    print(f"💾 File Tokenizer     : {vocab_path} & {tok_json_path}")

    # Smoke test generasi
    if args.smoke_test:
        test_model = PureGPT(cfg).to(device)
        test_ckpt = best_path if best_path.exists() else latest_path
        test_model.load_state_dict(torch.load(test_ckpt, map_location=device))
        run_smoke_test(test_model, tok, device)


if __name__ == "__main__":
    main()
