"""GPT murni dari nol — 100% kode sendiri, PyTorch saja.
Tanpa transformers, tanpa Ollama, tanpa download model internet.

Arsitektur: decoder-only causal (mirip GPT-2):
  token_emb + pos_emb -> N x (LN->MHA causal->residu, LN->FFN GELU->residu) -> LN -> lm_head

Ukuran:
  tiny  ~7M   (untuk tes nyala di CPU sini)
  small ~100M (latihan serius, butuh GPU)
  b1    ~1B   (targetmu, training di device GPU lain)

Bobot awal acak -> output awal ngawur (wajar, semua GPT dari nol begitu).
Bisa ngobrol beneran SETELAH training dengan data chat Indonesia.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # causal: is_causal=True (torch >= 2.0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(),
            nn.Linear(ff, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PureGPT(nn.Module):
    def __init__(self, vocab: int = 5000, dim: int = 256, layers: int = 6,
                 heads: int = 8, ff: int = 1024, max_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, ff, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids, targets=None):
        B, L = ids.shape
        assert L <= self.max_len, f"panjang {L} > max_len {self.max_len}"
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new: int = 50, temperature: float = 0.8, top_k: int = 40):
        self.eval()
        out = ids
        for _ in range(max_new):
            cur = out[:, -self.max_len:]
            logits, _ = self.forward(cur)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k < next_logits.size(-1):
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
        return out


CONFIGS = {
    # target latihan di CPU sini
    "tiny":  dict(vocab=5000, dim=256, layers=6, heads=8, ff=1024, max_len=256),
    # latihan GPU menengah
    "small": dict(vocab=8000, dim=768, layers=12, heads=12, ff=3072, max_len=512),
    # target 1B — training di device GPU lain, JANGAN di CPU sini
    "b1":    dict(vocab=16000, dim=2048, layers=22, heads=16, ff=5632, max_len=1024),
}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    cfg = CONFIGS[name]
    m = PureGPT(**cfg)
    total = count_params(m)
    print(f"PureGPT-{name}: {total:,} ({total/1e9:.3f}B / {total/1e6:.1f}M)")
    print(cfg)
    if name != "b1":
        x = torch.randint(0, cfg["vocab"], (1, 16))
        logits, loss = m(x, x)
        print("forward OK:", tuple(logits.shape), "loss:", float(loss))
        g = m.generate(x, max_new=10)
        print("generate OK:", tuple(g.shape))
    else:
        print("Mode b1: arsitektur dibuat di GPU saja (CPU terlalu berat untuk forward).")
