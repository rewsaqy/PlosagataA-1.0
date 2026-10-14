"""Lapisan transformer Plosagata — kode sendiri, tanpa library model."""
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

    def forward(self, x, past_kv=None):
        """past_kv=(k,v) cache untuk decoding cepat; return (y, (k,v))."""
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past_kv is None),
            dropout_p=self.drop.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y), (k.detach(), v.detach())


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

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln1(x), past_kv)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, present
