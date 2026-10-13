import math
import torch
import torch.nn as nn


class TinyHoaxTransformer(nn.Module):
    """
    Transformer Encoder classifier ~12 juta parameter.
    Config default: vocab 5000, d_model 384, 6 layer, 6 head, ff 1280
    Total ≈ 11.6M (disebut 12M).
    Classes: 0=Valid, 1=Hoaks, 2=Meragukan
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 6,
        dim_ff: int = 1280,
        max_len: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # CLS head: pakai mean-pool + MLP kecil
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, ids, mask=None):
        # ids: (B, L), mask: (B, L) 1=isi, 0=pad
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(ids) + self.pos_emb(pos)
        x = self.dropout(x)

        if mask is not None:
            # TransformerEncoder batch_first pakai src_key_padding_mask
            # True = ignore (pad)
            pad_mask = (mask == 0)
        else:
            pad_mask = None

        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        if mask is not None:
            m = mask.unsqueeze(-1).float()
            summed = (h * m).sum(dim=1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        else:
            pooled = h.mean(dim=1)

        return self.head(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TinyHoaxTransformer()
    total = count_params(m)
    print(f"Total parameter: {total:,} ({total/1e6:.2f}M)")
    # sanity forward
    x = torch.randint(0, 5000, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)
    out = m(x, mask)
    print("logits shape:", tuple(out.shape))
