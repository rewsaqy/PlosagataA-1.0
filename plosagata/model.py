"""Model Plosagata — decoder-only causal, 100% kode sendiri."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .layers import Block


class PureGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.use_checkpoint = False  # hemat VRAM: aktifkan di GPU kecil
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.ff, cfg.dropout) for _ in range(cfg.layers)]
        )
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        # init kecil ala GPT-2: embedding N(0,0.02), proyeksi residu diskala
        # agar logit tidak meledak (sebelumnya loss 161, sekarang ~8.5)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=0.02)
        scale = (2 * len(self.blocks)) ** 0.5
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, std=0.02 / scale)
            nn.init.normal_(b.ff[2].weight, std=0.02 / scale)

    def forward(self, ids, targets=None):
        B, L = ids.shape
        assert L <= self.cfg.max_len, f"panjang {L} > max_len {self.cfg.max_len}"
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        if self.use_checkpoint and self.training:
            import torch.utils.checkpoint as ckpt
            for b in self.blocks:
                x = ckpt.checkpoint(b, x, use_reentrant=False)
        else:
            for b in self.blocks:
                x = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new: int = 50, temperature: float = 0.8,
                 top_k: int = 40, ignore_ids=None, repetition_penalty: float = 1.2):
        self.eval()
        out = ids
        for _ in range(max_new):
            cur = out[:, -self.cfg.max_len:]
            logits, _ = self.forward(cur)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if ignore_ids:
                next_logits[:, list(ignore_ids)] = float("-inf")
            if repetition_penalty != 1.0:  # cegah pengulangan kata (efek samping weight tying)
                for tok in set(out[0].tolist()):
                    if next_logits[0, tok] > 0:
                        next_logits[0, tok] /= repetition_penalty
                    else:
                        next_logits[0, tok] *= repetition_penalty
            if top_k and top_k < next_logits.size(-1):
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            out = torch.cat([out, torch.multinomial(probs, 1)], dim=1)
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
