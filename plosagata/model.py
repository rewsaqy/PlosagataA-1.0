"""Model Plosagata — decoder-only causal, 100% kode sendiri."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIGS, GPTConfig
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
                x, _ = ckpt.checkpoint(b, x, use_reentrant=False)
        else:
            for b in self.blocks:
                x, _ = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new: int = 50, temperature: float = 0.8,
                 top_k: int = 40, ignore_ids=None, repetition_penalty: float = 1.2,
                 no_repeat_ngram: int = 3, eos_ids=None, on_token=None):
        """Generate hemat: KV-cache (O(N) per token, bukan O(N^2)) + berhenti di EOS.

        on_token: callback opsional ``on_token(token_id)`` yang dipanggil tiap
        token BARU (EOS tidak termasuk) — dipakai streaming SSE di sagi.engine.
        """
        self.eval()
        ids = ids[:, -self.cfg.max_len:]
        out = ids
        # Prefill seluruh prompt sekali, simpan KV-cache per layer.
        B, L0 = ids.shape
        pos = torch.arange(L0, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        pasts = []
        for b in self.blocks:
            x, present = b(x)
            pasts.append(present)
        h = self.ln(x)
        stop = set(eos_ids) if eos_ids else set()
        for _ in range(max_new):
            logits = self.head(h[:, -1:, :])
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if ignore_ids:
                next_logits[:, list(ignore_ids)] = float("-inf")
            if repetition_penalty != 1.0:  # cegah pengulangan kata (efek samping weight tying)
                for tok in set(out[0].tolist()):
                    if next_logits[0, tok] > 0:
                        next_logits[0, tok] /= repetition_penalty
                    else:
                        next_logits[0, tok] *= repetition_penalty
            if no_repeat_ngram > 0:  # blokir frasa berulang (anti-ngelantur)
                seq = out[0].tolist()
                if len(seq) >= no_repeat_ngram - 1:
                    prefix = tuple(seq[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
                    banned = set()
                    for i in range(len(seq) - no_repeat_ngram + 1):
                        if tuple(seq[i:i + no_repeat_ngram - 1]) == prefix:
                            banned.add(seq[i + no_repeat_ngram - 1])
                    if banned:
                        next_logits[0, list(banned)] = float("-inf")
            if top_k and top_k < next_logits.size(-1):
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
            if int(nxt[0, 0]) in stop:
                break  # EOS: berhenti, jangan ngelantur sampai max_new
            if on_token is not None:
                on_token(int(nxt[0, 0]))  # streaming: kabari token baru
            # Decode 1 token memakai cache (posisi absolut, dibatasi max_len).
            Lnow = min(out.shape[1] - 1, self.cfg.max_len - 1)
            ppos = torch.tensor([[Lnow]], device=ids.device)
            x1 = self.tok_emb(nxt) + self.pos_emb(ppos)
            new_pasts = []
            for bb, past in zip(self.blocks, pasts):
                # Jaga cache tidak melebihi max_len (buang paling lama).
                if past[0].shape[2] >= self.cfg.max_len:
                    past = (past[0][:, :, 1:, :], past[1][:, :, 1:, :])
                x1, pr = bb(x1, past)
                new_pasts.append(pr)
            pasts = new_pasts
            h = self.ln(x1)
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def detect_size(ckpt_path: str):
    """Tebak nama config dari isi checkpoint (layer/dim/vocab).

    Mencegah salah pasang: ckpt Go dimuat ke model More, dll.
    Return nama config atau None bila tidak cocok satupun.
    """
    import torch
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    dim = sd.get("tok_emb.weight", torch.empty(0)).shape[1] if "tok_emb.weight" in sd else 0
    vocab = sd.get("tok_emb.weight", torch.empty(0)).shape[0] if "tok_emb.weight" in sd else 0
    for name, cfg in CONFIGS.items():
        if cfg.layers == layers and cfg.dim == dim and cfg.vocab == vocab:
            return name
    return None
