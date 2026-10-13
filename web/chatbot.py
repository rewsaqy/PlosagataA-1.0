"""SagiEngine — otak chatbot web: guard + model murni + verdict 12M."""
import os

import torch
import torch.nn.functional as F

from hoaxcore import (ID2LABEL, TinyHoaxTransformer, WordTokenizer, cari_sumber,
                      local_verdict)
from plosagata import (CONFIGS, PureGPT, is_news_topic, normalize_slang,
                       sagi_hello, sagi_redirect)


class SagiEngine:
    def __init__(self, ckpt_dir="checkpoints", size=None, ckpt=""):
        from pathlib import Path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        size = size or os.getenv("SAGI_SIZE", "Plosagata_Go")
        ckpt = ckpt or os.getenv("SAGI_CKPT", "")
        self.cfg = CONFIGS[size]
        self.size = size

        # tokenizer chat (dengan seed kata) — bangun sekali
        seeds = ["halo", "hai", "sagi", "berita", "hoaks", "cek"]
        self.tok = WordTokenizer.build(seeds, vocab_size=self.cfg.vocab,
                                       max_len=256, seed_file="data/vocab_seed.txt")
        self.ign = {i for t, i in self.tok.vocab.items()
                    if t.startswith("<pad") or t in ("<unk>", "<cls>", "<sep>")}

        self.model = None
        self.ckpt = ckpt
        self.ckpt_dir = Path(ckpt_dir)

        # validator 12M (ringan, selalu aktif untuk label + skor)
        try:
            vt = WordTokenizer.load(self.ckpt_dir / "vocab.json")
            vm = TinyHoaxTransformer(len(vt.vocab), max_len=vt.max_len)
            for n in ["best.pt", "last.pt"]:
                p = self.ckpt_dir / n
                if p.exists():
                    vm.load_state_dict(torch.load(p, map_location=self.device,
                                                  weights_only=True))
                    break
            vm.to(self.device).eval()
            self.vtok, self.vmodel = vt, vm
        except Exception:
            self.vtok, self.vmodel = None, None

    def _get_model(self):
        if self.model is None:
            m = PureGPT(self.cfg).to(self.device)
            if self.ckpt:
                m.load_state_dict(torch.load(self.ckpt, map_location=self.device))
            m.eval()
            self.model = m
        return self.model

    def verdict(self, text):
        if self.vmodel is None:
            return {"label": "meragukan", "probs": [0.33, 0.33, 0.34]}
        ids, mask = self.vtok.encode(text)
        with torch.no_grad():
            probs = F.softmax(self.vmodel(torch.tensor([ids]).to(self.device),
                                          torch.tensor([mask]).to(self.device)),
                              dim=-1)[0].tolist()
        return {"label": ID2LABEL[max(range(3), key=lambda i: probs[i])], "probs": probs}

    def reply(self, text, max_new=60):
        import os as _os
        pure_think = _os.getenv("SAGI_PURE_THINK", "0") == "1"
        clean = normalize_slang(text)
        topic = is_news_topic(clean)
        if not pure_think:
            if topic == "offtopic":
                return {"reply": sagi_redirect(text), "topic": topic, "verdict": None}
            if topic == "greeting":
                return {"reply": sagi_hello(text), "topic": topic, "verdict": None}
        ids, _ = self.tok.encode(clean)
        real = [i for i in ids if i != self.tok.pad_id][:64]
        out = self._get_model().generate(torch.tensor([real]).to(self.device),
                                         max_new=max_new, ignore_ids=self.ign)[0].tolist()
        v = self.verdict(text)
        lv = local_verdict(text, v)
        try:
            sumber = cari_sumber(text)
        except Exception:
            sumber = {"kueri": "", "wikipedia": [], "cek_fakta": []}
        return {"reply": self.tok.decode(out[len(real):]), "topic": topic,
                "verdict": {**v, "skor": lv["skor_hoaks"],
                            "kesimpulan": lv["kesimpulan"]},
                "sumber": sumber}
