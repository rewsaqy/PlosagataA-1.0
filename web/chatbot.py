"""SagiEngine — otak chatbot web: guard + model murni + verdict 12M."""
import os

import torch
import torch.nn.functional as F

from hoaxcore import (ID2LABEL, TinyHoaxTransformer, WordTokenizer, cari_sumber,
                       final_label, local_verdict, ringkasan_hasil)
from plosagata import (CONFIGS, PureGPT, is_follow_up_safe, is_news_topic,
                       normalize_slang, sagi_hello, sagi_redirect)

MAX_SESI = 500   # batas memori topik (in-memory, per proses)
SESI_UMUR = 7200  # detik — sesi lebih tua dari ini dibuang


class SagiEngine:
    def __init__(self, ckpt_dir="checkpoints", size=None, ckpt=""):
        from pathlib import Path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        size = size or os.getenv("SAGI_SIZE", "Plosagata_Go")
        ckpt = ckpt or os.getenv("SAGI_CKPT", "")
        self.cfg = CONFIGS[size]
        self.size = size

        # tokenizer chat — muat vocab berpasangan jika ada checkpoint, atau build fallback
        self.ckpt = ckpt
        self.ckpt_dir = Path(ckpt_dir)
        self.trained = bool(ckpt and Path(ckpt).exists())

        loaded_tok = False
        if self.trained:
            cands = [str(ckpt) + ".vocab.pt", str(ckpt).replace(".pt", ".vocab.pt")]
            if "-best.pt" in str(ckpt):
                base = str(ckpt).replace("-best.pt", ".pt")
                cands += [base + ".vocab.pt", base.replace(".pt", ".vocab.pt")]
            for cand in cands:
                p_cand = Path(cand)
                if p_cand.exists():
                    try:
                        vocab = torch.load(p_cand, map_location="cpu", weights_only=True)
                        self.tok = WordTokenizer(vocab, max_len=self.cfg.max_len)
                        loaded_tok = True
                        break
                    except Exception:
                        pass

        if not loaded_tok:
            seeds = ["halo", "hai", "sagi", "berita", "hoaks", "cek"]
            self.tok = WordTokenizer.build(seeds, vocab_size=self.cfg.vocab,
                                           max_len=256, seed_file="data/vocab_seed.txt")

        self.ign = {i for t, i in self.tok.vocab.items()
                    if t.startswith("<pad") or t in ("<unk>", "<cls>")}
        # <sep> diizinkan sebagai EOS agar generate bisa berhenti sendiri.
        self.eos = {self.tok.sep_id}
        self.model = None

        # Memori topik per sesi: session_id -> {"topik": str|None, "aktif": ts}
        # Topik = berita terakhir yg ditempel; follow-up merujuk ke sini.
        self._sesi = {}

        # validator 12M: pakai instance tunggal (hemat RAM, tidak diduplikat).
        try:
            from hoaxcore import get_classifier
            self.vmodel, self.vtok = get_classifier(str(self.ckpt_dir), self.device)
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
        ids, mask = self.vtok.encode(normalize_slang(text))
        with torch.no_grad():
            probs = F.softmax(self.vmodel(torch.tensor([ids]).to(self.device),
                                          torch.tensor([mask]).to(self.device)),
                              dim=-1)[0].tolist()
        return {"label": ID2LABEL[max(range(3), key=lambda i: probs[i])], "probs": probs}

    def _sesi_get(self, session_id):
        """Ambil/buat slot sesi + buang sesi kadaluarsa. Return dict|None."""
        import time as _t
        if not session_id:
            return None
        now = _t.time()
        if len(self._sesi) > MAX_SESI:
            tua = [k for k, s in self._sesi.items() if now - s["aktif"] > SESI_UMUR]
            for k in tua:
                del self._sesi[k]
            while len(self._sesi) > MAX_SESI:
                self._sesi.pop(next(iter(self._sesi)))
        s = self._sesi.get(session_id)
        if s is None:
            s = {"topik": None, "aktif": now}
            self._sesi[session_id] = s
        s["aktif"] = now
        return s

    def reply(self, text, max_new=60, session_id=""):
        import os as _os
        pure_think = _os.getenv("SAGI_PURE_THINK", "0") == "1"
        clean = normalize_slang(text)
        topic = is_news_topic(clean)
        sess = self._sesi_get(session_id)
        topik = (sess or {}).get("topik")
        if not pure_think:
            if topic == "offtopic":
                # Merujuk topik aktif? → follow-up, JANGAN tolak.
                # is_follow_up_safe: kata off-topic ("pacar") mem-veto.
                if topik and is_follow_up_safe(clean):
                    topic = "followup"
                else:
                    return {"reply": sagi_redirect(text), "topic": topic, "verdict": None}
            if topic == "greeting":
                return {"reply": sagi_hello(text), "topic": topic, "verdict": None}
        if topik and topic == "news" and len(clean) < 80 and is_follow_up_safe(clean):
            # Pertanyaan pendek tanpa kata kunci berita ("kapan cairnya?")
            # = merujuk topik aktif, bukan berita baru.
            from plosagata.guard import NEWS_KEYWORDS
            if not any(k in clean.lower() for k in NEWS_KEYWORDS):
                topic = "followup"
        if topic == "news":
            klaim, kueri = text, text
            if sess is not None:
                sess["topik"] = text  # berita baru = topik baru
        elif topic == "followup":
            klaim, kueri = topik, topik + " " + text
        else:  # pure_think: anggap berita
            klaim, kueri = text, text
        v = self.verdict(klaim)
        lv = local_verdict(klaim, v)
        try:
            sumber = cari_sumber(kueri)
        except Exception:
            sumber = {"kueri": "", "wikipedia": [], "cek_fakta": []}
        verdict = {**v, "label": lv["label_final"],
                   "skor": lv["skor_hoaks"], "kesimpulan": lv["kesimpulan"]}
        if not self.trained and not pure_think:
            isi = ringkasan_hasil(verdict["label"], lv["skor_hoaks"],
                                  lv["kesimpulan"], sumber,
                                  lv["sinyal"].get("fiksi_hits"))
            if topic == "followup":
                isi = "Merujuk berita tadi:\n" + isi
            return {"reply": isi, "topic": topic,
                    "verdict": verdict, "sumber": sumber}
        ids, _ = self.tok.encode(kueri if topic == "followup" else clean)
        real = [i for i in ids if i != self.tok.pad_id][:64]
        out = self._get_model().generate(
            torch.tensor([real]).to(self.device), max_new=max_new,
            temperature=float(os.getenv("SAGI_TEMP", "0.7")),
            top_k=int(os.getenv("SAGI_TOPK", "40")),
            no_repeat_ngram=int(os.getenv("SAGI_NOREPEAT", "3")),
            eos_ids=self.eos,
            ignore_ids=self.ign)[0].tolist()
        return {"reply": self.tok.decode(out[len(real):]), "topic": topic,
                "verdict": verdict, "sumber": sumber}
