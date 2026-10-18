"""SagiEngine — otak chatbot web: guard + model murni + verdict 12M."""
import os

import torch
import torch.nn.functional as F

from hoaxcore import (
    ID2LABEL,
    WordTokenizer,
    cari_sumber,
    local_verdict,
    ringkasan_hasil,
)
from plosagata import (
    CONFIGS,
    PureGPT,
    inspect_security,
    is_follow_up_safe,
    is_news_topic,
    normalize_slang,
    redact_pii,
    sagi_hello,
    sagi_redirect,
)

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

        # tokenizer chat — urutan: JSON penuh > dict+pieces > build fallback.
        # JSON penuh memulihkan backend BPE persis seperti saat training.
        # Eksplisit > implisit: ckpt HANYA dari param/env, tidak auto-attach.
        # (Auto-attach file liar bikin perilaku tak tentu — mis. eval gagal
        # karena mesin tetangga. Lihat BUGS.md.)
        # Untuk LIHAT ketersediaan tanpa memakai: _ckpt_tersedia(size).
        self.ckpt = ckpt
        self.ckpt_dir = Path(ckpt_dir)
        self.trained = bool(self.ckpt and Path(self.ckpt).exists())
        self.use_pieces = False
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
                        self.use_pieces = True
                        loaded_tok = True
                        break
                    except Exception:
                        pass
            jcand = str(ckpt).replace(".pt", ".tokenizer.json")
            if Path(jcand).exists():
                try:
                    self.tok = WordTokenizer.load(jcand)
                    self.use_pieces = False
                    loaded_tok = True
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

    @staticmethod
    def _ckpt_tersedia(size):
        """Daftar file ckpt pasangan di folder proyek (hanya info/display)."""
        from pathlib import Path as _P
        ada = []
        cands = list(_P(".").glob(f"pure-{size}*.pt")) + list(_P(".").glob("pure-*.pt"))
        if _P("checkpoints").exists():
            cands += list(_P("checkpoints").glob(f"pure-{size}*.pt")) + list(_P("checkpoints").glob("pure-*.pt"))
        for p in cands:
            if ".vocab." in p.name:
                continue
            try:
                from plosagata import detect_size
                if detect_size(str(p)) == size:
                    ada.append(str(p))
            except Exception:
                continue
        return sorted(list(set(ada)))

    def _get_model(self):
        if self.model is None:
            m = PureGPT(self.cfg).to(self.device)
            if self.ckpt:
                try:
                    m.load_state_dict(torch.load(self.ckpt, map_location=self.device))
                except RuntimeError:
                    # Salah ukuran? Deteksi dari isi checkpoint, bukan tebakan.
                    from plosagata import detect_size
                    pas = detect_size(self.ckpt)
                    if pas and pas != self.size:
                        print(f"[sagi] peringatan: ckpt {self.ckpt} berukuran {pas}; "
                              f"permintaan '{self.size}' menyesuaikan ke {pas}.")
                        self.cfg = CONFIGS[pas]
                        self.size = pas
                        # utama: vocab pasangan ckpt; fallback: seeds
                        self.use_pieces = False
                        for cand in [str(self.ckpt) + ".vocab.pt",
                                     str(self.ckpt).replace(".pt", ".vocab.pt")]:
                            from pathlib import Path as _P
                            if _P(cand).exists():
                                try:
                                    _v = torch.load(cand, map_location="cpu",
                                                    weights_only=True)
                                    self.tok = WordTokenizer(_v, max_len=self.cfg.max_len)
                                    self.use_pieces = True
                                    break
                                except Exception:
                                    pass
                        else:
                            seeds = ["halo", "hai", "sagi", "berita", "hoaks", "cek"]
                            self.tok = WordTokenizer.build(
                                seeds, vocab_size=self.cfg.vocab, max_len=256,
                                seed_file="data/vocab_seed.txt")
                        self.ign = {i for t, i in self.tok.vocab.items()
                                    if t.startswith("<pad") or t in ("<unk>", "<cls>")}
                        self.eos = {self.tok.sep_id}
                        m = PureGPT(self.cfg).to(self.device)
                        m.load_state_dict(torch.load(self.ckpt, map_location=self.device))
                    else:
                        raise RuntimeError(
                            f"Checkpoint {self.ckpt} tidak cocok dengan arsitektur "
                            f"{self.size}. Pakai --size yang sesuai (terdeteksi: {pas}).")
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

        # 0. AUDIT KEAMANAN KONSTITUSIONAL MULTILAPIS (Frontier AI Safety Defense)
        # Deteksi Prompt Injection, Jailbreak DAN, Pencurian System Prompt, Senjata Siber, dan Krisis Jiwa
        sec = inspect_security(text)
        if sec["is_threat"] and not pure_think:
            return {
                "reply": sec["reply"],
                "topic": "offtopic",  # Kompatibel penuh dengan test suite & client
                "verdict": None,
                "security": sec,
            }

        # Bersihkan PII jika ada sebelum pemrosesan lebih lanjut
        text_safe = sec["sanitized_text"]
        clean = normalize_slang(text_safe)
        topic = is_news_topic(clean)
        sess = self._sesi_get(session_id)
        topik = (sess or {}).get("topik")

        def _wrap_reply(d):
            if "reply" in d and d["reply"]:
                redacted, _ = redact_pii(d["reply"])
                if sec["pii_detected"] and "disensor" not in redacted.lower():
                    redacted += (
                        "\n\n> 🔒 *Catatan Privasi: Data sensitif (seperti NIK / Nomor Kartu) "
                        "disensor secara otomatis oleh Plosagata Data Guard demi melindungi keamananmu.*"
                    )
                d["reply"] = redacted
            if "security" not in d:
                d["security"] = sec
            return d

        if not pure_think:
            if topic == "offtopic":
                # Merujuk topik aktif? → follow-up, JANGAN tolak.
                # is_follow_up_safe: kata off-topic ("pacar") mem-veto.
                if topik and is_follow_up_safe(clean):
                    topic = "followup"
                else:
                    return _wrap_reply({"reply": sagi_redirect(text_safe), "topic": topic, "verdict": None})
            if topic == "greeting":
                return _wrap_reply({"reply": sagi_hello(text_safe), "topic": topic, "verdict": None})
        if topik and topic == "news" and len(clean) < 80 and is_follow_up_safe(clean):
            # Pertanyaan pendek tanpa kata kunci berita ("kapan cairnya?")
            # = merujuk topik aktif, bukan berita baru.
            from plosagata.guard import NEWS_KEYWORDS
            if not any(k in clean.lower() for k in NEWS_KEYWORDS):
                topic = "followup"
        if topic == "news":
            klaim, kueri = text_safe, text_safe
            if sess is not None:
                sess["topik"] = text_safe  # berita baru = topik baru
        elif topic == "followup":
            klaim = topik if topik else text_safe
            kueri = f"{topik} {text_safe}".strip() if topik else text_safe
        else:  # pure_think: anggap berita
            klaim, kueri = text_safe, text_safe
        v = self.verdict(klaim)
        lv = local_verdict(klaim, v)
        try:
            sumber = cari_sumber(kueri)
        except Exception:
            sumber = {"kueri": "", "wikipedia": [], "web": [], "cek_fakta": []}
        verdict = {**v, "label": lv["label_final"],
                   "skor": lv["skor_hoaks"], "kesimpulan": lv["kesimpulan"]}
        if not self.trained and not pure_think:
            isi = ringkasan_hasil(
                label=verdict["label"],
                skor=lv["skor_hoaks"],
                kesimpulan=lv["kesimpulan"],
                fiksi=lv["sinyal"].get("fiksi_hits"),
                teks=klaim,
                sinyal=lv["sinyal"],
            )
            if topic == "followup":
                isi = "Merujuk berita tadi:\n" + isi
            return _wrap_reply({"reply": isi, "topic": topic,
                                "verdict": verdict, "sumber": sumber})
        src = kueri if topic == "followup" else clean
        if self.use_pieces:
            from hoaxcore import piece_encode
            ids, _ = piece_encode(src, self.tok.vocab, self.tok.max_len)
        else:
            ids, _ = self.tok.encode(src)
        real = [i for i in ids if i != self.tok.pad_id][:64]
        out = self._get_model().generate(
            torch.tensor([real]).to(self.device), max_new=max_new,
            temperature=float(os.getenv("SAGI_TEMP", "0.7")),
            top_k=int(os.getenv("SAGI_TOPK", "40")),
            no_repeat_ngram=int(os.getenv("SAGI_NOREPEAT", "3")),
            eos_ids=self.eos,
            ignore_ids=self.ign)[0].tolist()
        if self.use_pieces:
            from hoaxcore import piece_decode as _pd
            inv = {i: t for t, i in self.tok.vocab.items()}
            return _wrap_reply({"reply": _pd(out[len(real):], inv), "topic": topic,
                                "verdict": verdict, "sumber": sumber})
        return _wrap_reply({"reply": self.tok.decode(out[len(real):]), "topic": topic,
                            "verdict": verdict, "sumber": sumber})
