"""SagiEngine — mesin chatbot: guard keamanan + topik + verdict 12M + generasi.

Dipakai oleh web (web/app.py — endpoint SSE /api/chat/stream) dan CLI
(scripts/pure_chat.py). Prompt dibangun di sagi.prompt agar formatnya sama
persis dengan data training; memori percakapan ada di sagi.session.
"""
import os
import queue
import threading

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

from .prompt import bangun_prompt, bersihkan_output
from .session import SesiStore
from .sources import blok_sitasi

CATATAN_PII = (
    "\n\n> 🔒 *Catatan Privasi: Data sensitif (seperti NIK / Nomor Kartu) "
    "disensor secara otomatis oleh Plosagata Data Guard demi melindungi keamananmu.*"
)


class _Dihentikan(Exception):
    """Sinyal internal: hentikan generasi (klien SSE memutus koneksi)."""


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
        if self.trained:
            self._sesuaikan_max_len()

        # Memori percakapan per sesi: topik terakhir + riwayat giliran.
        self.sesi = SesiStore()

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

    def _sesuaikan_max_len(self):
        """Ckpt hasil latihan dengan --max-len berbeda tetap bisa dipakai.

        pos_emb.weight menyimpan panjang konteks saat latihan; kalau beda
        dengan config, pakai nilai ckpt (config global tidak diutak-atik).
        """
        try:
            import copy as _copy
            state = torch.load(self.ckpt, map_location="cpu", weights_only=True)
            pos = state.get("pos_emb.weight") if isinstance(state, dict) else None
            if pos is not None and int(pos.shape[0]) != int(self.cfg.max_len):
                baru = int(pos.shape[0])
                print(f"[sagi] ckpt {self.ckpt} ber-max_len {baru} "
                      f"(config {self.cfg.max_len}) — menyesuaikan.")
                self.cfg = _copy.copy(self.cfg)
                self.cfg.max_len = baru
        except Exception:
            pass

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

    # ------------------------------------------------------------------
    # Prompt & tokenisasi (format sama persis dengan training — B-35)
    # ------------------------------------------------------------------
    def _tokenisasi(self, teks_prompt: str) -> list[int]:
        """Token id tanpa padding & tanpa <sep> di ujung."""
        if self.use_pieces:
            from hoaxcore import piece_encode
            ids, _ = piece_encode(teks_prompt, self.tok.vocab, self.tok.max_len)
        else:
            ids, _ = self.tok.encode(teks_prompt)
        real = [i for i in ids if i != self.tok.pad_id]
        # <sep> = penanda akhir sekuens saat training. Jangan pernah ditaruh
        # sebelum generasi — model akan menganggap percakapan sudah selesai.
        if real and real[-1] == self.tok.sep_id:
            real.pop()
        return real

    def _ids_prompt(self, pesan_user: str, riwayat, max_new: int) -> list[int]:
        """Token prompt yang pasti muat: buang giliran terlama dulu."""
        budget = max(8, self.cfg.max_len - int(max_new) - 2)
        sisa = list(riwayat or [])
        while True:
            ids = self._tokenisasi(bangun_prompt(pesan_user, sisa))
            if len(ids) <= budget:
                return ids or [self.tok.cls_id]
            if not sisa:
                return ids[-budget:]  # terpaksa: potong kepala, ekor (pertanyaan) tetap
            sisa.pop(0)

    # ------------------------------------------------------------------
    # Generasi (streaming token via thread pekerja + antrean)
    # ------------------------------------------------------------------
    def _setelan_generate(self, max_new: int) -> dict:
        return dict(
            max_new=int(max_new),
            temperature=float(os.getenv("SAGI_TEMP", "0.7")),
            top_k=int(os.getenv("SAGI_TOPK", "40")),
            no_repeat_ngram=int(os.getenv("SAGI_NOREPEAT", "3")),
        )

    def _decode(self, ids) -> str:
        if self.use_pieces:
            from hoaxcore import piece_decode
            inv = {i: t for t, i in self.tok.vocab.items()}
            return piece_decode(list(ids), inv)
        return self.tok.decode(list(ids))

    def _generate_stream(self, ids_prompt, max_new: int):
        """Hasilkan ("t", delta_teks) per potongan, lalu ("akhir", teks_penuh).

        generate() berjalan di thread pekerja; callback on_token menaruh id ke
        antrean supaya potongan teks bisa mengalir real-time ke SSE.
        """
        ids_prompt = list(ids_prompt) or [self.tok.cls_id]
        antre: "queue.Queue" = queue.Queue()
        berhenti = threading.Event()

        def pekerja():
            try:
                def on_token(tid):
                    if berhenti.is_set():
                        raise _Dihentikan()
                    antre.put(int(tid))

                self._get_model().generate(
                    torch.tensor([ids_prompt]).to(self.device),
                    eos_ids=self.eos,
                    ignore_ids=self.ign,
                    on_token=on_token,
                    **self._setelan_generate(max_new),
                )
                antre.put(None)
            except _Dihentikan:
                antre.put(None)
            except Exception as e:  # diteruskan ke pemanggil
                antre.put(e)

        th = threading.Thread(target=pekerja, daemon=True)
        th.start()
        hasil: list[int] = []
        teks_lama = ""
        habis = False
        try:
            while True:
                item = antre.get()
                if item is None:
                    habis = True
                    break
                if isinstance(item, Exception):
                    raise item
                hasil.append(item)
                teks_now = self._decode(hasil)
                if teks_now.startswith(teks_lama):
                    delta = teks_now[len(teks_lama):]
                else:
                    delta = teks_now  # byte utf-8 terpotong di perbatasan token
                teks_lama = teks_now
                if delta:
                    yield ("t", delta)
        finally:
            berhenti.set()
            th.join(timeout=2.0)
        if habis:
            yield ("akhir", teks_lama)

    # ------------------------------------------------------------------
    # Inti percakapan
    # ------------------------------------------------------------------
    def _teks_template(self, lv: dict, klaim: str, topic: str) -> str:
        isi = ringkasan_hasil(
            label=lv["label_final"],
            skor=lv["skor_hoaks"],
            kesimpulan=lv["kesimpulan"],
            fiksi=lv["sinyal"].get("fiksi_hits"),
            teks=klaim,
            sinyal=lv["sinyal"],
        )
        if topic == "followup":
            isi = "Merujuk berita tadi:\n" + isi
        return isi

    def _rapikan_balasan(self, teks: str, sec: dict, blok: str = "") -> str:
        """Sensor PII + catatan privasi (dulu: _wrap_reply di web/chatbot.py)."""
        rapi, _ = redact_pii((teks or "") + (blok or ""))
        if sec.get("pii_detected") and "disensor" not in rapi.lower():
            rapi += CATATAN_PII
        return rapi

    def reply_stream(self, text: str, max_new: int = 60, session_id: str = ""):
        """Generator kejadian: meta -> token* -> final (dipakai SSE & reply())."""
        pure_think = os.getenv("SAGI_PURE_THINK", "0") == "1"

        # 0. AUDIT KEAMANAN KONSTITUSIONAL MULTILAPIS (Frontier AI Safety Defense)
        sec = inspect_security(text)
        text_safe = sec["sanitized_text"]
        clean = normalize_slang(text_safe)
        topic = is_news_topic(clean)
        sesi = self.sesi.ambil(session_id)
        topik = sesi.topik if sesi else None

        def aliran_statistik(isi, topic_now, verdict_now, sumber_now, simpan_teks=None):
            """Balasan non-generatif (guard/template): satu token lalu final."""
            yield {"type": "meta", "topic": topic_now,
                   "verdict": verdict_now, "sumber": sumber_now}
            blok = blok_sitasi(verdict_now, sumber_now) \
                if topic_now in ("news", "followup") else ""
            final_text = self._rapikan_balasan(isi, sec, blok)
            yield {"type": "token", "t": final_text}
            if simpan_teks and sesi is not None:
                self.sesi.catat_giliran(session_id, text_safe, simpan_teks)
            yield {"type": "final", "text": final_text, "security": sec}

        if sec["is_threat"] and not pure_think:
            yield from aliran_statistik(sec["reply"], "offtopic", None, None)
            return

        if not pure_think:
            if topic == "offtopic":
                # Merujuk topik aktif? -> follow-up, JANGAN tolak.
                if topik and is_follow_up_safe(clean):
                    topic = "followup"
                else:
                    yield from aliran_statistik(sagi_redirect(text_safe), topic, None, None)
                    return
            if topic == "greeting":
                yield from aliran_statistik(sagi_hello(text_safe), topic, None, None)
                return

        if topik and topic == "news" and len(clean) < 80 and is_follow_up_safe(clean):
            # Pertanyaan pendek tanpa kata kunci berita ("kapan cairnya?")
            # = merujuk topik aktif, bukan berita baru.
            from plosagata.guard import NEWS_KEYWORDS
            if not any(k in clean.lower() for k in NEWS_KEYWORDS):
                topic = "followup"

        if topic == "news":
            klaim, kueri = text_safe, text_safe
            if sesi is not None:
                sesi.topik = text_safe  # berita baru = topik baru
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
                   "skor": lv["skor_hoaks"], "kesimpulan": lv["kesimpulan"],
                   "lantai": bool(lv.get("lantai"))}

        # Jalur template: belum ada bobot (dan bukan pure_think).
        if not self.trained and not pure_think:
            isi = self._teks_template(lv, klaim, topic)
            yield from aliran_statistik(isi, topic, verdict, sumber, simpan_teks=isi)
            return

        # Jalur model saraf: prompt ala training + generasi streaming.
        yield {"type": "meta", "topic": topic, "verdict": verdict, "sumber": sumber}
        riwayat = list(sesi.giliran) if sesi else []
        ids_prompt = self._ids_prompt(text_safe, riwayat, max_new)
        teks_gen = ""
        ada_token = False
        for jenis, data in self._generate_stream(ids_prompt, max_new):
            if jenis == "t":
                ada_token = True
                yield {"type": "token", "t": data}
            elif jenis == "akhir":
                teks_gen = data
        teks_bersih = bersihkan_output(teks_gen)
        if not teks_bersih:
            # Model keluarkan sampah/kosong — jangan biarkan pengguna bingung.
            teks_bersih = self._teks_template(lv, klaim, topic)
        if sesi is not None:
            self.sesi.catat_giliran(session_id, text_safe, teks_bersih)
        blok = blok_sitasi(verdict, sumber) if topic in ("news", "followup") else ""
        final_text = self._rapikan_balasan(teks_bersih, sec, blok)
        if not ada_token:
            # Model langsung EOS / tak menghasilkan apa-apa: kirim teks final
            # sebagai satu token agar protokol klien tetap konsisten.
            yield {"type": "token", "t": final_text}
        yield {"type": "final", "text": final_text, "security": sec}

    def reply(self, text: str, max_new: int = 60, session_id: str = ""):
        """Versi non-streaming (API lama / tes): konsumsi aliran kejadian."""
        potongan, meta, final = [], {"topic": None, "verdict": None, "sumber": None}, {}
        for ev in self.reply_stream(text, max_new=max_new, session_id=session_id):
            jenis = ev.get("type")
            if jenis == "token":
                potongan.append(ev.get("t", ""))
            elif jenis == "meta":
                for k in ("topic", "verdict", "sumber"):
                    if k in ev:
                        meta[k] = ev[k]
            elif jenis == "final":
                final = ev
        return {
            "reply": final.get("text") or "".join(potongan),
            "topic": meta.get("topic"),
            "verdict": meta.get("verdict"),
            "sumber": meta.get("sumber"),
            "security": final.get("security"),
        }
