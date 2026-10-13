"""Chat dengan GPT murni milikmu (bobot acak -> awalnya ngawur, wajar).

  python3 pure_chat.py --size Plosagata_Go --prompt "Halo bang!"
  python3 pure_chat.py --size Plosagata_Go   (interaktif)

Setelah training (lihat train_pure.py), muat checkpoint:
  python3 pure_chat.py --size Plosagata_Go --ckpt pure-Plosagata_Go.pt --prompt "Halo!"
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

from plosagata import (CONFIGS, PureGPT, is_news_topic, sagi_hello,
                       sagi_redirect, normalize_slang)
from hoaxcore import WordTokenizer, cari_sumber, local_verdict, ringkasan_hasil


def build_tokenizer(vocab_size: int):
    seeds = ["halo", "hai", "pagi", "bang", "berita", "hoaks", "bener", "gak",
             "giveaway", "klik", "link", "covid", "vaksin", "pemerintah", "cek"]
    return WordTokenizer.build(seeds, vocab_size=vocab_size, max_len=256,
                               seed_file="data/vocab_seed.txt")


def ignore_ids_for(tok: WordTokenizer) -> set[int]:
    # jangan pernah generate token sampah padding/unknown
    return {i for t, i in tok.vocab.items()
            if t.startswith("<pad") or t in ("<unk>", "<cls>", "<sep>")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="Plosagata_More", choices=list(CONFIGS.keys()))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--pure-think", action="store_true",
                    help="Tanpa template: SEMUA pesan (termasuk halo/off-topic) "
                         "lewat tokenizer -> model -> decode. Butuh model TERLATIH, "
                         "kalau tidak output ngawur.")
    a = ap.parse_args()

    cfg = CONFIGS[a.size]
    if a.size != "Plosagata_Go":
        print("[model besar di CPU lambat; untuk serius pakai GPU + checkpoint hasil training]")

    tok = build_tokenizer(cfg.vocab)
    ign = ignore_ids_for(tok)
    _model = None

    def get_model():
        nonlocal _model
        if _model is None:
            m = PureGPT(cfg)
            if a.ckpt and os.path.exists(a.ckpt):
                try:
                    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
                    print(f"[ckpt dimuat: {a.ckpt}]")
                except RuntimeError as e:
                    print(f"[ckpt tak cocok, pakai bobot acak: {e}]")
            else:
                print("[bobot ACAK — output awal ngawur itu normal sebelum training]")
            _model = m
        return _model

    def answer(prompt: str) -> str:
        clean = normalize_slang(prompt)
        topic = is_news_topic(clean)
        if not a.pure_think:
            if topic == "offtopic":
                return sagi_redirect(prompt)
            if topic == "greeting":
                return sagi_hello(prompt)
        trained = bool(a.ckpt and os.path.exists(a.ckpt))
        if not trained and not a.pure_think:
            # Model belum terlatih (bobot acak) — jawab ekstraktif saja.
            v = {"label": "meragukan", "probs": [0.33, 0.33, 0.34]}
            try:  # validator 12M bila ada (sama seperti SagiEngine)
                from hoaxcore import ID2LABEL, TinyHoaxTransformer
                import torch.nn.functional as _F
                vt = WordTokenizer.load("checkpoints/vocab.json")
                vm = TinyHoaxTransformer(len(vt.vocab), max_len=vt.max_len)
                for n in ["best.pt", "last.pt"]:
                    p = Path("checkpoints") / n
                    if p.exists():
                        vm.load_state_dict(torch.load(p, map_location="cpu",
                                                      weights_only=True))
                        break
                vm.eval()
                _ids, _mask = vt.encode(prompt)
                with torch.no_grad():
                    _pr = _F.softmax(vm(torch.tensor([_ids]),
                                        torch.tensor([_mask])), dim=-1)[0].tolist()
                v = {"label": ID2LABEL[max(range(3), key=lambda i: _pr[i])],
                     "probs": _pr}
            except Exception:
                pass
            lv = local_verdict(prompt, v)
            try:
                sumber = cari_sumber(prompt)
            except Exception:
                sumber = {"kueri": "", "wikipedia": [], "cek_fakta": []}
            return ringkasan_hasil(v["label"], lv["skor_hoaks"],
                                   lv["kesimpulan"], sumber)
        # mode berpikir-murni (atau topik berita): tokenizer -> model -> decode
        ids, _ = tok.encode(clean)
        real = [i for i in ids if i != tok.pad_id][:64]
        out = get_model().generate(torch.tensor([real]), max_new=a.max_new,
                                   ignore_ids=ign)[0].tolist()
        # BPE: decode via tokenizer (mapping id->kata mentah menghasilkan byte sampah)
        return tok.decode(out[len(real):])

    if a.prompt:
        print(answer(a.prompt))
        return
    print(f"Plosagata-{a.size} siap (murni kodinganmu). Ketik 'keluar' untuk berhenti.")
    while True:
        try:
            u = input("kamu> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("keluar", "exit", "quit"):
            break
        if u:
            print("ai> " + answer(u) + "\n")


if __name__ == "__main__":
    main()
