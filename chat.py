"""Chat AI validator hoaks — nyala dulu di CPU, training lanjutan di GPU lain.

Mode:
  python chat.py --dummy            -> tanpa download, template lokal (langsung nyala)
  python chat.py --model Qwen/Qwen2.5-0.5B-Instruct  -> chat beneran (download ~1GB sekali)

Integrasi: tiap pesan user dicek model 12M + heuristic, hasilnya disuntik
ke system prompt supaya AI bisa ngobrolin berita + sapaan harian.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

SYSTEM_BASE = (
    "Kamu adalah asisten validator berita berbahasa Indonesia. "
    "Tugasmu HANYA: membahas berita, cek fakta, dan validasi hoaks. "
    "Sapaan singkat boleh dibalas ramah, tapi langsung arahkan ke topik berita. "
    "Jika user membahas hal di luar berita (tugas sekolah, coding, resep, curhat, "
    "hiburan, dll), TOLAK dengan sopan dalam 1-2 kalimat dan ajak kembali "
    "membahas berita. Jangan pernah mengerjakan topik di luar berita. "
    "Untuk klaim berita, nilai kebenarannya dan sarankan sumber resmi "
    "(turnbackhoax.id, cekfakta.com, situs .go.id). Jangan sebarkan hoaks."
)

# Kata kunci topik berita — di luar ini = off-topic
NEWS_KEYWORDS = [
    "berita", "hoaks", "hoax", "fakta", "cek", "valid", "bener", "benar",
    "viral", "klaim", "artikel", "poster", "broadcast", "pesan berantai",
    "sumber", "koran", "media", "wartawan", "pemilu", "presiden", "menteri",
    "bencana", "gempa", "banjir", "vaksin", "covid", "giveaway", "penipuan",
    "lowongan", "bantuan", "bansos", "isu", "kabar", "informasi", "sebar",
    "turnbackhoax", "cekfakta", "mafindo", "kominfo",
]
GREETING_ONLY = ["halo", "hai", "hello", "pagi", "siang", "sore", "malam",
                 "assalamu", "tes", "permisi"]


def is_news_topic(text: str) -> str:
    """Return 'greeting' | 'news' | 'offtopic'."""
    t = text.lower().strip()
    if len(t) < 30 and any(g in t for g in GREETING_ONLY):
        return "greeting"
    if any(k in t for k in NEWS_KEYWORDS):
        return "news"
    if "?" in text and len(t) < 200:
        # pertanyaan pendek tanpa kata kunci berita -> anggap off-topic
        return "offtopic"
    if len(t) < 200:
        return "offtopic"
    # teks panjang tanpa sinyal berita (mis. tempelan artikel) tetap diproses
    return "news"


def redirect_reply() -> str:
    return (
        "Maaf, saya hanya bisa membahas seputar berita dan cek fakta. "
        "Ada berita atau klaim yang mau dicek kebenarannya? Tempel saja di sini."
    )


def hoax_context(text: str) -> str:
    """Ambil konteks dari model 12M + heuristic. Gagal -> string kosong (tetap nyala)."""
    try:
        from factcheck import local_verdict
        try:
            import torch
            from dataset import ID2LABEL
            from model import TinyHoaxTransformer
            from tokenizer import WordTokenizer
            from pathlib import Path
            ckpt = Path(__file__).parent / "checkpoints"
            if (ckpt / "vocab.json").exists():
                tok = WordTokenizer.load(ckpt / "vocab.json")
                m = TinyHoaxTransformer(len(tok.vocab), max_len=tok.max_len)
                for n in ["best.pt", "last.pt"]:
                    p = ckpt / n
                    if p.exists():
                        m.load_state_dict(__import__("torch").load(p, map_location="cpu"))
                        break
                m.eval()
                ids, mask = tok.encode(text)
                import torch.nn.functional as F
                with torch.no_grad():
                    probs = F.softmax(m(torch.tensor([ids]), torch.tensor([mask])), -1)[0].tolist()
                pred = max(range(3), key=lambda i: probs[i])
                label = ID2LABEL[pred]
            else:
                label, probs = "meragukan", [0.33, 0.33, 0.34]
        except Exception:
            label, probs = "meragukan", [0.33, 0.33, 0.34]
        v = local_verdict(text, {"label": label})
        return (
            f"\n[Internal validator: label={label} probs={probs} "
            f"skor_hoaks={v['skor_hoaks']} kesimpulan={v['kesimpulan']}]"
        )
    except Exception:
        return ""


def dummy_reply(user_text: str) -> str:
    """Fallback offline — tetap bisa demo sapa + bahas berita."""
    if is_news_topic(user_text) == "offtopic":
        return redirect_reply()
    t = user_text.lower().strip()
    sapaan = ["halo", "hai", "pagi", "siang", "sore", "malam", "assalamu", "bang", "kak", "tes"]
    if any(s in t for s in ["halo", "hai", "hello", "pagi", "siang", "sore", "malam", "assalamu"]) and len(t) < 40:
        return (
            "Halo! Saya AI validator hoaks buatanmu (mode ringan, belum fine-tune 1B). "
            "Kamu bisa sapa-sapa atau tempel berita/poster buat saya cek. Contoh: "
            "'Bener gak sih giveaway iPhone gratis klik link ini?'"
        )
    ctx = hoax_context(user_text)
    # heuristic singkat untuk jawaban dummy
    try:
        from factcheck import local_verdict
        v = local_verdict(user_text, {"label": "meragukan"})
        return (
            f"{v['kesimpulan']} (skor hoaks {v['skor_hoaks']}/100, mode dummy).\n"
            + "\n".join(f"- {s}" for s in v["saran"])
            + f"\nSumber: {', '.join(v['sumber_disarankan'])}."
            + ("\n" + ctx if ctx else "")
            + "\n\nCatatan: ini jawaban template. Untuk jawaban kayak ChatGPT, "
              "jalankan: python chat.py (mode model) atau fine-tune di GPU (lihat train_chat_lora.py)."
        )
    except Exception as e:
        return f"Maaf, saya belum bisa jawab ({e}). Coba lagi."


class ChatAI:
    def __init__(self, model_id: str, adapter: str = ""):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"[chat] load {model_id} ... (pertama kali download ~1GB)")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # CPU: fp32 biar kompatibel
            device_map="cpu",
            trust_remote_code=True,
        )
        if adapter:
            from peft import PeftModel
            print(f"[chat] load adapter LoRA: {adapter}")
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.device = "cpu"
        print("[chat] siap. Ketik 'keluar' untuk berhenti.")

    def reply(self, history: list[dict], user_text: str, max_new: int = 256) -> str:
        import torch
        topic = is_news_topic(user_text)
        if topic == "offtopic":
            return redirect_reply()
        ctx = hoax_context(user_text)
        msgs = [{"role": "system", "content": SYSTEM_BASE + ctx}] + history + [
            {"role": "user", "content": user_text}
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok([prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp, max_new_tokens=max_new, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=self.tok.eos_token_id,
            )
        gen = out[0][inp["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Ganti ke meta-llama/Llama-3.2-1B-Instruct untuk 1B pas (butuh HF login).")
    ap.add_argument("--dummy", action="store_true", help="Mode tanpa download, langsung nyala.")
    ap.add_argument("--adapter", default="", help="Folder adapter LoRA hasil train_chat_lora.py.")
    ap.add_argument("--once", default="", help="Satu pesan saja (non-interaktif, buat test).")
    a = ap.parse_args()

    if a.dummy:
        print("[mode dummy — langsung nyala, tanpa download]")
        if a.once:
            print(dummy_reply(a.once))
            return
        while True:
            try:
                u = input("kamu> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if u.lower() in ("keluar", "exit", "quit"):
                break
            print("ai> " + dummy_reply(u) + "\n")
        return

    bot = ChatAI(a.model, a.adapter)
    hist: list[dict] = []
    if a.once:
        print(bot.reply(hist, a.once))
        return
    while True:
        try:
            u = input("kamu> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            continue
        if u.lower() in ("keluar", "exit", "quit"):
            break
        r = bot.reply(hist, u)
        print("ai> " + r + "\n")
        hist += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        hist = hist[-10:]  # jaga memori


if __name__ == "__main__":
    main()
