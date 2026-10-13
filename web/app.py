"""Web validator hoaks: teks + upload poster -> model 12M + OCR + fact-check."""
import os
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from flask import Flask, jsonify, render_template, request

from hoaxcore import ID2LABEL, TinyHoaxTransformer, WordTokenizer, local_verdict, ocr_poster
from plosagata import CONFIGS

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path(os.getenv("CKPT_DIR", "checkpoints"))
SAGI_TOKEN = os.getenv("SAGI_TOKEN", "")  # kalau diisi, /api/chat wajib pakai token


def _cek_token():
    if not SAGI_TOKEN:
        return True  # mode lokal terbuka
    return (request.headers.get("X-Sagi-Token") == SAGI_TOKEN
            or request.args.get("token") == SAGI_TOKEN)

_history: deque = deque(maxlen=20)  # riwayat 20 cek terakhir (in-memory)

_model = None
_tok = None
_engines = {}


def get_engine(size=None):
    key = size or os.getenv("SAGI_SIZE", "Plosagata_Go")
    if key not in CONFIGS:
        key = "Plosagata_Go"
    if key not in _engines:
        from chatbot import SagiEngine
        _engines[key] = SagiEngine(size=key)
    return _engines[key]


def warmup():
    """Nyalakan AI 12M saat server start (bukan saat request pertama)."""
    if os.getenv("SAGI_WARMUP", "1") != "1":
        return
    try:
        get_model()
        print("[sagi] validator 12M siap (auto-nyala).")
    except Exception as e:
        print(f"[sagi] warmup gagal: {e}")


def get_model():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    try:
        _tok = WordTokenizer.load(CKPT / "vocab.json")
        _model = TinyHoaxTransformer(
            vocab_size=len(_tok.vocab), max_len=_tok.max_len, num_classes=3
        )
        for name in ["best.pt", "last.pt"]:
            p = CKPT / name
            if p.exists():
                _model.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
                break
        _model.to(DEVICE).eval()
    except Exception as e:
        print(f"[warn] model belum ada ({e}), pakai mode heuristic saja.")
        _model, _tok = None, None
    return _model, _tok


def model_predict(text):
    model, tok = get_model()
    if model is None:
        return {"label": "meragukan", "probs": [0.33, 0.33, 0.34], "mode": "heuristic-only"}
    import torch.nn.functional as F
    ids, mask = tok.encode(text)
    ids_t = torch.tensor([ids]).to(DEVICE)
    mask_t = torch.tensor([mask]).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(ids_t, mask_t), dim=-1)[0].tolist()
    pred = max(range(3), key=lambda i: probs[i])
    return {"label": ID2LABEL[pred], "probs": probs, "mode": "model-12M"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    if not _cek_token():
        return jsonify({"ok": False, "error": "Token salah / hilang."}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("message", "") or request.form.get("text", "")).strip()
    size = data.get("model", "") or request.form.get("model", "")
    if not text:
        return jsonify({"ok": False, "error": "Pesan kosong."}), 400
    try:
        r = get_engine(size or None).reply(text)
        return jsonify({"ok": True, "model_size": get_engine(size or None).size, **r})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Engine gagal: {e}"}), 500


@app.route("/api/check", methods=["POST"])
def check():
    text = request.form.get("text", "") or (request.json.get("text") if request.is_json else "")
    poster_text = ""

    # upload poster opsional
    if "poster" in request.files:
        f = request.files["poster"]
        if f and f.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix) as tmp:
                f.save(tmp.name)
                ocr = ocr_poster(tmp.name)
            poster_text = ocr.get("text", "")
            if poster_text:
                text = (text + "\n" + poster_text).strip()

    if not text.strip():
        return jsonify({"ok": False, "error": "Teks kosong. Isi teks atau upload poster."}), 400

    hasil_model = model_predict(text)
    verifikasi = local_verdict(text, hasil_model)
    _history.appendleft({
        "waktu": time.strftime("%H:%M:%S"),
        "cuplikan": text[:80],
        "label": hasil_model["label"],
        "skor": verifikasi["skor_hoaks"],
    })
    return jsonify({
        "ok": True,
        "input": text[:2000],
        "poster_text": poster_text[:2000],
        "model": hasil_model,
        "verifikasi": verifikasi,
    })


@app.route("/api/history")
def history():
    return jsonify({"ok": True, "riwayat": list(_history)})


@app.route("/api/stats")
def stats():
    c = Counter(h["label"] for h in _history)
    return jsonify({"ok": True, "total": len(_history), "per_label": dict(c)})


if __name__ == "__main__":
    warmup()
    app.run(host="0.0.0.0", port=5000, debug=True)
