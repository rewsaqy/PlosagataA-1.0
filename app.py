"""Web validator hoaks: teks + upload poster -> model 12M + OCR + fact-check."""
import os
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

from dataset import ID2LABEL
from factcheck import local_verdict
from model import TinyHoaxTransformer
from tokenizer import WordTokenizer

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path(os.getenv("CKPT_DIR", "checkpoints"))

_history: deque = deque(maxlen=20)  # riwayat 20 cek terakhir (in-memory)

_model = None
_tok = None


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


@app.route("/api/check", methods=["POST"])
def check():
    text = request.form.get("text", "") or (request.json.get("text") if request.is_json else "")
    poster_text = ""

    # upload poster opsional
    if "poster" in request.files:
        f = request.files["poster"]
        if f and f.filename:
            from ocr_poster import ocr_poster
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
    app.run(host="0.0.0.0", port=5000, debug=True)
