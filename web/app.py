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

from hoaxcore import ID2LABEL, TinyHoaxTransformer, WordTokenizer, final_label, local_verdict, ocr_poster
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
ACTIVE = {"model": os.getenv("SAGI_SIZE", "Plosagata_Go")}


def get_engine(size=None):
    key = size or ACTIVE["model"]
    if key not in CONFIGS:
        key = "Plosagata_Go"
    if key not in _engines:
        try:
            from web.chatbot import SagiEngine
        except ImportError:
            from chatbot import SagiEngine
        _engines[key] = SagiEngine(size=key)
    return _engines[key]


def unload_engine(name):
    """Bongkar total engine lama (RAM/VRAM balik). Return True bila ada yg dibongkar."""
    eng = _engines.pop(name, None)
    if eng is None:
        return False
    try:
        eng.model = None
        eng.vmodel = None
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def select_model(name):
    """Ganti model AKTIF: matikan semua engine lain, nyalakan yg baru (sesi fresh)."""
    if name not in CONFIGS:
        return {"ok": False, "error": f"Model tak dikenal: {name}"}
    dimatikan = [k for k in list(_engines) if k != name and unload_engine(k)]
    ACTIVE["model"] = name
    eng = get_engine(name)  # bangun + warmup ringan
    from plosagata import PARAMS
    return {"ok": True, "active": name, "dimatikan": dimatikan,
            "trained": eng.trained, "params": PARAMS.get(name, 0)}


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
    """Instance tunggal via hoaxcore.shared (tidak diduplikat per modul)."""
    global _model, _tok
    if _model is not None:
        return _model, _tok
    try:
        from hoaxcore import get_classifier
        _model, _tok = get_classifier(str(CKPT), DEVICE)
    except Exception as e:
        print(f"[warn] model belum ada ({e}), pakai mode heuristic saja.")
        _model, _tok = None, None
    return _model, _tok


def model_predict(text):
    model, tok = get_model()
    if model is None:
        return {"label": "meragukan", "probs": [0.33, 0.33, 0.34], "mode": "heuristic-only"}
    import torch.nn.functional as F
    try:
        from plosagata.slang import normalize_slang
        text = normalize_slang(text)
    except Exception:
        pass
    ids, mask = tok.encode(text)
    ids_t = torch.tensor([ids]).to(DEVICE)
    mask_t = torch.tensor([mask]).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(ids_t, mask_t), dim=-1)[0].tolist()
    pred = max(range(3), key=lambda i: probs[i])
    return {"label": ID2LABEL[pred], "probs": probs, "mode": "model-12M"}


@app.after_request
def _cors(resp):
    # Biar browser di rumah bisa akses server teman (beda origin).
    # Token tetap wajib — CORS bukan pengganti auth.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Sagi-Token"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/models")
def models_page():
    return render_template("models.html")


@app.route("/api/models")
def models_api():
    from plosagata import DISPLAY, PARAMS
    items = []
    for n in CONFIGS:
        items.append({"name": n, "display": DISPLAY.get(n, n),
                      "params": PARAMS.get(n, 0),
                      "layers": CONFIGS[n].layers, "dim": CONFIGS[n].dim,
                      "loaded": n in _engines,
                      "trained": bool(_engines[n].trained) if n in _engines else False})
    return jsonify({"ok": True, "active": ACTIVE["model"], "models": items})


@app.route("/api/models/select", methods=["POST"])
def models_select():
    data = request.get_json(silent=True) or {}
    name = (data.get("model", "") or request.form.get("model", "")).strip()
    r = select_model(name)
    return jsonify(r), (200 if r.get("ok") else 400)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    if not _cek_token():
        return jsonify({"ok": False, "error": "Token salah / hilang."}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("message", "") or request.form.get("text", "")).strip()
    size = data.get("model", "") or request.form.get("model", "")
    session_id = (data.get("session", "") or request.form.get("session", "")).strip()
    if not text:
        return jsonify({"ok": False, "error": "Pesan kosong."}), 400
    try:
        if size:
            # Minta model eksplisit = pindah beneran (matikan yg lain).
            s = select_model(size)
            if not s.get("ok"):
                return jsonify({"ok": False, "error": s.get("error")}), 400
            eng = get_engine(size)
        else:
            eng = get_engine()
        r = eng.reply(text, session_id=session_id)
        return jsonify({"ok": True, "model_size": eng.size, **r})
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
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix) as tmp:
                    f.save(tmp.name)
                    tmp_path = tmp.name
                ocr = ocr_poster(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            poster_text = ocr.get("text", "")
            if poster_text:
                text = (text + "\n" + poster_text).strip()

    if not text.strip():
        return jsonify({"ok": False, "error": "Teks kosong. Isi teks atau upload poster."}), 400

    hasil_model = model_predict(text)
    verifikasi = local_verdict(text, hasil_model)
    label = final_label(hasil_model["label"], verifikasi["skor_hoaks"], fiksi=bool(verifikasi.get("sinyal", {}).get("fiksi_hits")))
    verifikasi["label_final"] = label
    _history.appendleft({
        "waktu": time.strftime("%H:%M:%S"),
        "cuplikan": text[:80],
        "label": label,
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
    import torch as _t
    _t.set_num_threads(max(1, (os.cpu_count() or 4) // 2))  # stabil, hemat CPU
    warmup()
    # Reloader MATI default: reloader + thread torch = crash
    # ("terminate called"). Nyalakan hanya saat ngoding UI:
    #   SAGI_RELOAD=1 .venv/bin/python web/app.py
    app.run(host="0.0.0.0", port=5000, debug=True,
            use_reloader=os.getenv("SAGI_RELOAD", "0") == "1")
