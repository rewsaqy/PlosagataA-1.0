"""RAG lokal BM25+TF-IDF — tanpa model unduhan, CPU-only."""
from pathlib import Path
import pickle

from plosagata.slang import normalize_slang
from hoaxcore.tokenizer import STOPWORDS, basic_clean

INDEX_DIR = Path("data/rag_index")
MIN_SKOR = 0.25  # cosine di bawah ini = tak relevan -> fallback live
# (0.2 masih lolos untuk 1 kata bersama, mis. "mewah"; kasus baik >= 0.35)


def tokenize_id(teks: str) -> list[str]:
    return [w for w in basic_clean(normalize_slang(teks or "")).split()
            if w not in STOPWORDS]


def chunk_teks(teks: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = (teks or "").split()
    if not words:
        return []
    out = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        c = words[i:i + chunk_size]
        if not c:
            break
        out.append(" ".join(c))
        if i + chunk_size >= len(words):
            break
    return out


def build_bm25(tokenized):
    from rank_bm25 import BM25Okapi
    return BM25Okapi(tokenized)


def build_tfidf(chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    normed = [" ".join(tokenize_id(c)) for c in chunks]
    mat = vec.fit_transform(normed)
    return vec, mat


def save_index(path, bm25, vectorizer, tfidf_mat, chunks, titles=None):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(p / "tfidf.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_mat,
                     "chunks": chunks, "titles": titles or []}, f)
    (p / "meta.json").write_text('{"chunk":300,"overlap":50}', encoding="utf-8")


def _load_pickle(p):
    import pickle as _pk
    with open(p, "rb") as f:
        return _pk.load(f)


_CACHE = {}


def load_index(path=None):
    from pathlib import Path as _P
    d = _P(path or INDEX_DIR)
    if str(d) in _CACHE:
        return _CACHE[str(d)]
    try:
        bm25 = _load_pickle(d / "bm25.pkl")
        tfidf_bundle = _load_pickle(d / "tfidf.pkl")
        idx = {"bm25": bm25, "vectorizer": tfidf_bundle["vectorizer"],
               "matrix": tfidf_bundle["matrix"], "chunks": tfidf_bundle["chunks"],
               "titles": tfidf_bundle.get("titles") or []}
    except Exception:
        return None
    _CACHE[str(d)] = idx
    return idx


def rag_search(query: str, k: int = 3):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    q = (query or "").strip()
    if len(q) < 2:
        return []
    idx = load_index()
    if idx is None:
        return []
    qtok = tokenize_id(q)
    if not qtok:
        return []
    scores = idx["bm25"].get_scores(qtok)
    top_n = min(20, len(scores))
    # Catatan: jangan filter scores>0 — IDF BM25 bisa 0 untuk term yang muncul
    # di setengah korpus (mis. demo kecil), kandidat tetap di-rerank TF-IDF.
    top_idx = [int(i) for i in np.argsort(scores)[::-1][:top_n]]
    if not top_idx:
        return []
    qvec = idx["vectorizer"].transform([" ".join(qtok)])
    cand_mat = idx["matrix"][top_idx]
    cos = cosine_similarity(qvec, cand_mat)[0]
    order = np.argsort(cos)[::-1][:k]
    out = []
    for j in order:
        if cos[int(j)] < MIN_SKOR:
            continue  # tak relevan -> fallback live yang bicara
        ci = top_idx[int(j)]
        teks = idx["chunks"][ci]
        titles = idx.get("titles") or []
        judul = titles[ci] if ci < len(titles) else f"Korpus#{ci}"
        out.append({"id": ci, "judul": judul, "cuplikan": teks[:220], "skor": float(cos[int(j)])})
    return out
