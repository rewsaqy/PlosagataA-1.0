"""RAG lokal BM25+TF-IDF — tanpa model unduhan, CPU-only."""
from pathlib import Path
import pickle

from plosagata.slang import normalize_slang
from hoaxcore.tokenizer import basic_clean

INDEX_DIR = Path("data/rag_index")


def tokenize_id(teks: str) -> list[str]:
    return basic_clean(normalize_slang(teks or "")).split()


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
    mat = vec.fit_transform(chunks)
    return vec, mat


def save_index(path, bm25, vectorizer, tfidf_mat, chunks):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(p / "tfidf.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_mat, "chunks": chunks}, f)
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
               "matrix": tfidf_bundle["matrix"], "chunks": tfidf_bundle["chunks"]}
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
    top_idx = np.argsort(scores)[::-1][:top_n]
    top_idx = [int(i) for i in top_idx if scores[i] > 0][:top_n]
    if not top_idx:
        return []
    qvec = idx["vectorizer"].transform([" ".join(qtok)])
    cand_mat = idx["matrix"][top_idx]
    cos = cosine_similarity(qvec, cand_mat)[0]
    order = np.argsort(cos)[::-1][:k]
    out = []
    for j in order:
        ci = top_idx[int(j)]
        teks = idx["chunks"][ci]
        out.append({"id": ci, "judul": f"Korpus#{ci}", "cuplikan": teks[:220], "skor": float(cos[int(j)])})
    return out
