"""HoaxCore — klasifier hoaks 12M + util (tokenizer, fact-check, OCR)."""
from .dataset import ID2LABEL, LABEL2ID, HoaxDataset, load_csv
from .factcheck import (deteksi_fiksi, final_label, format_creator_script,
                         heuristic_signals, llm_crosscheck, local_verdict,
                         ringkasan_hasil, susun_jawaban_cerdas)
from .shared import get_classifier, predict_proba
from .model import TinyHoaxTransformer, count_params
from .ocr_poster import ocr_poster
from .retrieval import cari_sumber, kata_kunci, link_cek_fakta, saring_relevan
from .rag import rag_search
from .tokenizer import (WordTokenizer, basic_clean, piece_decode, piece_encode,
                         tokenize_words)

__all__ = [
    "TinyHoaxTransformer", "count_params",
    "WordTokenizer", "basic_clean", "piece_decode", "piece_encode",
    "tokenize_words",
    "HoaxDataset", "load_csv", "LABEL2ID", "ID2LABEL",
    "local_verdict", "heuristic_signals", "llm_crosscheck",
    "format_creator_script",
    "get_classifier", "predict_proba", "ringkasan_hasil",
    "susun_jawaban_cerdas",
    "deteksi_fiksi", "final_label",
    "ocr_poster",
    "cari_sumber", "kata_kunci", "link_cek_fakta", "saring_relevan",
    "rag_search",
]
