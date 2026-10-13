"""HoaxCore — klasifier hoaks 12M + util (tokenizer, fact-check, OCR)."""
from .dataset import ID2LABEL, LABEL2ID, HoaxDataset, load_csv
from .factcheck import heuristic_signals, llm_crosscheck, local_verdict
from .model import TinyHoaxTransformer, count_params
from .ocr_poster import ocr_poster
from .tokenizer import WordTokenizer, basic_clean, tokenize_words

__all__ = [
    "TinyHoaxTransformer", "count_params",
    "WordTokenizer", "basic_clean", "tokenize_words",
    "HoaxDataset", "load_csv", "LABEL2ID", "ID2LABEL",
    "local_verdict", "heuristic_signals", "llm_crosscheck",
    "ocr_poster",
]
