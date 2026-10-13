import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK, CLS, SEP = "<pad>", "<unk>", "<cls>", "<sep>"
SPECIAL = [PAD, UNK, CLS, SEP]


def basic_clean(text: str) -> str:
    text = text.lower()
    # pisahkan URL, angka, emoji kasar -> token sederhana
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"[^a-z0-9\s.,!?\-<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str):
    return basic_clean(text).split()


class WordTokenizer:
    def __init__(self, vocab: dict[str, int], max_len: int = 256):
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        self.max_len = max_len
        self.pad_id = vocab[PAD]
        self.unk_id = vocab[UNK]

    @classmethod
    def build(cls, texts, vocab_size: int = 5000, max_len: int = 256):
        counter = Counter()
        for t in texts:
            counter.update(tokenize_words(t))
        most = [w for w, _ in counter.most_common(vocab_size - len(SPECIAL) - 1)]
        vocab = {tok: i for i, tok in enumerate(SPECIAL)}
        # <url> pastikan ada
        if "<url>" not in vocab:
            vocab["<url>"] = len(vocab)
        for w in most:
            if w not in vocab:
                vocab[w] = len(vocab)
        # pad sampai vocab_size agar shape embedding stabil (klaim 12M tetap valid)
        i = 0
        while len(vocab) < vocab_size:
            dummy = f"<pad_{i}>"
            if dummy not in vocab:
                vocab[dummy] = len(vocab)
            i += 1
        return cls(vocab, max_len)

    def encode(self, text: str):
        toks = tokenize_words(text)[: self.max_len - 2]
        ids = [self.vocab.get(CLS, 2)]
        ids += [self.vocab.get(t, self.unk_id) for t in toks]
        ids += [self.vocab.get(SEP, 3)]
        mask = [1] * len(ids)
        # pad
        while len(ids) < self.max_len:
            ids.append(self.pad_id)
            mask.append(0)
        return ids[: self.max_len], mask[: self.max_len]

    def save(self, path):
        Path(path).write_text(json.dumps(
            {"vocab": self.vocab, "max_len": self.max_len}, ensure_ascii=False, indent=2
        ), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], d["max_len"])
