import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.implementations import ByteLevelBPETokenizer

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
    """Byte-Pair Encoding (BPE) tokenizer with drop-in WordTokenizer API compatibility.

    Uses HuggingFace tokenizers ByteLevelBPETokenizer internally to prevent OOV issues
    on Indonesian affixes, slang, typos, and Unicode characters while preserving the
    public interface expected by the hoax-validator models.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        max_len: int = 256,
        tokenizer: Tokenizer | None = None,
        is_legacy: bool = False,
    ):
        self.vocab = dict(vocab)
        self.inv = {i: t for t, i in self.vocab.items()}
        self.max_len = max_len
        self.pad_id = self.vocab.get(PAD, 0)
        self.unk_id = self.vocab.get(UNK, 1)
        self.cls_id = self.vocab.get(CLS, 2)
        self.sep_id = self.vocab.get(SEP, 3)
        self._tokenizer = tokenizer
        self._is_legacy = is_legacy

    def __len__(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)

    def id_to_token(self, idx: int) -> str | None:
        return self.inv.get(idx)

    @classmethod
    def build(
        cls,
        texts: Iterable[str] | str,
        vocab_size: int = 5000,
        max_len: int = 256,
    ) -> "WordTokenizer":
        if isinstance(texts, str):
            raw_texts = [texts]
        else:
            raw_texts = [str(t) for t in texts if t is not None]

        if not raw_texts:
            raw_texts = [" "]

        bpe = ByteLevelBPETokenizer(add_prefix_space=False, lowercase=False)
        special_tokens = [PAD, UNK, CLS, SEP, "<url>"]

        # Train BPE model from scratch on provided texts
        bpe.train_from_iterator(
            raw_texts,
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=False,
            special_tokens=special_tokens,
        )

        # Pad vocabulary up to vocab_size so embedding layer shapes remain stable
        current_vocab = bpe.get_vocab()
        current_size = len(current_vocab)
        if current_size < vocab_size:
            needed = vocab_size - current_size
            dummies = []
            i = 0
            while len(dummies) < needed:
                dummy = f"<pad_{i}>"
                if dummy not in current_vocab:
                    dummies.append(dummy)
                i += 1
            bpe.add_special_tokens(dummies)

        vocab = {k: v for k, v in sorted(bpe.get_vocab().items(), key=lambda item: item[1])}
        return cls(vocab=vocab, max_len=max_len, tokenizer=bpe._tokenizer, is_legacy=False)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        text = str(text) if text is not None else ""
        if self._tokenizer is not None and not self._is_legacy:
            sub_ids = self._tokenizer.encode(text).ids
        else:
            toks = tokenize_words(text)
            sub_ids = [self.vocab.get(t, self.unk_id) for t in toks]

        # Truncate content tokens to reserve slots for CLS and SEP
        max_content = max(0, self.max_len - 2)
        sub_ids = sub_ids[:max_content]

        ids = [self.cls_id] + sub_ids + [self.sep_id]
        mask = [1] * len(ids)

        if len(ids) < self.max_len:
            pad_count = self.max_len - len(ids)
            ids.extend([self.pad_id] * pad_count)
            mask.extend([0] * pad_count)

        return ids[: self.max_len], mask[: self.max_len]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)

        if self._tokenizer is not None and not self._is_legacy:
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        special_names = {PAD, UNK, CLS, SEP, "<url>"}
        words = []
        for i in ids:
            t = self.inv.get(i, "")
            if skip_special_tokens:
                if t in special_names or t.startswith("<pad"):
                    continue
            words.append(t)
        return " ".join(words).strip()

    def save(self, path: str | Path):
        data = {
            "vocab": self.vocab,
            "max_len": self.max_len,
        }
        if self._tokenizer is not None and not self._is_legacy:
            data["tokenizer_json"] = json.loads(self._tokenizer.to_str())

        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        content = Path(path).read_text(encoding="utf-8")
        d = json.loads(content)
        max_len = d.get("max_len", 256)
        vocab = d.get("vocab")

        backend_tok = None
        is_legacy = False

        tokenizer_json = d.get("tokenizer_json")
        if tokenizer_json is not None:
            if isinstance(tokenizer_json, dict):
                tokenizer_str = json.dumps(tokenizer_json)
            else:
                tokenizer_str = str(tokenizer_json)
            backend_tok = Tokenizer.from_str(tokenizer_str)
            if vocab is None:
                vocab = backend_tok.get_vocab()
        elif "model" in d and ("version" in d or "type" in d.get("model", {})):
            # Direct HuggingFace Tokenizer JSON export
            backend_tok = Tokenizer.from_str(content)
            if vocab is None:
                vocab = backend_tok.get_vocab()
        else:
            # Legacy vocab file without tokenizer_json
            is_legacy = True
            if vocab is None:
                vocab = {}
            try:
                backend_tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
                backend_tok.pre_tokenizer = pre_tokenizers.Whitespace()
            except Exception:
                backend_tok = None

        if vocab is None:
            vocab = {}

        return cls(vocab=vocab, max_len=max_len, tokenizer=backend_tok, is_legacy=is_legacy)
