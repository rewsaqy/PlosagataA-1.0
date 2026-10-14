import csv
import torch
from torch.utils.data import Dataset


LABEL2ID = {"valid": 0, "hoaks": 1, "meragukan": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_csv(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        assert "text" in r.fieldnames and "label" in r.fieldnames, \
            "CSV harus punya kolom text,label"
        for row in r:
            lab = row["label"].strip().lower()
            if lab not in LABEL2ID:
                continue
            texts.append(row["text"])
            labels.append(LABEL2ID[lab])
    return texts, labels


class HoaxDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        # Slang dinormalisasi di train DAN inferensi (konsisten).
        # Import lokal agar dataset tetap ringan tanpa plosagata.
        try:
            from plosagata.slang import normalize_slang
            teks = normalize_slang(self.texts[i])
        except Exception:
            teks = self.texts[i]
        ids, mask = self.tok.encode(teks)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
        )
