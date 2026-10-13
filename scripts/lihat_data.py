"""Cermin baca data JSONL -> tabel Markdown.

  python3 scripts/lihat_data.py
  -> data/contoh_chat.md, data/chat_hoaks.md (enak dibaca di GitHub)

Jalankan tiap habis nambah data. File .md = cermin saja,
yang dilatih tetap .jsonl-nya.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def md_table(rows, judul):
    out = [f"# {judul}\n", f"Total: **{len(rows)}** contoh.\n",
           "| No | User | Assistant |", "|---|---|---|"]
    for i, r in enumerate(rows, 1):
        msgs = {m["role"]: m["content"].replace("|", "/").replace("\n", " ")
                for m in r["messages"] if m["role"] in ("user", "assistant")}
        out.append(f"| {i} | {msgs.get('user','')} | {msgs.get('assistant','')} |")
    return "\n".join(out) + "\n"


def main():
    for src, judul in [("data/contoh_chat.jsonl", "Contoh Chat Sagi (tulis manual)"),
                       ("data/chat_hoaks.jsonl", "Contoh Chat dari CSV (otomatis)")]:
        p = ROOT / src
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = p.with_suffix(".md")
        out.write_text(md_table(rows, judul), encoding="utf-8")
        print(f"{out.name}: {len(rows)} baris")


if __name__ == "__main__":
    main()
