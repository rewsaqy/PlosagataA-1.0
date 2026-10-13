"""Smoke + eval RAG (dibuat via TDD Task 1)."""
import sys
sys.path.insert(0, ".")
from hoaxcore.rag import chunk_teks, tokenize_id
assert chunk_teks("kata " * 650, chunk_size=300, overlap=50) is not None
assert len(chunk_teks("kata " * 650, chunk_size=300, overlap=50)) == 3, "650 kata /300 overlap50 harus 3 chunk"
assert tokenize_id("Gaskeun ngab, SPILL Giveaway!") != []
print("rag-smoke OK")
