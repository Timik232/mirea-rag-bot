"""Build the BM25 (sparse) index from the chunk corpus and save to disk.

The BM25 index must be rebuilt whenever the chunk corpus changes
(``mirea_data/faiss/metadata.json``), exactly like the FAISS/Qdrant dense
index. The two indexes share the same positional doc ids — chunk N in
metadata.json is point N in Qdrant and doc N in BM25 — so hybrid fusion can
join results by id without any id mapping.

Usage:
    python scripts/build_bm25.py                  # default paths from config
    python scripts/build_bm25.py --rebuild-qdrant # also rebuild Qdrant (full regen)

Outputs:
    mirea_data/bm25/bm25_index.pkl   pickled BM25Index
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import config
from src.retrieval.bm25 import BM25Index, default_index_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_bm25")

METADATA_PATH = Path(config.DATA_FAISS_DIR) / "metadata.json"
OUTPUT_PATH = Path(default_index_path())


def main() -> int:
    if not METADATA_PATH.exists():
        log.error("metadata.json not found at %s — run the scraper/pipeline first", METADATA_PATH)
        return 1

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    n_chunks = len(metadata)
    log.info("Loaded %d chunks from %s", n_chunks, METADATA_PATH)

    # Sanity: every chunk must have a non-empty text field (BM25 over empty
    # docs produces zero scores and wastes vocabulary space).
    empty = [i for i, m in enumerate(metadata) if not m.get("text", "").strip()]
    if empty:
        log.warning("%d chunks have empty text: %s", len(empty), empty[:10])

    corpus = [m.get("text", "") for m in metadata]

    k1 = getattr(config, "BM25_K1", 1.5)
    b = getattr(config, "BM25_B", 0.75)
    idx = BM25Index.build(corpus, k1=k1, b=b)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(OUTPUT_PATH))

    # Verify round-trip: reload and run one query.
    log.info("Verifying index round-trip...")
    idx2 = BM25Index.load(str(OUTPUT_PATH))
    probe = "проходной балл"
    hits = idx2.search(probe, top_k=3)
    log.info("Probe query '%s' → top-3: %s", probe, [(d, round(s, 2)) for d, s in hits])
    assert idx2.n_docs == n_chunks, f"doc count mismatch: {idx2.n_docs} != {n_chunks}"
    log.info("BM25 index OK: %d docs, %d terms → %s", idx2.n_docs, len(idx2.inverted), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
