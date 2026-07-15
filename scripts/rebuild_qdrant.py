"""Rebuild the Qdrant collection from current faiss data (fix staleness).

The Qdrant local collection can drift out of sync with mirea_data/faiss/
when the corpus is regenerated without also rebuilding Qdrant (see parent
task t_8ab04e61 infra finding). This script rebuilds it in-place — no
re-encoding, just re-upserts the existing vectors+metadata.
"""
import logging, sys
from pathlib import Path
PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
import config
from src.embeddings.indexer import build_qdrant_index
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Drop existing collection first (clean slate)
qc = QdrantClient(path=config.QDRANT_PATH)
try:
    qc.delete_collection(config.QDRANT_COLLECTION)
    print(f"dropped old collection '{config.QDRANT_COLLECTION}'")
except Exception:
    pass
qc.close()

info = build_qdrant_index()
print(f"rebuilt qdrant: {info}")
