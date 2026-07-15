import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.embeddings.encoder import encode_chunks
from src.embeddings.indexer import build_index, build_qdrant_index


def build_qdrant():
    encode_chunks()
    vectors_path = os.path.join(config.DATA_FAISS_DIR, "vectors.npy")
    metadata_path = os.path.join(config.DATA_FAISS_DIR, "metadata.json")
    info = build_qdrant_index(
        vectors_path=vectors_path,
        metadata_path=metadata_path,
    )
    print(f"Qdrant collection '{config.QDRANT_COLLECTION}': {info['points_count']} points, status={info['status']}")


def build_numpy():
    encode_chunks()
    build_index()
    print(f"NumPy index built in {config.DATA_FAISS_DIR}")


def main():
    build_qdrant()


if __name__ == "__main__":
    main()
