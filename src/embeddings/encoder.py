"""Encode text chunks into embeddings using sentence-transformers."""

import json
import logging
import os

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 16


def encode_chunks(chunks_path: str | None = None, output_dir: str | None = None) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Load chunks, encode with E5 model, save vectors and metadata.

    Args:
        chunks_path: Path to chunks.json. Defaults to config.DATA_PROCESSED_DIR/chunks.json.
        output_dir: Directory for output files. Defaults to config.DATA_FAISS_DIR.

    Returns:
        Tuple of (embeddings array, metadata list).
    """
    if chunks_path is None:
        chunks_path = os.path.join(config.DATA_PROCESSED_DIR, "chunks.json")
    if output_dir is None:
        output_dir = config.DATA_FAISS_DIR

    os.makedirs(output_dir, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    logger.info("Loading model %s ...", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # E5 convention: prefix with "passage: " for indexing
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    all_embeddings: list[np.ndarray] = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False)
        all_embeddings.append(batch_emb)
        logger.info("Encoded batch %d/%d", i // BATCH_SIZE + 1, n_batches)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info("Encoded %d chunks in %d batches", len(chunks), n_batches)

    vectors_path = os.path.join(output_dir, "vectors.npy")
    np.save(vectors_path, embeddings)
    logger.info("Saved vectors %s shape=%s dtype=%s", vectors_path, embeddings.shape, embeddings.dtype)

    # Metadata ordering must match vector ordering for correct retrieval
    metadata = [
        {
            "source_title": chunk["source_title"],
            "source_url": chunk["source_url"],
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata %s (%d entries)", metadata_path, len(metadata))

    return embeddings, metadata


if __name__ == "__main__":
    encode_chunks()
