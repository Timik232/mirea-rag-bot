"""Build vector index from pre-computed embedding vectors.

Uses FAISS when available (Jetson / production), falls back to numpy
(cosine similarity via matrix multiply) when FAISS cannot be imported
(dev machine, Python 3.14).
"""

import json
import logging
import os
import subprocess
import sys

import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection  — uses a *subprocess* probe so that a core-dumping or
# hanging ``import faiss`` cannot bring down the main process.
# ---------------------------------------------------------------------------
_faiss_available: bool | None = None
_faiss_mod = None


def _detect_faiss() -> bool:
    """Return True if faiss can be imported without crashing / hanging.

    The first call spawns a short-lived child process that tries to import
    faiss.  If the child exits with code 0 within 5 s we know faiss is safe
    to import in the parent as well.
    """
    global _faiss_available, _faiss_mod
    if _faiss_available is not None:
        return _faiss_available
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import faiss"],
            timeout=5,
            capture_output=True,
        )
        if result.returncode == 0:
            import faiss as _f  # noqa: F811

            _faiss_mod = _f
            _faiss_available = True
            logger.info("Using FAISS backend")
        else:
            _faiss_available = False
            logger.info("Using numpy fallback backend (FAISS import failed)")
    except Exception:
        _faiss_available = False
        logger.info("Using numpy fallback backend (FAISS unavailable)")
    return _faiss_available


# ---------------------------------------------------------------------------
# Numpy fallback helpers
# ---------------------------------------------------------------------------

class NumpyIndex:
    """Minimal FAISS-like index backed by numpy cosine similarity."""

    def __init__(self, vectors: np.ndarray):
        # Store L2-normalised vectors; cosine sim == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.vectors = (vectors / norms).astype(np.float32)
        self.count = len(vectors)

    def search(self, query: np.ndarray, top_k: int):
        """Return (distances, indices) shaped (1, top_k)."""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        # Normalise query
        qnorm = np.linalg.norm(query, axis=1, keepdims=True)
        qnorm = np.maximum(qnorm, 1e-10)
        query_n = (query / qnorm).astype(np.float32)

        scores = query_n @ self.vectors.T  # (1, N)
        # Top-k via argpartition (faster than full sort for large N)
        k = min(top_k, self.count)
        if k >= self.count:
            idx = np.argsort(-scores[0])
        else:
            part = np.argpartition(-scores[0], k)[:k]
            idx = part[np.argsort(-scores[0][part])]
        distances = scores[0][idx].reshape(1, -1)
        indices = idx.reshape(1, -1)
        return distances, indices

    def save(self, path: str):
        """Save vectors as .npz with a small manifest."""
        np.savez(path, vectors=self.vectors)
        manifest_path = path.rsplit(".", 1)[0] + "_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {"type": "numpy", "dimensions": self.vectors.shape[1], "count": self.count},
                f,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(vectors_path: str | None = None, output_dir: str | None = None):
    """Load vectors, build index, and save.

    Args:
        vectors_path: Path to vectors.npy. Defaults to config.DATA_FAISS_DIR/vectors.npy.
        output_dir: Directory for index output. Defaults to config.DATA_FAISS_DIR.

    Returns:
        The built index (FAISS IndexFlatIP or NumpyIndex).
    """
    if vectors_path is None:
        vectors_path = os.path.join(config.DATA_FAISS_DIR, "vectors.npy")
    if output_dir is None:
        output_dir = config.DATA_FAISS_DIR

    os.makedirs(output_dir, exist_ok=True)

    vectors = np.load(vectors_path).astype(np.float32)
    logger.info("Loaded vectors %s shape=%s", vectors_path, vectors.shape)

    if _detect_faiss():
        return _build_faiss(vectors, output_dir)
    return _build_numpy(vectors, output_dir)


def _build_faiss(vectors: np.ndarray, output_dir: str):
    """Build and save a FAISS IndexFlatIP."""
    assert _faiss_mod is not None, "FAISS not available"
    faiss = _faiss_mod
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(config.FAISS_DIMENSIONS)
    index.add(vectors)
    logger.info("Indexed %d vectors into FAISS", index.ntotal)

    index_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, index_path)
    logger.info("Saved FAISS index to %s", index_path)

    return index


def _build_numpy(vectors: np.ndarray, output_dir: str):
    """Build and save a numpy-backed index."""
    np_index = NumpyIndex(vectors)
    logger.info("Indexed %d vectors into numpy index", np_index.count)

    index_path = os.path.join(output_dir, "index.npz")
    np_index.save(index_path)
    logger.info("Saved numpy index to %s", index_path)

    return np_index


def search(query_vector: np.ndarray, top_k: int = config.TOP_K_RETRIEVAL, index_dir: str | None = None):
    """Load the saved index and search for top_k nearest neighbours.

    Works with both FAISS (.faiss) and numpy (.npz) backends.

    Args:
        query_vector: 1-D array of shape (dimensions,).
        top_k: Number of results to return.
        index_dir: Directory containing the index files.

    Returns:
        Tuple (distances, indices) — distances and indices of top_k results.
    """
    if index_dir is None:
        index_dir = config.DATA_FAISS_DIR

    if _detect_faiss():
        return _search_faiss(query_vector, top_k, index_dir)
    return _search_numpy(query_vector, top_k, index_dir)


def _search_faiss(query_vector: np.ndarray, top_k: int, index_dir: str):
    assert _faiss_mod is not None, "FAISS not available"
    faiss = _faiss_mod
    index_path = os.path.join(index_dir, "index.faiss")
    index = faiss.read_index(index_path)
    q = query_vector.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    distances, indices = index.search(q, top_k)
    return distances, indices


def _search_numpy(query_vector: np.ndarray, top_k: int, index_dir: str):
    index_path = os.path.join(index_dir, "index.npz")
    data = np.load(index_path)
    np_index = NumpyIndex(data["vectors"])
    return np_index.search(query_vector.astype(np.float32), top_k)


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------

def build_qdrant_index(
    vectors_path: str | None = None,
    metadata_path: str | None = None,
    qdrant_path: str | None = None,
    collection_name: str | None = None,
    vector_size: int | None = None,
):
    """Build a Qdrant index from pre-computed vectors and metadata.

    Args:
        vectors_path: Path to vectors.npy.
        metadata_path: Path to metadata.json.
        qdrant_path: Qdrant storage path. Defaults to config.QDRANT_PATH.
        collection_name: Qdrant collection name. Defaults to config.QDRANT_COLLECTION.
        vector_size: Vector dimensionality. Defaults to config.QDRANT_VECTOR_SIZE.

    Returns:
        dict with collection info.
    """
    from src.embeddings.qdrant_indexer import QdrantIndexer

    if vectors_path is None:
        vectors_path = os.path.join(config.DATA_FAISS_DIR, "vectors.npy")
    if metadata_path is None:
        metadata_path = os.path.join(config.DATA_FAISS_DIR, "metadata.json")
    if qdrant_path is None:
        qdrant_path = config.QDRANT_PATH
    if collection_name is None:
        collection_name = config.QDRANT_COLLECTION
    if vector_size is None:
        vector_size = config.QDRANT_VECTOR_SIZE

    vectors = np.load(vectors_path).astype(np.float32)
    logger.info("Loaded vectors %s shape=%s", vectors_path, vectors.shape)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info("Loaded metadata %s (%d entries)", metadata_path, len(metadata))

    assert len(vectors) == len(metadata), (
        f"Vector count ({len(vectors)}) != metadata count ({len(metadata)})"
    )

    qi = QdrantIndexer()
    qi.create_collection(qdrant_path, collection_name, vector_size)
    logger.info("Created Qdrant collection '%s' at %s", collection_name, qdrant_path)

    ids = list(range(len(vectors)))
    vecs = vectors.tolist()
    payloads = metadata

    qi.upsert_chunks(qdrant_path, collection_name, ids, vecs, payloads)
    logger.info("Upserted %d points into '%s'", len(vectors), collection_name)

    info = qi.get_collection_info(qdrant_path, collection_name)
    logger.info("Collection info: %s", info)
    return info


def search_qdrant(
    query_vector: np.ndarray,
    qdrant_path: str | None = None,
    collection_name: str | None = None,
    top_k: int = config.TOP_K_RETRIEVAL,
) -> list[dict]:
    """Search Qdrant collection for top_k nearest neighbours.

    Args:
        query_vector: 1-D array of shape (dimensions,).
        qdrant_path: Qdrant storage path. Defaults to config.QDRANT_PATH.
        collection_name: Qdrant collection name. Defaults to config.QDRANT_COLLECTION.
        top_k: Number of results to return.

    Returns:
        List of dicts with id, score, payload.
    """
    from src.embeddings.qdrant_indexer import QdrantIndexer

    if qdrant_path is None:
        qdrant_path = config.QDRANT_PATH
    if collection_name is None:
        collection_name = config.QDRANT_COLLECTION

    qi = QdrantIndexer()
    query = query_vector.astype(np.float32).tolist()
    results = qi.search(qdrant_path, collection_name, query, limit=top_k)
    logger.info("Qdrant search returned %d results", len(results))
    return results


if __name__ == "__main__":
    build_index()
