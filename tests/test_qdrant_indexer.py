"""Tests for QdrantIndexer — TDD RED phase."""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings.qdrant_indexer import QdrantIndexer


@pytest.fixture
def tmp_qdrant(tmp_path):
    """Provide a temporary Qdrant storage path, cleaned up after test."""
    qdrant_dir = tmp_path / "qdrant_test"
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    yield str(qdrant_dir)
    if qdrant_dir.exists():
        shutil.rmtree(qdrant_dir, ignore_errors=True)


@pytest.fixture
def indexer():
    return QdrantIndexer()


VECTOR_SIZE = 384
COLLECTION = "test_chunks"


def _make_vector(seed=0):
    """Generate a deterministic 384-dim float32 vector."""
    import random
    rng = random.Random(seed)
    return [rng.random() for _ in range(VECTOR_SIZE)]


def _make_vectors(n, start_seed=0):
    return [_make_vector(seed) for seed in range(start_seed, start_seed + n)]


def _make_payload(i):
    """Generate a standard chunk payload."""
    return {
        "text": f"chunk text {i}",
        "section": f"section_{i % 3}",
        "subsection": f"subsection_{i % 2}",
        "url": f"https://example.com/page/{i}",
        "source": "test_source",
        "chunk_index": i,
    }


# ── create_collection tests ──


class TestCreateCollection:
    def test_creates_collection_successfully(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert info["status"] == "green"
        assert info["points_count"] == 0

    def test_recreate_existing_collection(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        indexer.upsert_chunks(
            tmp_qdrant, COLLECTION, [1], [_make_vector(0)], [_make_payload(0)]
        )
        # Recreate — should wipe data
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert info["points_count"] == 0

    def test_custom_vector_size(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, "small_vec", 128)
        info = indexer.get_collection_info(tmp_qdrant, "small_vec")
        assert info["status"] == "green"


# ── upsert_chunks tests ──


class TestUpsertChunks:
    def test_single_upsert(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        ids = [42]
        vectors = [_make_vector(0)]
        payloads = [_make_payload(0)]
        indexer.upsert_chunks(tmp_qdrant, COLLECTION, ids, vectors, payloads)
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert info["points_count"] == 1

    def test_multiple_upserts(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        n = 5
        ids = list(range(n))
        vectors = _make_vectors(n)
        payloads = [_make_payload(i) for i in range(n)]
        indexer.upsert_chunks(tmp_qdrant, COLLECTION, ids, vectors, payloads)
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert info["points_count"] == n

    def test_batch_upsert_over_100(self, indexer, tmp_qdrant):
        """Verify batch upsert works for more than 100 items (batch_size=100)."""
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        n = 250
        ids = list(range(n))
        vectors = _make_vectors(n)
        payloads = [_make_payload(i) for i in range(n)]
        indexer.upsert_chunks(tmp_qdrant, COLLECTION, ids, vectors, payloads)
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert info["points_count"] == n

    def test_upsert_payload_structure(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        payload = _make_payload(0)
        indexer.upsert_chunks(
            tmp_qdrant, COLLECTION, [1], [_make_vector(0)], [payload]
        )
        results = indexer.search(tmp_qdrant, COLLECTION, _make_vector(0), limit=1)
        assert len(results) == 1
        hit = results[0]
        assert hit["payload"]["text"] == payload["text"]
        assert hit["payload"]["section"] == payload["section"]
        assert hit["payload"]["subsection"] == payload["subsection"]
        assert hit["payload"]["url"] == payload["url"]
        assert hit["payload"]["source"] == payload["source"]
        assert hit["payload"]["chunk_index"] == payload["chunk_index"]


# ── search tests ──


class TestSearch:
    def test_basic_search(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        # Insert a few chunks
        for i in range(3):
            indexer.upsert_chunks(
                tmp_qdrant, COLLECTION, [i], [_make_vector(i)], [_make_payload(i)]
            )
        results = indexer.search(tmp_qdrant, COLLECTION, _make_vector(0), limit=3)
        assert len(results) == 3
        assert results[0]["id"] == 0

    def test_search_limit(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        n = 10
        ids = list(range(n))
        vectors = _make_vectors(n)
        payloads = [_make_payload(i) for i in range(n)]
        indexer.upsert_chunks(tmp_qdrant, COLLECTION, ids, vectors, payloads)
        results = indexer.search(tmp_qdrant, COLLECTION, _make_vector(0), limit=3)
        assert len(results) == 3

    def test_search_returns_scores(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        indexer.upsert_chunks(
            tmp_qdrant, COLLECTION, [0], [_make_vector(0)], [_make_payload(0)]
        )
        results = indexer.search(tmp_qdrant, COLLECTION, _make_vector(0), limit=1)
        assert len(results) == 1
        assert "score" in results[0]
        assert results[0]["score"] > 0.99  # identical vector → cosine ~1.0

    def test_search_empty_collection(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        results = indexer.search(tmp_qdrant, COLLECTION, _make_vector(0), limit=3)
        assert results == []


# ── get_collection_info tests ──


class TestGetCollectionInfo:
    def test_empty_collection(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert "points_count" in info
        assert "status" in info
        assert info["points_count"] == 0

    def test_after_upsert(self, indexer, tmp_qdrant):
        indexer.create_collection(tmp_qdrant, COLLECTION, VECTOR_SIZE)
        indexer.upsert_chunks(
            tmp_qdrant, COLLECTION, [1], [_make_vector(0)], [_make_payload(0)]
        )
        info = indexer.get_collection_info(tmp_qdrant, COLLECTION)
        assert info["points_count"] == 1
        assert info["status"] == "green"
