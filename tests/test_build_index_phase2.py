"""TDD tests for Phase 2: Qdrant index quality after full pipeline build.

These tests verify the Qdrant collection built by `python scripts/build_index.py`
contains real data from all 7 sources with correct structure and searchability.

Run AFTER the full pipeline has been executed.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.embeddings.qdrant_indexer import QdrantIndexer

QDRANT_PATH = config.QDRANT_PATH
COLLECTION = config.QDRANT_COLLECTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_indexer():
    return QdrantIndexer()


def _collection_exists(path: str, name: str) -> bool:
    """Check if Qdrant collection exists at the given path."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=path)
    exists = client.collection_exists(name)
    client.close()
    return exists


def _skip_if_no_collection():
    """Skip test if Qdrant collection hasn't been built yet."""
    if not os.path.isdir(QDRANT_PATH):
        pytest.skip("Qdrant directory not found — run build_index.py first")
    if not _collection_exists(QDRANT_PATH, COLLECTION):
        pytest.skip(f"Collection '{COLLECTION}' not found — run build_index.py first")


@pytest.fixture(autouse=True, scope="module")
def verify_collection_exists():
    """Ensure collection exists before running any test in this module."""
    _skip_if_no_collection()


# ---------------------------------------------------------------------------
# 1. Collection existence and status
# ---------------------------------------------------------------------------


class TestQdrantCollectionExists:
    """Verify the Qdrant collection was created and is healthy."""

    def test_collection_directory_exists(self):
        assert os.path.isdir(QDRANT_PATH), f"Qdrant directory missing: {QDRANT_PATH}"

    def test_collection_is_green(self):
        qi = _get_indexer()
        info = qi.get_collection_info(QDRANT_PATH, COLLECTION)
        assert info["status"] == "green", f"Collection status is {info['status']}, expected 'green'"

    def test_collection_has_vectors(self):
        qi = _get_indexer()
        info = qi.get_collection_info(QDRANT_PATH, COLLECTION)
        assert info["points_count"] > 0, "Collection has zero points"


# ---------------------------------------------------------------------------
# 2. Vector count ≥ 100 (from 203 raw items across 7 sources)
# ---------------------------------------------------------------------------


class TestVectorCount:
    """Verify sufficient vectors were indexed from all data sources."""

    def test_vector_count_above_100(self):
        qi = _get_indexer()
        info = qi.get_collection_info(QDRANT_PATH, COLLECTION)
        assert info["points_count"] >= 100, (
            f"Expected 100+ vectors, got {info['points_count']}. "
            f"Ensure all 7 raw data files are processed."
        )

    def test_vector_count_matches_chunks(self):
        """Vector count should match the number of chunks in chunks.json."""
        chunks_path = os.path.join(config.DATA_PROCESSED_DIR, "chunks.json")
        if not os.path.isfile(chunks_path):
            pytest.skip("chunks.json not found")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        qi = _get_indexer()
        info = qi.get_collection_info(QDRANT_PATH, COLLECTION)
        assert info["points_count"] == len(chunks), (
            f"Vector count ({info['points_count']}) != chunk count ({len(chunks)})"
        )


# ---------------------------------------------------------------------------
# 3. All chunks have required metadata fields
# ---------------------------------------------------------------------------


class TestAllMetadataFieldsPresent:
    """Verify every vector has complete metadata payload."""

    REQUIRED_FIELDS = {"section", "source_url", "source_title", "text", "chunk_index"}

    def test_all_metadata_fields_present(self):
        """Search with a generic query and verify payload fields on results."""
        qi = _get_indexer()
        # Use a random vector to get a broad sample of results
        query_vec = np.random.randn(config.QDRANT_VECTOR_SIZE).astype(np.float32).tolist()
        results = qi.search(QDRANT_PATH, COLLECTION, query_vec, limit=20)
        assert len(results) > 0, "Search returned no results"

        for hit in results:
            payload = hit["payload"]
            missing = self.REQUIRED_FIELDS - set(payload.keys())
            assert not missing, (
                f"Point {hit['id']} missing metadata fields: {missing}"
            )

    def test_metadata_text_is_nonempty(self):
        """All chunks should have non-empty text in metadata."""
        qi = _get_indexer()
        query_vec = np.random.randn(config.QDRANT_VECTOR_SIZE).astype(np.float32).tolist()
        results = qi.search(QDRANT_PATH, COLLECTION, query_vec, limit=20)

        for hit in results:
            text = hit["payload"].get("text", "")
            assert text.strip(), f"Point {hit['id']} has empty text"


# ---------------------------------------------------------------------------
# 4. Cosine similarity search returns relevant results
# ---------------------------------------------------------------------------


class TestSearchReturnsResults:
    """Verify search works with real Russian queries about MIREA."""

    def _encode_query(self, text: str) -> list[float]:
        """Encode a search query using the same model as the pipeline."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBEDDING_MODEL)
        # E5 convention: "query: " prefix for search queries
        prefixed = f"query: {text}"
        vec = model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=False)
        return vec[0].astype(np.float32).tolist()

    def test_search_russian_query_returns_results(self):
        """Search with Russian query about MIREA should return results."""
        qi = _get_indexer()
        query_vec = self._encode_query("Какие факультеты есть в МГТУ МИРЭА?")
        results = qi.search(QDRANT_PATH, COLLECTION, query_vec, limit=5)
        assert len(results) > 0, "Search returned no results for Russian query"

    def test_search_results_have_positive_scores(self):
        """All search results should have positive cosine similarity scores."""
        qi = _get_indexer()
        query_vec = self._encode_query("поступление в МИРЭА")
        results = qi.search(QDRANT_PATH, COLLECTION, query_vec, limit=5)
        assert len(results) > 0
        for hit in results:
            assert hit["score"] > 0, f"Point {hit['id']} has non-positive score: {hit['score']}"

    def test_search_returns_relevant_mirea_content(self):
        """Top results for MIREA query should contain MIREA-related text."""
        qi = _get_indexer()
        query_vec = self._encode_query("МГТУ МИРЭА факультет")
        results = qi.search(QDRANT_PATH, COLLECTION, query_vec, limit=3)
        assert len(results) > 0
        # At least one result should mention МИРЭА or МГТУ
        found_relevant = False
        for hit in results:
            text = hit["payload"].get("text", "").lower()
            if "мирэа" in text or "мгту" in text:
                found_relevant = True
                break
        assert found_relevant, "No result mentions МИРЭА/МГТУ"

    def test_different_queries_return_different_results(self):
        """Distinct queries should produce different top-1 results."""
        qi = _get_indexer()
        vec1 = self._encode_query("контакты приемной комиссии")
        vec2 = self._encode_query("расписание занятий студента")
        results1 = qi.search(QDRANT_PATH, COLLECTION, vec1, limit=1)
        results2 = qi.search(QDRANT_PATH, COLLECTION, vec2, limit=1)
        assert len(results1) > 0 and len(results2) > 0
        # Different queries should give different top results (different IDs)
        assert results1[0]["id"] != results2[0]["id"], (
            "Different queries returned the same top-1 result"
        )


# ---------------------------------------------------------------------------
# 5. Statistics verification
# ---------------------------------------------------------------------------


class TestPipelineStatistics:
    """Verify statistics about the built index match expectations."""

    def test_raw_data_has_seven_sources(self):
        """All 7 raw data files should exist."""
        raw_dir = config.DATA_RAW_DIR
        expected_files = [
            "education.json",
            "applicants.json",
            "sveden.json",
            "contacts.json",
            "qa_pairs.json",
            "deeper.json",
            "student_portal.json",
        ]
        for fname in expected_files:
            fpath = os.path.join(raw_dir, fname)
            assert os.path.isfile(fpath), f"Missing raw data file: {fname}"

    def test_raw_data_total_items_above_200(self):
        """Raw data should have 200+ items across all sources."""
        raw_dir = config.DATA_RAW_DIR
        total = 0
        for fname in os.listdir(raw_dir):
            if fname.endswith(".json") and fname != ".gitkeep":
                with open(os.path.join(raw_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                total += len(data)
        assert total >= 200, f"Expected 200+ raw items, got {total}"

    def test_vector_dimension_matches_config(self):
        """Vector size in collection should match QDRANT_VECTOR_SIZE config."""
        qi = _get_indexer()
        # Encode a query and search — if dimensions mismatch, search would fail
        query_vec = np.zeros(config.QDRANT_VECTOR_SIZE, dtype=np.float32).tolist()
        results = qi.search(QDRANT_PATH, COLLECTION, query_vec, limit=1)
        # If we get here without error, dimensions match
        assert isinstance(results, list)
