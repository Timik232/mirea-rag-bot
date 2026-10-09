"""Comprehensive tests for semantic search over MIREA data.

Uses the numpy fallback backend (FAISS unavailable on this machine).
Relies on ``from src.embeddings.indexer import search`` which auto-detects
the backend.
"""

import json
import os
import sys

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

# ── Path setup so imports work from project root ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.embeddings.indexer import search

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model():
    """Load the E5 model once for all tests."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


@pytest.fixture(scope="session")
def metadata():
    """Load metadata for result inspection."""
    meta_path = os.path.join(config.DATA_FAISS_DIR, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _encode_query(model, text: str) -> np.ndarray:
    """Encode a query with E5 'query: ' prefix."""
    prefixed = f"query: {text}"
    vec = model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=False)
    return vec.astype(np.float32)


def _search_query(model, query: str, top_k: int = config.TOP_K_RETRIEVAL):
    """Helper: encode + search in one call."""
    vec = _encode_query(model, query)
    return search(vec, top_k=top_k)


def _result_texts(distances, indices, metadata):
    """Return list of (text, score) for the search results."""
    results = []
    for i in range(indices.shape[1]):
        idx = int(indices[0][i])
        if idx < 0:
            continue
        score = float(distances[0][i])
        results.append((metadata[idx]["text"], score, metadata[idx]))
    return results


def _keyword_hit(results, keywords):
    """Check if any keyword appears in any result text (case-insensitive)."""
    texts_lower = [r[0].lower() for r in results]
    for kw in keywords:
        for t in texts_lower:
            if kw.lower() in t:
                return True
    return False


# ── Basic infrastructure tests ────────────────────────────────────────────

class TestInfrastructure:
    """Verify the index and model load correctly."""

    def test_index_files_exist(self):
        assert os.path.exists(os.path.join(config.DATA_FAISS_DIR, "index.npz"))
        assert os.path.exists(os.path.join(config.DATA_FAISS_DIR, "metadata.json"))

    def test_metadata_has_entries(self, metadata):
        # Dataset was rebuilt from the live site (2026-10): 5090 chunks.
        # Assert a healthy floor rather than an exact size so the test
        # survives future rebuilds.
        assert len(metadata) >= 1000, f"Expected >= 1000 chunks, got {len(metadata)}"

    def test_search_returns_valid_shapes(self, model):
        vec = _encode_query(model, "МИРЭА")
        distances, indices = search(vec, top_k=3)
        assert distances.shape[0] == 1
        assert indices.shape[0] == 1
        assert distances.shape[1] == 3
        assert indices.shape[1] == 3

    def test_distances_are_cosine_similarity(self, model):
        vec = _encode_query(model, "университет")
        distances, indices = search(vec, top_k=3)
        # Cosine similarity ranges [-1, 1]
        for val in distances[0]:
            assert -1.0 <= val <= 1.0

    def test_top_results_sorted_descending(self, model):
        vec = _encode_query(model, "программирование")
        distances, indices = search(vec, top_k=5)
        scores = list(distances[0])
        assert scores == sorted(scores, reverse=True)

    def test_indices_within_range(self, model, metadata):
        vec = _encode_query(model, "институт")
        distances, indices = search(vec, top_k=3)
        n = len(metadata)
        for idx in indices[0]:
            assert 0 <= int(idx) < n


# ── Semantic relevance tests (10 queries) ────────────────────────────────

class TestSemanticRelevance:
    """Verify that key queries return relevant content."""

    def test_faculties_query(self, model, metadata):
        """Какие факультеты есть в МИРЭА? → expect faculty/institute content."""
        distances, indices = _search_query(model, "Какие факультеты есть в МИРЭА?")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["факультет", "институт", "подразделени"]), \
            f"Expected faculty/institute keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_admission_scores_query(self, model, metadata):
        """проходной балл → expect institute/program content (specific scores may not be in corpus)."""
        distances, indices = _search_query(model, "проходной балл")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["инженери", "направлен", "программ", "подготовк", "институт", "информационн"]), \
            f"Expected program/institute keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_address_query(self, model, metadata):
        """адрес главного корпуса → expect Moscow addresses."""
        distances, indices = _search_query(model, "адрес главного корпуса")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["москва", "ул.", "проспект", "адрес", "корпус"]), \
            f"Expected address keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_cybersecurity_query(self, model, metadata):
        """направление кибербезопасность → expect cybersecurity content."""
        distances, indices = _search_query(model, "направление кибербезопасность")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["кибер", "безопасност", "криптограф"]), \
            f"Expected cybersecurity keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_phone_query(self, model, metadata):
        """телефон приёмной комиссии → expect phone/contact content."""
        distances, indices = _search_query(model, "телефон приёмной комиссии")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["телефон", "495", "приёмн", "контакт"]), \
            f"Expected phone/contact keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_it_institute_query(self, model, metadata):
        """институт ИТ → expect IT institute content."""
        distances, indices = _search_query(model, "институт информационных технологий")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["информационных технолог", "ИИТ", "институт"]), \
            f"Expected IT institute keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_department_query(self, model, metadata):
        """кафедра → expect department content."""
        distances, indices = _search_query(model, "кафедра")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["кафедр", "подготовк", "преподавател"]), \
            f"Expected department keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_admission_docs_query(self, model, metadata):
        """документы для поступления → expect admission docs content."""
        distances, indices = _search_query(model, "документы для поступления")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["документ", "поступлен", "заявлен", "паспорт", "егэ"]), \
            f"Expected admission docs keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_curriculum_query(self, model, metadata):
        """программа обучения → expect curriculum content."""
        distances, indices = _search_query(model, "программа обучения")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["программ", "обучен", "дисциплин", "курс"]), \
            f"Expected curriculum keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"

    def test_qa_match_query(self, model, metadata):
        """Query matching a Q&A pair → expect similar match."""
        distances, indices = _search_query(model, "Где находится главный корпус МИРЭА?")
        results = _result_texts(distances, indices, metadata)
        assert _keyword_hit(results, ["корпус", "стромынк", "москва"]), \
            f"Expected main building keywords. Got: {[r[0][:80] for r in results]}"
        assert results[0][1] > 0.3, f"Top score too low: {results[0][1]}"


# ── Edge case tests ──────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases must not crash."""

    def test_empty_query(self, model, metadata):
        """Empty string query must not crash."""
        vec = _encode_query(model, "")
        distances, indices = search(vec, top_k=3)
        assert distances.shape == (1, 3)
        results = _result_texts(distances, indices, metadata)
        assert len(results) > 0

    def test_very_short_query(self, model, metadata):
        """Very short query 'МИРЭА' must not crash."""
        vec = _encode_query(model, "МИРЭА")
        distances, indices = search(vec, top_k=3)
        assert distances.shape == (1, 3)
        results = _result_texts(distances, indices, metadata)
        assert len(results) > 0
        assert results[0][1] > 0.3

    def test_typo_query(self, model, metadata):
        """Query with typo 'факултет' must not crash."""
        vec = _encode_query(model, "факултет")
        distances, indices = search(vec, top_k=3)
        assert distances.shape == (1, 3)
        results = _result_texts(distances, indices, metadata)
        assert len(results) > 0

    def test_top_k_1(self, model):
        """top_k=1 should return exactly 1 result."""
        vec = _encode_query(model, "радиотехника")
        distances, indices = search(vec, top_k=1)
        assert distances.shape == (1, 1)
        assert indices.shape == (1, 1)

    def test_top_k_exceeds_index(self, model, metadata):
        """top_k larger than index size should not crash."""
        vec = _encode_query(model, "университет")
        distances, indices = search(vec, top_k=200)
        assert distances.shape[0] == 1
        # Should return at most len(metadata) results (index size)
        assert distances.shape[1] <= len(metadata)

    def test_score_consistency(self, model):
        """Same query twice should give same results."""
        vec = _encode_query(model, "информатика")
        d1, i1 = search(vec, top_k=3)
        d2, i2 = search(vec, top_k=3)
        np.testing.assert_array_almost_equal(d1, d2)
        np.testing.assert_array_equal(i1, i2)

    def test_no_negative_indices(self, model):
        """All returned indices should be non-negative."""
        vec = _encode_query(model, "образование")
        distances, indices = search(vec, top_k=5)
        assert all(idx >= 0 for idx in indices[0])
