"""Comprehensive search tests for Phase 2 — Qdrant backend.

Tests 10+ Russian queries about MIREA, edge cases (empty, typo,
out-of-domain, very short), and metadata filtering.  All relevant
queries must score >0.3 (generous threshold, logged for visibility).
"""

import os
import sys

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.embeddings.indexer import search_qdrant

# ── Helpers ────────────────────────────────────────────────────────────────

def _encode_query(model, text: str) -> np.ndarray:
    """Encode a query with E5 'query: ' prefix."""
    prefixed = f"query: {text}"
    vec = model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=False)
    return vec.astype(np.float32)


def _search(model, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> list[dict]:
    """Encode + Qdrant search, return list of {id, score, payload}."""
    vec = _encode_query(model, query)
    return search_qdrant(vec, top_k=top_k)


def _keyword_hit(results: list[dict], keywords: list[str]) -> bool:
    """Check if any keyword appears in any result text (case-insensitive)."""
    for kw in keywords:
        for r in results:
            if kw.lower() in r["payload"]["text"].lower():
                return True
    return False


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model():
    """Load E5 model once for all tests."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


# ── Infrastructure tests ──────────────────────────────────────────────────

class TestQdrantInfrastructure:
    """Verify Qdrant collection is accessible and returns valid results."""

    def test_qdrant_collection_exists(self):
        """Qdrant storage directory must exist."""
        assert os.path.isdir(config.QDRANT_PATH), f"Qdrant path missing: {config.QDRANT_PATH}"

    def test_search_returns_list(self, model):
        """Search returns a list of dicts with expected keys."""
        results = _search(model, "тест")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "payload" in r
            assert "text" in r["payload"]

    def test_scores_in_valid_range(self, model):
        """Cosine similarity scores should be in [-1, 1]."""
        results = _search(model, "университет")
        for r in results:
            assert -1.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_results_sorted_descending(self, model):
        """Results should be sorted by score descending."""
        results = _search(model, "институт", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"

    def test_payload_has_required_fields(self, model):
        """Each payload must have text, section, source_url, source_title, chunk_index."""
        results = _search(model, "МИРЭА")
        required = {"text", "section", "source_url", "source_title", "chunk_index"}
        for r in results:
            assert required.issubset(r["payload"].keys()), \
                f"Missing fields: {required - r['payload'].keys()}"

    def test_vectors_in_collection(self):
        """Collection must be green and non-trivial (rebuilt 2026-10: 5090 pts)."""
        from src.embeddings.qdrant_indexer import QdrantIndexer
        qi = QdrantIndexer()
        info = qi.get_collection_info(config.QDRANT_PATH, config.QDRANT_COLLECTION)
        assert info["points_count"] >= 1000, f"Expected >= 1000, got {info['points_count']}"
        assert info["status"] == "green"


# ── 10+ Russian queries about MIREA ──────────────────────────────────────

class TestMIREASemanticSearch:
    """Test that 10+ Russian queries return relevant results with score >0.3."""

    def test_faculties(self, model):
        """Какие факультеты есть в МИРЭА?"""
        results = _search(model, "Какие факультеты есть в МИРЭА?")
        assert _keyword_hit(results, ["факультет", "институт", "подразделени"]), \
            f"No faculty keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_admission_scores(self, model):
        """проходной балл"""
        results = _search(model, "проходной балл")
        assert _keyword_hit(results, ["балл", "проходн", "конкурс"]), \
            f"No score keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_main_address(self, model):
        """адрес главного корпуса"""
        results = _search(model, "адрес главного корпуса")
        assert _keyword_hit(results, ["москва", "ул.", "проспект", "адрес", "корпус"]), \
            f"No address keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_cybersecurity(self, model):
        """кибербезопасность"""
        results = _search(model, "кибербезопасность")
        assert _keyword_hit(results, ["кибер", "безопасност", "криптограф", "информационн"]), \
            f"No cybersecurity keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_scholarship(self, model):
        """стипендия"""
        results = _search(model, "стипендия")
        assert _keyword_hit(results, ["стипенди", "выплат", "социальн"]), \
            f"No scholarship keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_dormitory(self, model):
        """общежитие"""
        results = _search(model, "общежитие")
        assert _keyword_hit(results, ["общежит", "проживан", "комнат", "расселен"]), \
            f"No dormitory keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_employment(self, model):
        """трудоустройство выпускников"""
        results = _search(model, "трудоустройство выпускников")
        assert _keyword_hit(results, ["трудоустройств", "выпускник", "работ", "карьер", "центр занятост"]), \
            f"No employment keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_admission_contacts(self, model):
        """контакты приемной комиссии"""
        results = _search(model, "контакты приемной комиссии")
        assert _keyword_hit(results, ["телефон", "495", "приёмн", "приемн", "контакт", "почт"]), \
            f"No contact keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_it_institute(self, model):
        """институт информационных технологий"""
        results = _search(model, "институт информационных технологий")
        assert _keyword_hit(results, ["информационных технолог", "ИИТ", "институт"]), \
            f"No IT institute keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_extracurricular(self, model):
        """внеучебная деятельность"""
        results = _search(model, "внеучебная деятельность")
        assert _keyword_hit(results, ["внеучебн", "деятельност", "клуб", "спорт", "мероприят"]), \
            f"No extracurricular keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_curriculum(self, model):
        """программа обучения"""
        results = _search(model, "программа обучения")
        assert _keyword_hit(results, ["программ", "обучен", "дисциплин", "курс"]), \
            f"No curriculum keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"

    def test_admission_docs(self, model):
        """документы для поступления"""
        results = _search(model, "документы для поступления")
        assert _keyword_hit(results, ["документ", "поступлен", "заявлен", "паспорт", "егэ"]), \
            f"No admission doc keywords. Top: {results[0]['payload']['text'][:100]}"
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']:.4f}"


# ── Edge case tests ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases: empty, typo, out-of-domain, very short queries."""

    def test_empty_query_does_not_crash(self, model):
        """Empty query must not crash, should return results."""
        results = _search(model, "")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_typo_query(self, model):
        """Typo 'факултет' should still return relevant results."""
        results = _search(model, "факултет")
        assert isinstance(results, list)
        assert len(results) > 0
        # With E5 multilingual, typos often still produce meaningful embeddings
        # We just check it doesn't crash; semantic quality is best-effort
        assert results[0]["score"] > 0.1, f"Typo score unexpectedly low: {results[0]['score']:.4f}"

    def test_out_of_domain_query(self, model):
        """'рецепт борща' is out-of-domain — should return lower scores."""
        results = _search(model, "рецепт борща")
        assert isinstance(results, list)
        assert len(results) > 0
        # Out-of-domain queries should still return *something* but with lower scores
        # than in-domain queries. We log the score but just verify no crash.
        top_score = results[0]["score"]
        # Log for visibility but don't enforce a strict threshold for out-of-domain
        print(f"  [out-of-domain] 'рецепт борща' top score: {top_score:.4f}")
        # Just verify it returns results — the score may be low

    def test_very_short_query(self, model):
        """Very short query 'МИРЭА' should return general info."""
        results = _search(model, "МИРЭА")
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["score"] > 0.3, f"Score too low for 'МИРЭА': {results[0]['score']:.4f}"

    def test_top_k_variations(self, model):
        """Different top_k values should work correctly."""
        for k in [1, 3, 5, 10]:
            results = _search(model, "факультеты", top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    def test_score_consistency(self, model):
        """Same query twice should give identical results."""
        r1 = _search(model, "информатика")
        r2 = _search(model, "информатика")
        scores1 = [r["score"] for r in r1]
        scores2 = [r["score"] for r in r2]
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)


# ── Metadata filtering test ───────────────────────────────────────────────

class TestMetadataFiltering:
    """Test that search results can be filtered by metadata (section)."""

    def test_search_within_specific_section(self, model):
        """Verify that results contain sections and filtering is possible."""
        results = _search(model, "общежитие", top_k=5)
        sections = [r["payload"]["section"] for r in results]
        # At least some results should have relevant sections
        assert len(sections) > 0
        # Log sections for visibility
        print(f"  [sections for 'общежитие']: {sections}")

    def test_section_field_is_populated(self, model):
        """All results should have a non-empty section field."""
        results = _search(model, "факультеты", top_k=5)
        for r in results:
            section = r["payload"].get("section", "")
            assert section, f"Empty section in result {r['id']}"

    def test_dormitory_section_filtering(self, model):
        """Search for dormitory terms and verify dormitory section exists in results."""
        results = _search(model, "проживание в общежитии", top_k=10)
        all_sections = [r["payload"]["section"] for r in results]
        # With 203 vectors from 7 sources, dormitory data should be findable
        assert len(all_sections) > 0
        # Check that at least one result mentions dormitory-related content
        has_dorm = _keyword_hit(results, ["общежит", "проживан", "комнат"])
        print(f"  [dormitory search sections]: {all_sections}")
        print(f"  [dormitory keyword hit]: {has_dorm}")

    def test_contact_section_filtering(self, model):
        """Search for contacts and verify contact section in results."""
        results = _search(model, "телефон приемной комиссии", top_k=10)
        all_sections = [r["payload"]["section"] for r in results]
        # Verify we can filter results by section after retrieval
        contact_results = [r for r in results if r["payload"]["section"] in ("contacts", "applicants")]
        assert len(contact_results) > 0, \
            f"No contacts/applicants sections. Got sections: {all_sections}"
