"""Tests for the hallucination guard (confidence threshold) — task t_737c1b5e.

Covers:
  1. _check_confidence() — bi-encoder and cross-encoder signal paths
  2. answer_question() integration — guard blocks generation on low confidence
  3. Result dict fields — confidence_score, confidence_signal, confidence_guarded
  4. Source surfacing — blocked queries still return sources for the user
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.answer import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    CE_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_ANSWER,
    NO_CONTEXT_ANSWER,
    _check_confidence,
    answer_question,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _chunk(score=0.9, rerank_score=None, text="Контекст про главный корпус.",
           url="https://mirea.ru/about"):
    c = {
        "id": 1,
        "score": score,
        "payload": {
            "text": text,
            "section": "about",
            "source_url": url,
            "source_title": "О МГРИ-МРТИ",
            "chunk_index": 0,
        },
    }
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    return c


# ═══════════════════════════════════════════════════════════════════════════
# 1. _check_confidence — unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckConfidenceBi:
    """Bi-encoder signal path (reranker disabled — production default)."""

    def test_high_score_passes(self):
        chunks = [_chunk(score=0.90)]
        passes, score, signal = _check_confidence(chunks)
        assert passes is True
        assert signal == "bi"
        assert abs(score - 0.90) < 1e-6

    def test_low_score_blocked(self):
        chunks = [_chunk(score=0.40)]
        passes, score, signal = _check_confidence(chunks)
        assert passes is False
        assert signal == "bi"
        assert abs(score - 0.40) < 1e-6

    def test_exact_threshold_passes(self):
        """Score == threshold should pass (>= comparison)."""
        chunks = [_chunk(score=CONFIDENCE_THRESHOLD)]
        passes, _, _ = _check_confidence(chunks)
        assert passes is True

    def test_just_below_threshold_blocked(self):
        chunks = [_chunk(score=CONFIDENCE_THRESHOLD - 0.001)]
        passes, _, _ = _check_confidence(chunks)
        assert passes is False

    def test_uses_first_chunk_score(self):
        """Guard evaluates the top-1 chunk, not the average."""
        chunks = [
            _chunk(score=0.90),  # top-1, high
            _chunk(score=0.30),  # low, but shouldn't matter
        ]
        passes, score, _ = _check_confidence(chunks)
        assert passes is True
        assert abs(score - 0.90) < 1e-6

    def test_missing_score_defaults_to_zero(self):
        chunks = [{"payload": {"text": "no score field"}}]
        passes, score, _ = _check_confidence(chunks)
        assert passes is False
        assert score == 0.0


class TestCheckConfidenceCe:
    """Cross-encoder signal path (reranker enabled)."""

    @pytest.fixture
    def ce_enabled(self, monkeypatch):
        """Patch rerank_is_enabled to return True and provide CE scores."""
        monkeypatch.setattr("src.rag.answer.rerank_is_enabled", lambda: True)

    def test_high_ce_passes(self, ce_enabled):
        chunks = [_chunk(score=0.85, rerank_score=0.99)]
        passes, score, signal = _check_confidence(chunks)
        assert passes is True
        assert signal == "ce"
        assert abs(score - 0.99) < 1e-6

    def test_low_ce_blocked(self, ce_enabled):
        chunks = [_chunk(score=0.90, rerank_score=0.15)]
        passes, score, signal = _check_confidence(chunks)
        assert passes is False
        assert signal == "ce"
        assert abs(score - 0.15) < 1e-6

    def test_ce_threshold_edge(self, ce_enabled):
        chunks = [_chunk(rerank_score=CE_CONFIDENCE_THRESHOLD)]
        passes, _, signal = _check_confidence(chunks)
        assert passes is True
        assert signal == "ce"

    def test_ce_none_falls_back_to_bi(self, ce_enabled):
        """If rerank_score is None (empty text chunk), falls back to bi score."""
        chunks = [_chunk(score=0.90, rerank_score=None)]
        passes, _, signal = _check_confidence(chunks)
        assert passes is True
        assert signal == "bi"

    def test_ce_disabled_uses_bi(self, monkeypatch):
        """When reranker is disabled, CE score is ignored even if present."""
        monkeypatch.setattr("src.rag.answer.rerank_is_enabled", lambda: False)
        chunks = [_chunk(score=0.90, rerank_score=0.01)]  # low CE, high bi
        passes, score, signal = _check_confidence(chunks)
        assert passes is True
        assert signal == "bi"
        assert abs(score - 0.90) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 2. answer_question — integration tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def stub_pipeline(monkeypatch):
    """Patch pipeline stages so answer_question runs in isolation."""
    encode = MagicMock(return_value=[0.0] * 384)
    retrieve = MagicMock(return_value=[_chunk()])
    generate = MagicMock(return_value=("Сгенерированный ответ.", {"total_tokens": 5}))
    monkeypatch.setattr("src.rag.answer._encode_query", encode)
    monkeypatch.setattr("src.rag.answer._retrieve", retrieve)
    monkeypatch.setattr("src.rag.answer._generate", generate)
    return {"encode": encode, "retrieve": retrieve, "generate": generate}


class TestAnswerQuestionGuard:
    """The guard must block generation and surface sources on low confidence."""

    def test_high_confidence_generates(self, stub_pipeline):
        """High-score retrieval → normal generation path."""
        stub_pipeline["retrieve"].return_value = [_chunk(score=0.90)]
        result = answer_question("Где корпус?")
        assert result["answer"] == "Сгенерированный ответ."
        assert result["confidence_guarded"] is False
        assert result["confidence_signal"] == "bi"
        assert abs(result["confidence_score"] - 0.90) < 1e-3
        stub_pipeline["generate"].assert_called_once()

    def test_low_confidence_blocks_generation(self, stub_pipeline):
        """Top-1 score below threshold → guard blocks, generation NOT called."""
        stub_pipeline["retrieve"].return_value = [_chunk(score=0.40)]
        result = answer_question("рецепт борща")
        assert result["answer"] == LOW_CONFIDENCE_ANSWER
        assert result["confidence_guarded"] is True
        assert result["confidence_signal"] == "bi"
        assert abs(result["confidence_score"] - 0.40) < 1e-3
        # Crucially, generation was never attempted.
        stub_pipeline["generate"].assert_not_called()

    def test_blocked_query_still_surfaces_sources(self, stub_pipeline):
        """Even when blocked, retrieved sources must be returned."""
        stub_pipeline["retrieve"].return_value = [
            _chunk(score=0.40, url="https://mirea.ru/sveden/"),
        ]
        result = answer_question("непонятный вопрос")
        assert result["confidence_guarded"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"] == "https://mirea.ru/sveden/"
        assert result["context_chunks"] == 1
        assert result["retrieval_scores"]  # non-empty

    def test_no_chunks_still_returns_no_context(self, stub_pipeline):
        """Zero chunks → NO_CONTEXT_ANSWER (existing behaviour, not guard)."""
        stub_pipeline["retrieve"].return_value = []
        result = answer_question("рецепт борща")
        assert result["answer"] == NO_CONTEXT_ANSWER
        assert result["confidence_guarded"] is False
        assert result["confidence_score"] == 0.0

    def test_result_has_confidence_keys(self, stub_pipeline):
        """All return paths must include the new confidence fields."""
        result = answer_question("тест")
        for key in ("confidence_score", "confidence_signal", "confidence_guarded"):
            assert key in result, f"missing key: {key}"

    def test_empty_query_confidence_fields(self, stub_pipeline):
        """Validation short-circuit also includes confidence fields."""
        result = answer_question(None)
        assert result["confidence_guarded"] is False
        assert result["confidence_score"] == 0.0
        assert result["confidence_signal"] == ""

    def test_mixed_scores_uses_top1(self, stub_pipeline):
        """Guard checks only top-1, not average — a low top-2 must not block."""
        stub_pipeline["retrieve"].return_value = [
            _chunk(score=0.90),
            _chunk(score=0.36),  # above RELEVANCE_THRESHOLD (0.35) but low
        ]
        result = answer_question("тест")
        assert result["confidence_guarded"] is False
        assert result["answer"] == "Сгенерированный ответ."
