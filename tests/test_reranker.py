"""Unit tests for the cross-encoder re-ranking layer (task t_007083df).

These tests mock the cross-encoder model so they run without the 2.2 GB
``bge-reranker-v2-m3`` download — they verify the *plumbing* (ordering,
threshold passthrough, empty-text de-ranking, graceful failure), not model
quality (which is covered by ``scripts/eval_rerank.py``).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import config  # noqa: E402
from src.rag import reranker  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _FakeCrossEncoder:
    """Returns a caller-controlled score per (query, passage) pair.

    Scores are looked up by passage substring so tests can express "the chunk
    mentioning X is the relevant one" without hard-coding ids.
    """

    def __init__(self, score_map: dict[str, float], default: float = 0.0):
        self.score_map = score_map
        self.default = default
        self.calls = 0

    def predict(self, pairs, convert_to_numpy=True, batch_size=None):
        self.calls += 1
        out = []
        for _q, passage in pairs:
            score = self.default
            low = passage.lower()
            for needle, val in self.score_map.items():
                if needle.lower() in low:
                    score = val
                    break
            out.append([score])
        return np.array(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_is_enabled_defaults_off(monkeypatch):
    """Re-ranking must be off by default so existing behaviour is preserved."""
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    assert reranker.is_enabled() is False


def test_is_enabled_reflects_flag(monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    assert reranker.is_enabled() is True


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    """The cross-encoder's ranking must override the bi-encoder order."""
    fake = _FakeCrossEncoder({"проходной": 0.99, "расписани": 0.10, "спорт": 0.05})
    monkeypatch.setattr(reranker, "_get_reranker_model", lambda: fake)

    chunks = [
        {"id": 3, "score": 0.95, "payload": {"text": "Студенческий спортзал."}},
        {"id": 2, "score": 0.80, "payload": {"text": "Расписание занятий."}},
        {"id": 1, "score": 0.70, "payload": {"text": "Проходной балл 260."}},
    ]
    out = reranker.rerank_chunks("проходной балл", chunks)

    ids = [c["id"] for c in out]
    assert ids == [1, 2, 3], f"expected CE order [1,2,3], got {ids}"
    # original bi-encoder score preserved for display
    assert out[0]["score"] == 0.70
    # rerank_score annotated
    assert out[0]["rerank_score"] == pytest.approx(0.99)


def test_rerank_respects_top_k(monkeypatch):
    fake = _FakeCrossEncoder({"a": 0.9, "b": 0.5, "c": 0.1})
    monkeypatch.setattr(reranker, "_get_reranker_model", lambda: fake)
    chunks = [
        {"id": i, "score": 0.5, "payload": {"text": t}}
        for i, t in enumerate(["c text", "b text", "a text"])
    ]
    out = reranker.rerank_chunks("q", chunks, top_k=2)
    assert len(out) == 2
    assert [c["id"] for c in out] == [2, 1]


def test_rerank_uses_top_k_from_config_when_none(monkeypatch):
    fake = _FakeCrossEncoder({"a": 0.9, "b": 0.5})
    monkeypatch.setattr(reranker, "_get_reranker_model", lambda: fake)
    monkeypatch.setattr(config, "RERANK_TOP_K", 1)
    chunks = [
        {"id": 1, "score": 0.5, "payload": {"text": "b"}},
        {"id": 2, "score": 0.5, "payload": {"text": "a"}},
    ]
    out = reranker.rerank_chunks("q", chunks, top_k=None)
    assert len(out) == 1
    assert out[0]["id"] == 2


def test_rerank_empty_text_de_ranked_to_bottom(monkeypatch):
    """A chunk with no passage text must not win on a spurious CE score."""
    fake = _FakeCrossEncoder({}, default=0.99)  # everything scores high
    monkeypatch.setattr(reranker, "_get_reranker_model", lambda: fake)
    chunks = [
        {"id": 1, "score": 0.9, "payload": {"text": ""}},   # empty
        {"id": 2, "score": 0.5, "payload": {"text": "real passage"}},
    ]
    out = reranker.rerank_chunks("q", chunks)
    assert out[-1]["id"] == 1                      # empty sinks to bottom
    assert out[-1]["rerank_score"] is None         # and has no CE score


def test_rerank_empty_input_returns_empty():
    assert reranker.rerank_chunks("q", []) == []


def test_rerank_graceful_failure_returns_input_order(monkeypatch):
    """If the cross-encoder blows up, retrieval must still return something."""
    def _boom():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(reranker, "_get_reranker_model", _boom)
    chunks = [
        {"id": i, "score": s, "payload": {"text": f"text {i}"}}
        for i, s in enumerate([0.9, 0.8, 0.7])
    ]
    out = reranker.rerank_chunks("q", chunks, top_k=2)
    # bi-encoder order preserved (input order), top_k honoured
    assert [c["id"] for c in out] == [0, 1]


def test_rerank_accepts_top_level_text_field(monkeypatch):
    """Eval/numpy path carries text alongside, not nested in payload."""
    fake = _FakeCrossEncoder({"good": 0.9}, default=0.1)
    monkeypatch.setattr(reranker, "_get_reranker_model", lambda: fake)
    chunks = [
        {"id": 1, "score": 0.5, "text": "noise"},
        {"id": 2, "score": 0.5, "text": "this is good"},
    ]
    out = reranker.rerank_chunks("q", chunks)
    assert out[0]["id"] == 2
