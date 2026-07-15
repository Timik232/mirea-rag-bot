"""Integration tests for the FastAPI server (src.rag.api).

These tests exercise the HTTP layer — the ``/ask`` and ``/health`` endpoints —
in isolation from the real external dependencies. The E5 embedding model, the
Qdrant collection, and the LM Studio generation server are all mocked through
``unittest.mock``, so the suite runs fast and hermetically in CI without
network access, large model downloads, or a running Qdrant / LM Studio.

Strategy
--------
* ``TestClient`` (httpx-backed) drives the app in-process.
* ``answer_question`` is patched in ``src.rag.api`` so ``/ask`` tests assert
  HTTP-layer behaviour (validation, response shaping, status-code mapping)
  rather than the RAG pipeline, which has its own unit tests.
* ``/health`` patches ``QdrantIndexer`` and ``requests.get`` to simulate the
  reachable / unreachable backends.

Run:
    python -m pytest tests/test_api_integration.py -v
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup so `import config` / `from src.rag...` resolve from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import src.rag.api as api_module  # noqa: E402
from src.rag.api import app  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """A ``TestClient`` with the embedding-model loader stubbed.

    Using the context manager triggers the FastAPI startup event, which would
    otherwise download the E5 model on first call. We stub
    ``_get_embedding_model`` so startup completes instantly, then pin
    ``_model_ready = True`` so ``/health`` is deterministic regardless of
    whether Starlette ran lifespan for this client instance.
    """
    with patch.object(api_module, "_get_embedding_model", return_value=MagicMock()):
        with TestClient(app) as c:
            api_module._model_ready = True
            yield c
            api_module._model_ready = False


def _mock_answer_result() -> dict:
    """A realistic ``answer_question`` return value for happy-path tests."""
    return {
        "query": "Где находится МИРЭА?",
        "answer": "Главный корпус находится по адресу проспект Вернадского, 78.",
        "sources": [
            {
                "url": "https://mirea.ru/contacts",
                "section": "Контакты",
                "title": "Контакты",
                "score": 0.82,
            },
            {
                "url": "https://mirea.ru/about",
                "section": "Об университете",
                "title": "О МИРЭА",
                "score": 0.71,
            },
        ],
        "context_chunks": 3,
        "retrieval_scores": [0.82, 0.71, 0.55],
        "generation_time": 1.23,
        "usage": {"total_tokens": 150},
    }


# ── /health ───────────────────────────────────────────────────────────────


class TestHealth:
    """GET /health — service health check."""

    def test_health_ok_with_all_deps_up(self, client):
        """Qdrant green + LM Studio reachable → status 'ok'."""
        qdrant_info = {"status": "green", "points_count": 52}
        lm_response = MagicMock(status_code=200)

        with patch("src.embeddings.qdrant_indexer.QdrantIndexer") as MockQI, \
                patch("requests.get", return_value=lm_response):
            MockQI.return_value.get_collection_info.return_value = qdrant_info
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["qdrant"] == {"ok": True, "points": 52}
        assert body["lm_studio"]["ok"] is True
        assert body["lm_studio"]["url"] == "http://localhost:1234/v1"
        assert body["uptime_seconds"] >= 0

    def test_health_degraded_when_qdrant_and_lm_down(self, client):
        """Both backends failing → status 'degraded' but still HTTP 200."""
        with patch("src.embeddings.qdrant_indexer.QdrantIndexer") as MockQI, \
                patch("requests.get") as mock_get:
            MockQI.return_value.get_collection_info.side_effect = RuntimeError("no qdrant")
            mock_get.side_effect = RuntimeError("no lm studio")
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["qdrant"]["ok"] is False
        assert body["qdrant"]["points"] == 0
        assert body["lm_studio"]["ok"] is False

    def test_health_degraded_when_qdrant_not_green(self, client):
        """A non-green collection status is treated as unavailable."""
        with patch("src.embeddings.qdrant_indexer.QdrantIndexer") as MockQI, \
                patch("requests.get", return_value=MagicMock(status_code=200)):
            MockQI.return_value.get_collection_info.return_value = {
                "status": "yellow", "points_count": 10,
            }
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        # qdrant_ok requires status == "green"; yellow → not ok → degraded
        assert body["qdrant"]["ok"] is False
        assert body["status"] == "degraded"

    def test_health_response_keys(self, client):
        """The health payload always exposes the documented keys."""
        with patch("src.embeddings.qdrant_indexer.QdrantIndexer") as MockQI, \
                patch("requests.get") as mock_get:
            MockQI.return_value.get_collection_info.side_effect = RuntimeError()
            mock_get.side_effect = RuntimeError()
            resp = client.get("/health")

        body = resp.json()
        assert set(body) == {
            "status", "model_loaded", "qdrant", "lm_studio", "uptime_seconds",
        }
        assert set(body["qdrant"]) == {"ok", "points"}
        assert set(body["lm_studio"]) == {"ok", "url"}

    def test_health_never_500_on_backend_failure(self, client):
        """A crashing backend must not bubble up as a 500."""
        with patch("src.embeddings.qdrant_indexer.QdrantIndexer") as MockQI, \
                patch("requests.get") as mock_get:
            MockQI.side_effect = RuntimeError("ctor exploded")
            mock_get.side_effect = Exception("net gone")
            resp = client.get("/health")
        assert resp.status_code == 200


# ── /ask — success path and response shaping ─────────────────────────────


class TestAskSuccess:
    """POST /ask — happy path and ``AskResponse`` shaping."""

    def test_ask_returns_200_with_full_payload(self, client):
        with patch.object(api_module, "answer_question",
                          return_value=_mock_answer_result()):
            resp = client.post("/ask", json={"question": "Где находится МИРЭА?"})

        assert resp.status_code == 200
        body = resp.json()
        assert "Главный корпус" in body["answer"]
        assert body["context_chunks"] == 3
        assert body["retrieval_scores"] == [0.82, 0.71, 0.55]
        assert body["generation_time"] == pytest.approx(1.23)
        assert body["confidence"] == round(0.82, 4)
        assert len(body["sources"]) == 2
        assert body["sources"][0]["url"] == "https://mirea.ru/contacts"
        assert body["sources"][0]["score"] == 0.82

    def test_ask_confidence_equals_top_retrieval_score(self, client):
        result = _mock_answer_result()
        with patch.object(api_module, "answer_question", return_value=result):
            resp = client.post("/ask", json={"question": "вопрос"})
        assert resp.json()["confidence"] == round(result["retrieval_scores"][0], 4)

    def test_ask_handles_no_context_answer(self, client):
        """When retrieval finds nothing relevant, sources are empty."""
        result = {
            "answer": "У меня нет информации по этому вопросу.",
            "sources": [],
            "context_chunks": 0,
            "retrieval_scores": [],
            "generation_time": 0.42,
        }
        with patch.object(api_module, "answer_question", return_value=result):
            resp = client.post("/ask", json={"question": "случайный набор слов"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"] == []
        assert body["context_chunks"] == 0
        assert body["confidence"] == 0.0  # no scores → confidence 0

    def test_ask_passes_question_string_to_answer_fn(self, client):
        """The exact question text is forwarded to answer_question."""
        with patch.object(api_module, "answer_question",
                          return_value=_mock_answer_result()) as mock_fn:
            client.post("/ask", json={"question": "проходной балл"})
        mock_fn.assert_called_once_with("проходной балл")

    def test_ask_response_schema_matches_model(self, client):
        """Response keys match the declared ``AskResponse`` model exactly."""
        with patch.object(api_module, "answer_question",
                          return_value=_mock_answer_result()):
            resp = client.post("/ask", json={"question": "вопрос"})
        assert set(resp.json()) == {
            "answer", "sources", "context_chunks",
            "retrieval_scores", "generation_time", "confidence",
        }


# ── /ask — request validation ────────────────────────────────────────────


class TestAskValidation:
    """POST /ask — pydantic ``AskRequest`` validation."""

    def test_missing_question_field_rejected(self, client):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422

    def test_empty_question_rejected(self, client):
        """``min_length=1`` rejects the empty string."""
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_too_long_question_rejected(self, client):
        """``max_length=2000`` rejects an oversized payload."""
        resp = client.post("/ask", json={"question": "а" * 2001})
        assert resp.status_code == 422

    def test_max_length_boundary_accepted(self, client):
        """Exactly 2000 chars is the upper bound and must be accepted."""
        with patch.object(api_module, "answer_question",
                          return_value=_mock_answer_result()):
            resp = client.post("/ask", json={"question": "а" * 2000})
        assert resp.status_code == 200

    def test_non_string_question_rejected(self, client):
        """``question`` must be a string; a number fails validation."""
        resp = client.post("/ask", json={"question": 12345})
        assert resp.status_code == 422

    def test_malformed_json_body_rejected(self, client):
        """A body that is not valid JSON yields a client error."""
        resp = client.post(
            "/ask",
            content=b"not-json{",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    def test_unknown_fields_ignored(self, client):
        """Extra fields do not break the request (pydantic default)."""
        with patch.object(api_module, "answer_question",
                          return_value=_mock_answer_result()):
            resp = client.post("/ask", json={"question": "вопрос", "extra": 1})
        assert resp.status_code == 200


# ── /ask — error mapping ─────────────────────────────────────────────────


class TestAskErrors:
    """POST /ask — exception → HTTP status code mapping."""

    def test_connection_error_returns_503(self, client):
        """An unreachable LLM backend surfaces as 503 Service Unavailable."""
        with patch.object(api_module, "answer_question",
                          side_effect=ConnectionError("LM Studio недоступен")):
            resp = client.post("/ask", json={"question": "вопрос"})
        assert resp.status_code == 503
        assert "LM Studio недоступен" in resp.json()["detail"]

    def test_generic_error_returns_500(self, client):
        """Any unexpected exception maps to 500 with the message in detail."""
        with patch.object(api_module, "answer_question",
                          side_effect=RuntimeError("внутренний сбой")):
            resp = client.post("/ask", json={"question": "вопрос"})
        assert resp.status_code == 500
        assert "внутренний сбой" in resp.json()["detail"]

    def test_answer_question_called_once_per_request(self, client):
        with patch.object(api_module, "answer_question",
                          return_value=_mock_answer_result()) as mock_fn:
            client.post("/ask", json={"question": "вопрос"})
        assert mock_fn.call_count == 1


# ── API metadata ─────────────────────────────────────────────────────────


class TestApiMeta:
    """OpenAPI schema and documentation routes."""

    def test_openapi_schema_lists_endpoints(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "МИРЭА RAG Bot"
        assert "/ask" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_docs_route_serves_swagger(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"].lower()

    def test_ask_request_schema_documented(self, client):
        """The /ask request body enforces a non-empty question."""
        schema = client.get("/openapi.json").json()
        ask_body = schema["paths"]["/ask"]["post"]["requestBody"]
        ref_schema = ask_body["content"]["application/json"]["schema"]["$ref"]
        # Resolve the $ref (e.g. "#/components/schemas/AskRequest")
        name = ref_schema.split("/")[-1]
        assert name == "AskRequest"
        assert "AskRequest" in schema["components"]["schemas"]
