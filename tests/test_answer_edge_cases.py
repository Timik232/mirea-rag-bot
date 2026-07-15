"""Edge-case and robustness tests for the RAG answer layer (src.rag.answer).

These tests run WITHOUT any external service: no LM Studio, no Qdrant, no
embedding model. Every dependency is mocked, so the suite is fast and works on
a bare CI machine. They cover the three robustness concerns added in this task:

  1. input validation (_validate_query)        — type / length / control chars
  2. generation hardening (_generate)           — malformed, timeout, 5xx, retry
  3. graceful degradation in answer_question    — every failure stage -> fallback
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest
import requests

# ── Path setup so imports work from project root ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.answer import (  # noqa: E402
    GenerationError,
    LLM_FAILURE_ANSWER,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LETTERS,
    NO_CONTEXT_ANSWER,
    PIPELINE_FAILURE_ANSWER,
    _extract_answer,
    _format_answer,
    _generate,
    _normalize_query,
    _validate_query,
    answer_question,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. _validate_query
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateQuery:
    """Structural input validation: type, length, control chars, meaning."""

    @pytest.mark.parametrize("bad", [None, "", "   ", "\t\n  "])
    def test_empty_or_blank_rejected(self, bad):
        clean, error = _validate_query(bad)
        assert clean is None
        assert error  # non-empty reason

    def test_none_rejected(self):
        _, error = _validate_query(None)
        assert error == "Пожалуйста, задайте вопрос."

    def test_int_coerced(self):
        # Numbers are coerced to str — "2024" has >= 2 word-chars, passes.
        clean, error = _validate_query(2024)
        assert error is None
        assert clean == "2024"

    def test_float_coerced(self):
        clean, error = _validate_query(3.14)
        assert error is None
        assert clean == "3.14"

    @pytest.mark.parametrize("bad", [[1, 2], {"a": 1}, b"bytes", object()])
    def test_non_string_containers_rejected(self, bad):
        clean, error = _validate_query(bad)
        assert clean is None
        assert error  # rejected, not stringified

    def test_control_chars_stripped(self):
        # NUL / BEL / other C0 controls removed; \n collapsed to space.
        clean, error = _validate_query("Где\nкорпус\x00?")
        assert error is None
        assert "\x00" not in clean
        assert "Где" in clean

    def test_whitespace_collapsed(self):
        clean, error = _validate_query("  Где   находится   \n  корпус  ")
        assert error is None
        # internal whitespace collapsed to single spaces, stripped ends
        assert clean == "Где находится корпус"

    def test_long_query_truncated_not_rejected(self):
        long_query = "факультет " * (MAX_QUERY_LENGTH + 50)
        clean, error = _validate_query(long_query)
        assert error is None
        assert len(clean) <= MAX_QUERY_LENGTH

    def test_pure_punctuation_rejected(self):
        clean, error = _validate_query("???!!!....")
        assert clean is None
        assert "коротк" in error.lower() or "уточните" in error.lower()

    def test_pure_digits_rejected(self):
        # Single token "12" has 2 word-chars == MIN_QUERY_LETTERS — borderline.
        # Use a 1-char digit string to be safely below the threshold.
        clean, error = _validate_query("1")
        assert clean is None

    def test_normal_query_passes(self):
        q = "Где находится главный корпус МИРЭА?"
        clean, error = _validate_query(q)
        assert error is None
        assert clean == q

    def test_returns_tuple(self):
        result = _validate_query("тест")
        assert isinstance(result, tuple) and len(result) == 2


class TestNormalizeQuery:
    """Clean Cyrillic text is a no-op — the backward-compat contract.

    _normalize_query now applies whitespace/punctuation/layout normalisation
    (task t_bd93c958), but clean single-space, single-mark, Cyrillic queries
    pass through unchanged. See test_query_normalization.py for full coverage.
    """

    def test_passthrough_identity(self):
        assert _normalize_query("Где корпус?") == "Где корпус?"

    def test_returns_string(self):
        assert isinstance(_normalize_query("тест"), str)


# ═══════════════════════════════════════════════════════════════════════════
# 2. _extract_answer
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractAnswer:
    """Parsing of the OpenAI-style chat completion response."""

    def test_valid_response(self):
        data = {"choices": [{"message": {"content": "Главный корпус на Стромынке."}}]}
        assert _extract_answer(data) == "Главный корпус на Стромынке."

    def test_strips_whitespace(self):
        data = {"choices": [{"message": {"content": "  ответ  "}}]}
        assert _extract_answer(data) == "ответ"

    def test_missing_choices_raises(self):
        with pytest.raises(GenerationError):
            _extract_answer({"error": "something"})

    def test_empty_choices_raises(self):
        with pytest.raises(GenerationError):
            _extract_answer({"choices": []})

    def test_missing_message_raises(self):
        with pytest.raises(GenerationError):
            _extract_answer({"choices": [{"finish_reason": "stop"}]})

    def test_empty_content_raises(self):
        with pytest.raises(GenerationError):
            _extract_answer({"choices": [{"message": {"content": "   "}}]})

    def test_non_string_content_raises(self):
        with pytest.raises(GenerationError):
            _extract_answer({"choices": [{"message": {"content": None}}]})

    def test_null_data_raises(self):
        with pytest.raises(GenerationError):
            _extract_answer(None)


# ═══════════════════════════════════════════════════════════════════════════
# 2b. _format_answer — post-processing / presentation normalisation
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatAnswer:
    """Pure, idempotent presentation normalisation of generated answers."""

    def test_clean_prose_unchanged(self):
        # Already-clean text (no em-dash, no fence, no label) is returned as-is.
        s = "Главный корпус находится на ул. Стромынка, 20."
        assert _format_answer(s) == s

    def test_idempotent(self):
        # f(f(x)) == f(x) for any input — second pass changes nothing.
        for s in [
            "Чистый ответ.",
            "```ru\nв коде\n```",
            "Ответ: foo — bar — baz",
            "а\n\n\n\nб",
        ]:
            once = _format_answer(s)
            assert _format_answer(once) == once

    def test_empty_and_non_string_passthrough(self):
        assert _format_answer("") == ""
        assert _format_answer(None) is None
        assert _format_answer("   ") == ""

    def test_em_dash_normalised_to_en_dash(self):
        # Project rule: U+2014 (—) is never used; always U+2013 (–).
        s = "Бюджет — 150 мест — очная форма."
        out = _format_answer(s)
        assert "\u2014" not in out
        assert out.count("\u2013") == 2

    def test_hyphen_in_compound_word_preserved(self):
        # Hyphen-minus in compound words / numbers is NOT a dash to rewrite.
        s = "Wi-Fi, 09.03.04, 220-260"
        assert _format_answer(s) == s

    def test_strips_leading_label(self):
        assert _format_answer("Ответ: Корпус на Стромынке.") == "Корпус на Стромынке."
        assert _format_answer("Бот: Здравствуйте!") == "Здравствуйте!"

    def test_label_not_stripped_mid_sentence(self):
        # "Ответ:" appearing inside the body must be preserved.
        s = "Контакт: приёмная комиссия. Ответ: по телефону."
        assert "Ответ:" in _format_answer(s)

    def test_unwrap_code_fence(self):
        s = "```\nГлавный корпус на Стромынке.\n```"
        assert _format_answer(s) == "Главный корпус на Стромынке."

    def test_unwrap_code_fence_with_lang(self):
        s = "```ru\nТекст ответа.\n```"
        assert _format_answer(s) == "Текст ответа."

    def test_inline_backticks_preserved(self):
        # Only a fence wrapping the WHOLE answer is stripped; inline `code` stays.
        s = "Используй команду `help` для справки."
        assert _format_answer(s) == s

    def test_collapse_blank_runs(self):
        assert _format_answer("а\n\n\n\nб") == "а\n\nб"
        # A single blank line between paragraphs is preserved.
        assert _format_answer("а\n\nб") == "а\n\nб"

    def test_trailing_whitespace_trimmed(self):
        assert _format_answer("строка1   \nстрока2\t\n") == "строка1\nстрока2"

    def test_content_not_altered(self):
        # Facts / numbers survive every transform.
        s = "Проходной балл — 260 баллов (2024 год), код 09.03.04."
        out = _format_answer(s)
        assert "260" in out and "2024" in out and "09.03.04" in out


# ═══════════════════════════════════════════════════════════════════════════
# 3. _generate — HTTP robustness
# ═══════════════════════════════════════════════════════════════════════════

def _ok_response(text="Ответ сгенерирован."):
    """Build a fake 2xx requests.Response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"total_tokens": 42},
    }
    return resp


def _http_response(status_code, body="error"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = body
    return resp


@pytest.fixture
def no_sleep(monkeypatch):
    """Disable backoff sleeps so retry tests run instantly."""
    monkeypatch.setattr("src.rag.answer.time.sleep", lambda _s: None)


@pytest.fixture
def mock_post():
    """A MagicMock standing in for requests.post; configure .side_effect per test."""
    post = MagicMock()
    # Patch the real requests module's post — _generate does `import requests`
    # inside the function and resolves `requests.post` at call time.
    requests.post = post
    return post


class TestGenerateHttp:
    """_generate must classify HTTP outcomes and retry transient failures."""

    def test_success(self, mock_post, no_sleep):
        mock_post.return_value = _ok_response("Корпус на ул. Стромынка.")
        answer, usage = _generate([{"role": "user", "content": "q"}])
        assert answer == "Корпус на ул. Стромынка."
        assert usage == {"total_tokens": 42}
        assert mock_post.call_count == 1

    def test_connection_error_raises_immediately(self, mock_post, no_sleep):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(ConnectionError):
            _generate([{"role": "user", "content": "q"}])
        # No retry on connection error.
        assert mock_post.call_count == 1

    def test_timeout_then_success(self, mock_post, no_sleep):
        mock_post.side_effect = [
            requests.exceptions.Timeout("slow"),
            _ok_response("ok"),
        ]
        answer, _ = _generate([{"role": "user", "content": "q"}])
        assert answer == "ok"
        assert mock_post.call_count == 2

    def test_5xx_then_success(self, mock_post, no_sleep):
        mock_post.side_effect = [
            _http_response(503, "upstream down"),
            _ok_response("recovered"),
        ]
        answer, _ = _generate([{"role": "user", "content": "q"}])
        assert answer == "recovered"
        assert mock_post.call_count == 2

    def test_malformed_json_then_success(self, mock_post, no_sleep):
        bad = MagicMock()
        bad.status_code = 200
        bad.json.side_effect = ValueError("not json")
        mock_post.side_effect = [bad, _ok_response("ok")]
        answer, _ = _generate([{"role": "user", "content": "q"}])
        assert answer == "ok"
        assert mock_post.call_count == 2

    def test_empty_answer_then_success(self, mock_post, no_sleep):
        empty_body = {
            "choices": [{"message": {"content": "   "}}],
            "usage": {},
        }
        empty = MagicMock()
        empty.status_code = 200
        empty.json.return_value = empty_body
        mock_post.side_effect = [empty, _ok_response("real answer")]
        answer, _ = _generate([{"role": "user", "content": "q"}])
        assert answer == "real answer"
        assert mock_post.call_count == 2

    def test_4xx_raises_no_retry(self, mock_post, no_sleep):
        mock_post.return_value = _http_response(400, "bad request")
        with pytest.raises(GenerationError):
            _generate([{"role": "user", "content": "q"}])
        # 4xx is not retried.
        assert mock_post.call_count == 1

    def test_all_attempts_fail_raises_generation_error(self, mock_post, no_sleep):
        mock_post.return_value = _http_response(500, "down")
        with pytest.raises(GenerationError):
            _generate([{"role": "user", "content": "q"}])
        # 1 initial + GENERATION_MAX_RETRIES retries.
        from src.rag.answer import GENERATION_MAX_RETRIES
        assert mock_post.call_count == GENERATION_MAX_RETRIES + 1

    def test_request_exception_retried(self, mock_post, no_sleep):
        # Generic RequestException (e.g. ChunkedEncodingError) is retried.
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("broken"),
            _ok_response("ok"),
        ]
        answer, _ = _generate([{"role": "user", "content": "q"}])
        assert answer == "ok"
        assert mock_post.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. answer_question — graceful degradation end-to-end
# ═══════════════════════════════════════════════════════════════════════════

def _chunk(score=0.9, text="Контекст про главный корпус.", url="https://mirea.ru/about"):
    return {
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


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Patch the three pipeline stages so answer_question runs in isolation.

    Returns a dict of the mock objects so each test can configure behaviour.
    """
    encode = MagicMock(return_value=[0.0] * 384)
    retrieve = MagicMock(return_value=[_chunk()])
    generate = MagicMock(return_value=("Сгенерированный ответ.", {"total_tokens": 5}))
    monkeypatch.setattr("src.rag.answer._encode_query", encode)
    monkeypatch.setattr("src.rag.answer._retrieve", retrieve)
    monkeypatch.setattr("src.rag.answer._generate", generate)
    return {"encode": encode, "retrieve": retrieve, "generate": generate}


class TestAnswerQuestionDegradation:
    """Every failure stage must produce a fallback dict, never an exception."""

    # ── input validation short-circuits the pipeline ──────────────────────

    def test_none_query_returns_prompt(self, stub_pipeline):
        result = answer_question(None)
        assert result["answer"] == "Пожалуйста, задайте вопрос."
        assert result["sources"] == []
        # pipeline stages never called for invalid input
        stub_pipeline["encode"].assert_not_called()

    def test_empty_query_returns_prompt(self, stub_pipeline):
        result = answer_question("   ")
        assert "задайте вопрос" in result["answer"]
        stub_pipeline["encode"].assert_not_called()

    def test_punctuation_only_returns_prompt(self, stub_pipeline):
        result = answer_question("???!!!")
        assert result["sources"] == []
        stub_pipeline["encode"].assert_not_called()

    def test_list_query_rejected(self, stub_pipeline):
        result = answer_question(["не", "вопрос"])
        assert result["sources"] == []
        stub_pipeline["encode"].assert_not_called()

    def test_control_chars_handled(self, stub_pipeline):
        # Must not crash; control chars stripped, query proceeds.
        result = answer_question("Где\x00корпус?")
        assert result["answer"] == "Сгенерированный ответ."
        # The query passed downstream is cleaned.
        called_query = stub_pipeline["encode"].call_args[0][0]
        assert "\x00" not in called_query

    # ── encoding failure ──────────────────────────────────────────────────

    def test_encoding_failure_degrades(self, stub_pipeline):
        stub_pipeline["encode"].side_effect = RuntimeError("model load failed")
        result = answer_question("Где корпус?")
        assert result["answer"] == PIPELINE_FAILURE_ANSWER
        assert result["sources"] == []
        assert result["context_chunks"] == 0

    # ── retrieval failure / empty ─────────────────────────────────────────

    def test_no_chunks_returns_no_context(self, stub_pipeline):
        stub_pipeline["retrieve"].return_value = []
        result = answer_question("рецепт борща")
        assert result["answer"] == NO_CONTEXT_ANSWER
        assert result["sources"] == []

    # ── generation failure ────────────────────────────────────────────────

    def test_generation_error_keeps_sources(self, stub_pipeline):
        stub_pipeline["generate"].side_effect = GenerationError("flaky")
        result = answer_question("Где корпус?")
        assert result["answer"] == LLM_FAILURE_ANSWER
        # sources preserved even though generation failed
        assert len(result["sources"]) == 1
        assert result["context_chunks"] == 1
        assert result["retrieval_scores"]

    def test_connection_error_propagates(self, stub_pipeline):
        # LM Studio fully offline -> ConnectionError surfaces for the API 503.
        stub_pipeline["generate"].side_effect = ConnectionError("offline")
        with pytest.raises(ConnectionError):
            answer_question("Где корпус?")

    # ── happy path ────────────────────────────────────────────────────────

    def test_happy_path(self, stub_pipeline):
        result = answer_question("Где находится главный корпус МИРЭА?")
        assert result["answer"] == "Сгенерированный ответ."
        assert len(result["sources"]) == 1
        assert result["context_chunks"] == 1
        assert result["usage"] == {"total_tokens": 5}
        assert result["generation_time"] >= 0

    def test_result_has_all_keys(self, stub_pipeline):
        result = answer_question("тест вопроса")
        required = {
            "query", "answer", "sources", "context_chunks",
            "retrieval_scores", "generation_time", "usage",
        }
        assert required.issubset(result.keys())

    def test_clean_query_returned(self, stub_pipeline):
        # The returned query is the cleaned/normalised one, not the raw input.
        result = answer_question("  Где   корпус?  ")
        assert result["query"] == "Где корпус?"
