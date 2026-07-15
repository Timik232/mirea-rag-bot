"""Unit tests for the RAG answer layer (src/rag/answer.py).

Covers every function in the module:
  - _build_context    (pure)
  - _build_messages   (pure)
  - _format_sources   (pure)
  - _retrieve         (Qdrant mocked)
  - _encode_query     (SentenceTransformer mocked)
  - _generate         (LM Studio / requests.post mocked)
  - answer_question   (end-to-end, all deps mocked)

No network access, no model downloads, no running services required.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

# ── Path setup so imports work from project root ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.answer import (
    MAX_CONTEXT_CHUNKS,
    NO_CONTEXT_ANSWER,
    RELEVANCE_THRESHOLD,
    SYSTEM_PROMPT,
    _build_context,
    _build_messages,
    _format_sources,
    _generate,
    _retrieve,
    answer_question,
)
from src.rag import answer as answer_mod


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_chunk(
    text="Текст чанка про МИРЭА.",
    url="https://mirea.ru/page",
    section="Сведения",
    score=0.90,
    title="Заголовок страницы",
    chunk_index=0,
):
    """Build a Qdrant-shaped chunk dict as returned by search_qdrant."""
    return {
        "id": chunk_index,
        "score": score,
        "payload": {
            "text": text,
            "source_url": url,
            "section": section,
            "source_title": title,
            "chunk_index": chunk_index,
        },
    }


def _mock_embedding_model(dim=384):
    """A fake SentenceTransformer whose .encode() returns a real ndarray."""
    model = MagicMock()
    model.encode.return_value = np.ones(dim, dtype=np.float32)
    return model


def _mock_lm_response(text="Ответ бота.", usage=None):
    """A fake requests.Response for LM Studio chat completions."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": usage if usage is not None else {"total_tokens": 42},
    }
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# 1. Module constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Module-level constants must exist and have sane values."""

    def test_relevance_threshold_in_range(self):
        assert 0.0 < RELEVANCE_THRESHOLD < 1.0

    def test_max_context_chunks_positive(self):
        assert MAX_CONTEXT_CHUNKS >= 1

    def test_system_prompt_is_russian(self):
        assert "русск" in SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_hallucination(self):
        # Prompt must instruct to use only context / not invent facts.
        assert "только" in SYSTEM_PROMPT.lower() or "не выдумывай" in SYSTEM_PROMPT.lower()

    def test_no_context_answer_nonempty(self):
        assert len(NO_CONTEXT_ANSWER.strip()) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. _build_context — pure function
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildContext:
    """_build_context assembles a numbered context block from chunks."""

    def test_empty_chunks(self):
        assert _build_context([]) == ""

    def test_single_chunk_contains_text(self):
        chunk = _make_chunk(text="Главный корпус на Стромынке.")
        ctx = _build_context([chunk])
        assert "Главный корпус на Стромынке." in ctx

    def test_single_chunk_contains_index(self):
        ctx = _build_context([_make_chunk()])
        assert "[1]" in ctx

    def test_score_formatted_as_percent(self):
        chunk = _make_chunk(score=0.85)
        ctx = _build_context([chunk])
        assert "85.0%" in ctx

    def test_section_and_url_appear(self):
        chunk = _make_chunk(section="Контакты", url="https://mirea.ru/contacts")
        ctx = _build_context([chunk])
        assert "Контакты" in ctx
        assert "https://mirea.ru/contacts" in ctx

    def test_multiple_chunks_separated_by_double_newline(self):
        chunks = [_make_chunk(text="Первый."), _make_chunk(text="Второй.")]
        ctx = _build_context(chunks)
        blocks = ctx.split("\n\n")
        assert len(blocks) == 2
        assert "Первый." in blocks[0]
        assert "Второй." in blocks[1]

    def test_chunks_numbered_sequentially(self):
        chunks = [_make_chunk() for _ in range(3)]
        ctx = _build_context(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx

    def test_chunk_missing_payload_does_not_crash(self):
        # A chunk with no payload key at all.
        ctx = _build_context([{"id": 0, "score": 0.5}])
        assert "[1]" in ctx
        assert "50.0%" in ctx

    def test_chunk_missing_score_defaults_to_zero(self):
        ctx = _build_context([{"payload": {"text": "без оценки"}}])
        assert "0.0%" in ctx


# ═══════════════════════════════════════════════════════════════════════════
# 3. _build_messages — pure function
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildMessages:
    """_build_messages produces the system+user message list for the LLM."""

    def test_returns_two_messages(self):
        msgs = _build_messages("вопрос", "контекст")
        assert len(msgs) == 2

    def test_first_is_system(self):
        msgs = _build_messages("q", "c")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_second_is_user(self):
        msgs = _build_messages("q", "c")
        assert msgs[1]["role"] == "user"

    def test_user_content_contains_query(self):
        msgs = _build_messages("Где приёмная комиссия?", "ctx")
        assert "Где приёмная комиссия?" in msgs[1]["content"]

    def test_user_content_contains_context(self):
        msgs = _build_messages("q", "Контекст про общежитие")
        assert "Контекст про общежитие" in msgs[1]["content"]

    def test_user_content_has_question_label(self):
        msgs = _build_messages("мой вопрос", "ctx")
        assert "Вопрос:" in msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. _format_sources — pure function
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatSources:
    """_format_sources extracts and de-duplicates source metadata."""

    def test_empty_chunks(self):
        assert _format_sources([]) == []

    def test_single_chunk_returns_one_source(self):
        sources = _format_sources([_make_chunk(url="https://a.ru")])
        assert len(sources) == 1
        assert sources[0]["url"] == "https://a.ru"

    def test_deduplicates_by_url(self):
        chunks = [
            _make_chunk(url="https://same.ru", score=0.9),
            _make_chunk(url="https://same.ru", score=0.8),
        ]
        assert len(_format_sources(chunks)) == 1

    def test_different_urls_kept(self):
        chunks = [
            _make_chunk(url="https://a.ru"),
            _make_chunk(url="https://b.ru"),
        ]
        assert len(_format_sources(chunks)) == 2

    def test_extracts_fields(self):
        chunk = _make_chunk(url="https://x.ru", section="Сведения", title="Title", score=0.75)
        src = _format_sources([chunk])[0]
        assert src["url"] == "https://x.ru"
        assert src["section"] == "Сведения"
        assert src["title"] == "Title"

    def test_score_rounded_to_4_decimals(self):
        chunk = _make_chunk(score=0.8765432)
        src = _format_sources([chunk])[0]
        assert src["score"] == round(0.8765432, 4)

    def test_skips_chunks_without_url(self):
        chunk = {"score": 0.9, "payload": {"text": "нет урла"}}
        assert _format_sources([chunk]) == []

    def test_preserves_first_occurrence_order(self):
        chunks = [
            _make_chunk(url="https://a.ru", score=0.9),
            _make_chunk(url="https://b.ru", score=0.7),
            _make_chunk(url="https://a.ru", score=0.6),
        ]
        urls = [s["url"] for s in _format_sources(chunks)]
        assert urls == ["https://a.ru", "https://b.ru"]

    def test_chunk_missing_payload_does_not_crash(self):
        # No payload key at all.
        _format_sources([{"score": 0.5}])


# ═══════════════════════════════════════════════════════════════════════════
# 5. _retrieve — search_qdrant mocked
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrieve:
    """_retrieve calls search_qdrant and filters by RELEVANCE_THRESHOLD."""

    @patch("src.embeddings.indexer.search_qdrant")
    def test_filters_below_threshold(self, mock_search):
        mock_search.return_value = [
            _make_chunk(score=0.90),
            _make_chunk(score=0.10),  # below 0.35
        ]
        result = _retrieve(np.ones(384, dtype=np.float32))
        assert len(result) == 1
        assert result[0]["score"] == 0.90

    @patch("src.embeddings.indexer.search_qdrant")
    def test_keeps_score_at_threshold_boundary(self, mock_search):
        mock_search.return_value = [_make_chunk(score=RELEVANCE_THRESHOLD)]
        result = _retrieve(np.ones(384, dtype=np.float32))
        assert len(result) == 1  # boundary is >= so kept

    @patch("src.embeddings.indexer.search_qdrant")
    def test_passes_top_k(self, mock_search):
        mock_search.return_value = []
        _retrieve(np.ones(384, dtype=np.float32), top_k=7)
        _, kwargs = mock_search.call_args
        assert kwargs.get("top_k") == 7

    @patch("src.embeddings.indexer.search_qdrant")
    def test_empty_results(self, mock_search):
        mock_search.return_value = []
        assert _retrieve(np.ones(384, dtype=np.float32)) == []

    @patch("src.embeddings.indexer.search_qdrant")
    def test_all_below_threshold_returns_empty(self, mock_search):
        mock_search.return_value = [_make_chunk(score=0.1), _make_chunk(score=0.2)]
        assert _retrieve(np.ones(384, dtype=np.float32)) == []

    @patch("src.embeddings.indexer.search_qdrant")
    def test_default_top_k_is_max_context_chunks(self, mock_search):
        mock_search.return_value = []
        _retrieve(np.ones(384, dtype=np.float32))
        _, kwargs = mock_search.call_args
        assert kwargs.get("top_k") == MAX_CONTEXT_CHUNKS


# ═══════════════════════════════════════════════════════════════════════════
# 6. _encode_query — embedding model mocked
# ═══════════════════════════════════════════════════════════════════════════

class TestEncodeQuery:
    """_encode_query prefixes the query and returns a float32 vector."""

    @patch.object(answer_mod, "_get_embedding_model")
    def test_adds_e5_query_prefix(self, mock_get_model):
        model = _mock_embedding_model()
        mock_get_model.return_value = model
        answer_mod._encode_query("адрес МИРЭА")
        model.encode.assert_called_once()
        call_args = model.encode.call_args
        sent_text = call_args[0][0] if call_args[0] else call_args[1].get("sentences", "")
        assert sent_text == "query: адрес МИРЭА"

    @patch.object(answer_mod, "_get_embedding_model")
    def test_returns_float32_ndarray(self, mock_get_model):
        mock_get_model.return_value = _mock_embedding_model()
        vec = answer_mod._encode_query("вопрос")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    @patch.object(answer_mod, "_get_embedding_model")
    def test_passes_convert_to_numpy_and_no_normalize(self, mock_get_model):
        model = _mock_embedding_model()
        mock_get_model.return_value = model
        answer_mod._encode_query("q")
        _, kwargs = model.encode.call_args
        assert kwargs.get("convert_to_numpy") is True
        assert kwargs.get("normalize_embeddings") is False


# ═══════════════════════════════════════════════════════════════════════════
# 7. _generate — requests.post (LM Studio) mocked
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerate:
    """_generate calls LM Studio and extracts the answer."""

    @patch("requests.post")
    def test_successful_generation_returns_answer_and_usage(self, mock_post):
        mock_post.return_value = _mock_lm_response(text="  Ответ  ", usage={"total_tokens": 5})
        answer, usage = _generate([{"role": "user", "content": "q"}])
        assert answer == "Ответ"  # stripped
        assert usage == {"total_tokens": 5}

    @patch("requests.post")
    def test_strips_whitespace(self, mock_post):
        mock_post.return_value = _mock_lm_response(text="\n  Текст с пробелами  \n")
        answer, _ = _generate([{"role": "user", "content": "q"}])
        assert answer == "Текст с пробелами"

    @patch("requests.post")
    def test_connection_error_raises_builtin_connectionerror(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(ConnectionError):
            _generate([{"role": "user", "content": "q"}])

    @patch("requests.post")
    def test_http_error_propagates(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = resp
        with pytest.raises(requests.exceptions.HTTPError):
            _generate([{"role": "user", "content": "q"}])

    @patch("requests.post")
    def test_payload_uses_config(self, mock_post):
        import config

        mock_post.return_value = _mock_lm_response()
        _generate([{"role": "user", "content": "q"}])
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == config.LM_STUDIO_CONFIG["model"]
        assert payload["temperature"] == config.LM_STUDIO_CONFIG["temperature"]
        assert payload["messages"] == [{"role": "user", "content": "q"}]

    @patch("requests.post")
    def test_request_url_ends_with_chat_completions(self, mock_post):
        import config

        mock_post.return_value = _mock_lm_response()
        _generate([{"role": "user", "content": "q"}])
        args, _ = mock_post.call_args
        url = args[0]
        assert url == f"{config.LM_STUDIO_CONFIG['api_base']}/chat/completions"

    @patch("requests.post")
    def test_missing_usage_returns_empty_dict(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "ответ"}}]}
        mock_post.return_value = resp
        _, usage = _generate([{"role": "user", "content": "q"}])
        assert usage == {}


# ═══════════════════════════════════════════════════════════════════════════
# 8. answer_question — end-to-end (all deps mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnswerQuestion:
    """End-to-end orchestration with embedding, retrieval, and LLM mocked."""

    def test_empty_query_returns_fallback_without_generation(self):
        result = answer_question("")
        assert result["answer"] == "Пожалуйста, задайте вопрос."
        assert result["sources"] == []
        assert result["context_chunks"] == 0

    def test_whitespace_query_returns_fallback(self):
        result = answer_question("   ")
        assert result["answer"] == "Пожалуйста, задайте вопрос."
        assert result["context_chunks"] == 0

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_no_relevant_chunks_returns_no_context_answer(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = [_make_chunk(score=0.05)]  # below threshold
        result = answer_question("несвязанный вопрос")
        assert result["answer"] == NO_CONTEXT_ANSWER
        assert result["sources"] == []
        assert result["context_chunks"] == 0
        # LLM must NOT be called when there is no context.
        mock_post.assert_not_called()

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_successful_answer_has_all_fields(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = [_make_chunk(score=0.90, text="Корпус на Стромынке.")]
        mock_post.return_value = _mock_lm_response(text="Главный корпус на ул. Стромынка, 20.")

        result = answer_question("Где главный корпус?")

        assert result["query"] == "Где главный корпус?"
        assert "Стромынка" in result["answer"]
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) >= 1
        assert result["context_chunks"] == 1
        assert isinstance(result["retrieval_scores"], list)
        assert result["generation_time"] >= 0
        assert isinstance(result["usage"], dict)

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_lm_studio_unreachable_raises_connection_error(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = [_make_chunk(score=0.90)]
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(ConnectionError):
            answer_question("вопрос")

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_sources_deduplicated_in_result(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = [
            _make_chunk(url="https://mirea.ru/a", score=0.9),
            _make_chunk(url="https://mirea.ru/a", score=0.8),  # same URL
            _make_chunk(url="https://mirea.ru/b", score=0.7),
        ]
        mock_post.return_value = _mock_lm_response()
        result = answer_question("вопрос")
        urls = [s["url"] for s in result["sources"]]
        assert urls == ["https://mirea.ru/a", "https://mirea.ru/b"]

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_retrieval_scores_reflect_returned_chunks(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = [
            _make_chunk(score=0.91),
            _make_chunk(url="https://mirea.ru/b", score=0.77),
        ]
        mock_post.return_value = _mock_lm_response()
        result = answer_question("вопрос")
        assert result["retrieval_scores"] == [0.91, 0.77]
        assert result["context_chunks"] == 2

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_top_k_passed_through_to_retrieval(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = []
        answer_question("вопрос", top_k=12)
        _, kwargs = mock_search.call_args
        assert kwargs.get("top_k") == 12

    @patch("requests.post")
    @patch("src.embeddings.indexer.search_qdrant")
    @patch.object(answer_mod, "_get_embedding_model")
    def test_generation_time_is_positive_on_success(
        self, mock_model, mock_search, mock_post
    ):
        mock_model.return_value = _mock_embedding_model()
        mock_search.return_value = [_make_chunk(score=0.9)]
        mock_post.return_value = _mock_lm_response()
        result = answer_question("вопрос")
        assert result["generation_time"] >= 0
