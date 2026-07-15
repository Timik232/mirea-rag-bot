"""Tests for src/pipeline/llm_cleaner.py — LLM-augmented HTML cleaning.

TDD RED phase: these tests define the contract for the llm_cleaner module.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.pipeline.llm_cleaner import clean_html, clean_item, clean_batch, llm_summarize


# ---------------------------------------------------------------------------
# 1. HTML stripping
# ---------------------------------------------------------------------------
class TestCleanHtml:
    """clean_html must strip tags, decode entities, normalise whitespace."""

    def test_strips_tags(self):
        assert "hello" == clean_html("<p>hello</p>")

    def test_decodes_entities(self):
        result = clean_html("A &amp; B &lt; C &gt; D &nbsp; E")
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "&gt;" not in result
        assert "&nbsp;" not in result
        assert "A & B < C > D" in result

    def test_normalizes_whitespace(self):
        result = clean_html("  hello   \n  world  \t  ")
        assert result == "hello world"

    def test_complex_html(self):
        html = "<div><h1>Title</h1><p>Some &quot;text&quot; here</p></div>"
        result = clean_html(html)
        assert "<" not in result
        assert ">" not in result
        assert "&quot;" not in result
        assert "Title" in result
        assert '"text"' in result


# ---------------------------------------------------------------------------
# 2. LLM summarization (mocked)
# ---------------------------------------------------------------------------
class TestLlmSummarize:
    """llm_summarize must call litellm.completion and return cleaned output."""

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_returns_shorter_text(self, mock_litellm):
        long_text = "word " * 200  # 1000 chars
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Short summary."
        mock_litellm.completion.return_value = mock_response

        result = llm_summarize(long_text)
        assert len(result) < len(long_text)
        assert result == "Short summary."

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_uses_config(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_litellm.completion.return_value = mock_response

        custom_config = {
            "model": "test-model",
            "api_base": "http://test:1234",
            "api_key": "test-key",
            "temperature": 0.5,
            "max_tokens": 100,
        }
        llm_summarize("some text", config=custom_config)

        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("model") == "test-model"
        assert call_kwargs.kwargs.get("api_base") == "http://test:1234"


# ---------------------------------------------------------------------------
# 3. LLM output validation
# ---------------------------------------------------------------------------
class TestLlmOutputValidation:
    """LLM output must be non-empty, no HTML tags, reasonable content."""

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_no_html_in_output(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "<b>Clean</b> text"
        mock_litellm.completion.return_value = mock_response

        result = llm_summarize("input text", config={"model": "m", "api_base": "b", "api_key": "k", "temperature": 0, "max_tokens": 10})
        # The module should strip HTML from LLM output too
        assert "<b>" not in result

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_non_empty_output(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  "
        mock_litellm.completion.return_value = mock_response

        result = llm_summarize("some input text", config={"model": "m", "api_base": "b", "api_key": "k", "temperature": 0, "max_tokens": 10})
        # If LLM returns empty/whitespace, should fall back to original
        assert result == "some input text"


# ---------------------------------------------------------------------------
# 4. Fallback when LLM fails
# ---------------------------------------------------------------------------
class TestFallback:
    """When litellm raises, must return HTML-stripped text (not summarized)."""

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_exception_returns_html_stripped(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("API down")

        html_text = "<p>Hello &amp; world</p>"
        result = llm_summarize(html_text)
        assert result == "Hello & world"

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_timeout_exception(self, mock_litellm):
        mock_litellm.completion.side_effect = TimeoutError("timed out")

        result = llm_summarize("<div>Some text</div>")
        assert result == "Some text"


# ---------------------------------------------------------------------------
# 5. clean_item preserves dict structure
# ---------------------------------------------------------------------------
class TestCleanItem:
    """clean_item must return dict with url, title, text/content keys preserved."""

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_preserves_structure_with_text_key(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_litellm.completion.return_value = mock_response

        item = {
            "url": "https://example.com",
            "title": "<b>Title</b>",
            "text": "<p>Long content &amp; stuff</p>",
            "section": "edu",
            "subsection": "sub",
        }
        result = clean_item(item, use_llm=True)
        assert "url" in result
        assert "title" in result
        assert "text" in result
        assert result["url"] == "https://example.com"
        assert "<b>" not in result["title"]
        assert "<p>" not in result["text"]

    @patch("src.pipeline.llm_cleaner.litellm")
    def test_preserves_structure_with_content_key(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_litellm.completion.return_value = mock_response

        item = {
            "url": "https://example.com",
            "title": "Title",
            "content": "<div>Content here</div>",
        }
        result = clean_item(item, use_llm=True)
        assert "text" in result
        assert "<div>" not in result["text"]

    def test_no_llm_skips_summarization(self):
        item = {
            "url": "https://example.com",
            "title": "Title",
            "text": "<p>Just HTML &amp; nothing else</p>",
        }
        result = clean_item(item, use_llm=False)
        assert result["text"] == "Just HTML & nothing else"
        assert result["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# 6. Batch processing
# ---------------------------------------------------------------------------
class TestCleanBatch:
    """clean_batch processes multiple items with rate limiting."""

    @patch("src.pipeline.llm_cleaner.time")
    @patch("src.pipeline.llm_cleaner.litellm")
    def test_processes_all_items(self, mock_litellm, mock_time):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Cleaned"
        mock_litellm.completion.return_value = mock_response
        mock_time.sleep = MagicMock()

        items = [
            {"url": "https://a.com", "title": "A", "text": "<p>Text A</p>"},
            {"url": "https://b.com", "title": "B", "text": "<p>Text B</p>"},
            {"url": "https://c.com", "title": "C", "text": "<p>Text C</p>"},
        ]
        results = clean_batch(items, use_llm=True)
        assert len(results) == 3
        assert all("text" in r for r in results)

    @patch("src.pipeline.llm_cleaner.time")
    @patch("src.pipeline.llm_cleaner.litellm")
    def test_rate_limits_between_calls(self, mock_litellm, mock_time):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_litellm.completion.return_value = mock_response
        mock_time.sleep = MagicMock()

        items = [
            {"url": f"https://{i}.com", "title": str(i), "text": f"<p>Text {i}</p>"}
            for i in range(3)
        ]
        clean_batch(items, use_llm=True)
        # sleep(0.3) called between items: 2 calls for 3 items
        assert mock_time.sleep.call_count == 2
        mock_time.sleep.assert_called_with(0.3)

    def test_batch_without_llm(self):
        items = [
            {"url": "https://a.com", "title": "A", "text": "<p>Hello &amp; world</p>"},
            {"url": "https://b.com", "title": "B", "text": "<p>Foo &amp; bar</p>"},
        ]
        results = clean_batch(items, use_llm=False)
        assert len(results) == 2
        assert results[0]["text"] == "Hello & world"
        assert results[1]["text"] == "Foo & bar"
