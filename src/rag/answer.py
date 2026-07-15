"""RAG answer generation — retrieve relevant chunks and generate an answer.

This is the core module that ties together:
1. Query encoding (E5-small)
2. Qdrant retrieval
3. Prompt construction (Russian)
4. LLM generation (LM Studio / Gemma 4 E4B)

Usage:
    from src.rag.answer import answer_question
    result = answer_question("Где находится главный корпус МИРЭА?")
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Optional

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# System prompt selected via A/B prompt-engineering.
#
# Task t_b2d436d5: Variant B ("completeness + anti-refusal") beat the baseline
# (A) and the anti-vagueness variant (C). Task t_8dd0b704 then tested Variant D
# ("calibrated formatting") head-to-head against B over the 50-question golden
# set (gemma-4-26b gen, glm-5.2 judge):
#
#   B (n=50): overall 84.2/100, correctness 4.12, completeness 3.56
#   D (n=50): overall 85.5/100, correctness 4.18, completeness 3.78  <- winner
#
# D = B's completeness/anti-refusal base + two calibrated formatting directives:
#   (1) structured lists for list-type questions (boosted completeness +0.22);
#   (2) "preserve ranges as-is, don't invent exact figures" — this is the fix
#       for why C was rejected: C forbade ranges outright, which pushed the model
#       to hallucinate precise numbers (correctness -0.02). D keeps correctness
#       intact (+0.06 vs B) while still gaining completeness. See
#       mirea_data/eval/ab_report_bd.json for full results.
SYSTEM_PROMPT = (
    "Ты – справочный бот университета РТУ МИРЭА. "
    "Отвечай на вопросы абитуриентов и студентов на русском языке.\n\n"
    "Используй ТОЛЬКО информацию из предоставленного контекста. "
    "Не выдумывай факты, цифры, даты или названия.\n\n"
    "Внимательно изучи контекст и извлеки из него ВСЕ факты, относящиеся "
    "к вопросу: конкретные цифры, даты, суммы, названия, коды направлений. "
    "Если в контексте есть релевантная информация – обязательно используй "
    "её в ответе.\n\n"
    "Если вопрос предполагает перечисление нескольких пунктов (экзамены, "
    "направления подготовки, требования, контакты, документы), оформляй "
    "ответ структурированным списком – по одному пункту на элемент. "
    "Это делает ответ удобным для чтения.\n\n"
    "Приводи конкретные значения из контекста (цифры, даты, суммы, коды), "
    "когда они там есть. Если в контексте указан диапазон – приводи его "
    "как есть, не заменяя диапазон выдуманной точной цифрой.\n\n"
    "Говори «У меня нет информации по этому вопросу» ТОЛЬКО если в "
    "контексте действительно нет ничего, относящегося к вопросу. "
    "Не отказывайся от ответа, если в контексте есть хотя бы частичная "
    "информация. Не начинай ответ фразами «У меня нет точной информации», "
    "«Точной информации нет» и подобными — если в контексте есть число "
    "или факт, просто приведи его.\n\n"
    "Отвечай по существу, дружелюбным тоном. Начинай ответ сразу с "
    "существа дела, без приветствий («Здравствуйте», «Добрый день») и "
    "без вводных фраз («Я готов ответить», «Используя контекст»)."
)

# Minimum similarity score for a chunk to be considered relevant
RELEVANCE_THRESHOLD = 0.35
# Maximum chunks to include in context (to fit LM Studio context window)
MAX_CONTEXT_CHUNKS = 5
# Hallucination guard — minimum top-1 score for generation (task t_737c1b5e).
# If the best retrieved chunk scores below this, the pipeline refuses to
# generate rather than risk a hallucination on weak context. See config.py
# for calibration rationale. Overridable via config.CONFIDENCE_THRESHOLD.
CONFIDENCE_THRESHOLD = getattr(config, "CONFIDENCE_THRESHOLD", 0.55)
# When cross-encoder re-ranking is enabled, the CE score (already in [0, 1]
# via sentence-transformers sigmoid) is used as the confidence signal instead
# — wider dynamic range.
CE_CONFIDENCE_THRESHOLD = getattr(config, "CE_CONFIDENCE_THRESHOLD", 0.5)
# Fallback answer when no relevant chunks found
NO_CONTEXT_ANSWER = (
    "У меня нет информации по этому вопросу. "
    "Попробуйте переформулировать вопрос или обратитесь на официальный сайт mirea.ru."
)
# Hallucination guard — chunks were retrieved but the top score is too low to
# trust (task t_737c1b5e). We surface the sources so the user can judge for
# themselves, but we refuse to generate a possibly-fabricated answer.
LOW_CONFIDENCE_ANSWER = (
    "Я нашёл материалы, которые могут относиться к вашему вопросу, "
    "но недостаточно уверен в их релевантности, чтобы дать точный ответ. "
    "Пожалуйста, изучите источники ниже или уточните вопрос."
)
# Fallback answer when the LLM is unreachable or returns a broken response.
# Retrieval still succeeded, so we surface the sources instead of hiding them.
LLM_FAILURE_ANSWER = (
    "Не удалось сформировать ответ (служба генерации недоступна). "
    "По вашему вопросу найдены релевантные источники — см. ниже."
)
# Fallback answer when the embedding model or vector store itself failed, so we
# have nothing to show the user at all.
PIPELINE_FAILURE_ANSWER = (
    "Произошла ошибка при обработке запроса. "
    "Попробуйте повторить позже или переформулировать вопрос."
)

# Max query length (chars). Longer queries are truncated, not rejected — this
# keeps the bot responsive instead of erroring on a wall of text.
MAX_QUERY_LENGTH = getattr(config, "MAX_INPUT_LENGTH", 500)
# A query with fewer than this many *letters* is treated as noise.
MIN_QUERY_LETTERS = 2
# Generation robustness: how many times to retry a transient LM Studio failure
# (timeout, 5xx, malformed JSON) before degrading to LLM_FAILURE_ANSWER.
GENERATION_MAX_RETRIES = 2
GENERATION_RETRY_BACKOFF = 1.5  # seconds, multiplied per attempt
GENERATION_TIMEOUT = 60  # seconds per HTTP request

# Strip C0/C1 control characters except tab/newline/carriage-return, which we
# collapse to single spaces below. Keeps queries safe for the embedding model
# and the LLM prompt (guards against embedded NULs / prompt-injection via raw
# control chars).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Collapses any run of whitespace to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


class GenerationError(RuntimeError):
    """Raised when the LLM cannot produce a usable answer after retries."""

# Lazy-loaded model (singleton)
_embedding_model = None


def _get_embedding_model():
    """Load E5-small model lazily (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


# Lazy-loaded hybrid retriever (singleton). Only built when RETRIEVAL_MODE
# == "hybrid" — dense-only deployments never pay the BM25-index load cost.
_hybrid_retriever = None


def _get_hybrid_retriever():
    """Build the HybridRetriever lazily from config (singleton).

    Loads the BM25 index from ``config.BM25_INDEX_PATH`` and wires it to the
    production dense backend (``search_qdrant``). Raises if the index file is
    missing — callers in :func:`_retrieve` catch that and fall back to dense.
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from src.retrieval.hybrid import HybridRetriever
        logger.info("Loading hybrid retriever (BM25 + dense)")
        _hybrid_retriever = HybridRetriever.from_config()
    return _hybrid_retriever


def _validate_query(query) -> tuple[Optional[str], Optional[str]]:
    """Validate and sanitise raw user input.

    Returns ``(clean_query, error)`` where exactly one is non-None:
      * ``(None, "reason")``  — input is unusable; caller should reply ``reason``.
      * ``(clean_query, None)`` — input is usable, proceed.

    Handles: wrong type (None / numbers coerced, containers rejected), control
    characters, excessive length (truncated), and input with no meaningful
    textual content (only punctuation / digits).
    """
    # ── Type handling ──────────────────────────────────────────────────────
    if query is None:
        return None, "Пожалуйста, задайте вопрос."
    if isinstance(query, str):
        text = query
    elif isinstance(query, (int, float)):
        # Numeric input is almost certainly not a real question, but coerce
        # rather than crash — e.g. a typed "2024" might be a year query.
        text = str(query)
    else:
        # list / dict / bytes / arbitrary objects — reject, do not stringify
        # (could leak huge structures into the prompt).
        logger.warning("Rejected non-string query of type %s", type(query).__name__)
        return None, "Пожалуйста, задайте вопрос текстом."

    # ── Sanitise: control chars + whitespace ───────────────────────────────
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # ── Length: truncate, don't reject ─────────────────────────────────────
    if len(text) > MAX_QUERY_LENGTH:
        logger.info("Truncating query from %d to %d chars", len(text), MAX_QUERY_LENGTH)
        text = text[:MAX_QUERY_LENGTH].strip()

    if not text:
        return None, "Пожалуйста, задайте вопрос."

    # ── Meaningfulness: require at least MIN_QUERY_LETTERS word-characters ─
    # Rejects pure punctuation / symbol spam ("???!!!" , "-----") and pure
    # digit strings, which carry no semantic signal for embedding.
    letters = re.findall(r"\w", text, flags=re.UNICODE)
    if len(letters) < MIN_QUERY_LETTERS:
        return None, "Слишком короткий вопрос. Уточните, что вас интересует."

    return text, None


def _normalize_query(query: str) -> str:
    """Normalise a validated query before embedding.

    Structural validation (type / length / control chars) is done in
    :func:`_validate_query`.  Semantic normalisation applied here, in order:

      1. **Abbreviation expansion** – МИРЭА-domain abbreviations the user may
         type (ЕГЭ, ПГАС, ВУЦ, ИИТ, …) are expanded to their full Russian form
         so the embedder can match chunks that use the spelled-out form.  The
         original token is kept; see :mod:`src.rag.query_expansion` for the
         exact table, rationale, and the measured effect on retrieval.

      2. **Text normalisation** – whitespace collapse, punctuation folding,
         keyboard-layout fix (Russian typed on a QWERTY layout), and optional
         case folding.  Each sub-stage is independently feature-flagged in
         ``config.NORMALIZE_*`` so it can be benchmarked in isolation; see
         :mod:`src.rag.query_normalization`.

    All stages are no-ops on already-clean text.
    """
    from src.rag.query_expansion import expand_query
    from src.rag.query_normalization import normalize_query

    expanded = expand_query(query)
    return normalize_query(
        expanded,
        whitespace=getattr(config, "NORMALIZE_WHITESPACE", True),
        punctuation=getattr(config, "NORMALIZE_PUNCTUATION", True),
        keyboard_layout=getattr(config, "NORMALIZE_KEYBOARD_LAYOUT", True),
        case=getattr(config, "NORMALIZE_CASE", False),
    )


def _encode_query(query: str) -> np.ndarray:
    """Encode a user query using E5 convention (prefix 'query: ')."""
    model = _get_embedding_model()
    prefixed = f"query: {query}"
    vec = model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=False)
    return vec.astype(np.float32)


def _retrieve(
    query_vec: np.ndarray,
    top_k: int = MAX_CONTEXT_CHUNKS,
    query: Optional[str] = None,
) -> list[dict]:
    """Search for relevant chunks via the configured retrieval mode.

    Three-stage pipeline (each stage optional):

      1. **Candidate retrieval** — ``config.RETRIEVAL_MODE`` selects:
           * ``"dense"``  : Qdrant cosine (bi-encoder), the original path.
           * ``"hybrid"`` : BM25 + dense fusion (RRF/weighted). Needs the
             BM25 index on disk (``scripts/build_bm25.py``); falls back to
             dense if the index is missing or loading fails.
      2. **Cross-encoder re-ranking** (optional, ``config.RERANK_ENABLED``):
         re-scores the candidate shortlist with a cross-encoder, keeping
         ``top_k``. The stage-1 score is preserved for thresholding.
      3. **Threshold filter** — drop chunks below ``RELEVANCE_THRESHOLD``.

    Returns an empty list both when nothing is above threshold and when the
    vector store itself fails — callers treat both as "no context" and fall
    back to the canned answer. A retrieval outage must never crash the caller.
    A cross-encoder outage degrades silently to the stage-1 order (see
    :func:`src.rag.reranker.rerank_chunks`).
    """
    from src.rag.reranker import is_enabled as rerank_enabled, rerank_chunks

    do_rerank = rerank_enabled() and query is not None
    # When re-ranking, over-fetch the candidate shortlist; otherwise just top_k.
    fetch_k = max(getattr(config, "RERANK_CANDIDATES", 20), top_k) if do_rerank else top_k

    mode = getattr(config, "RETRIEVAL_MODE", "dense")
    stage_parts: list[str] = []
    results: list[dict] = []

    try:
        if mode == "hybrid" and query is not None:
            retriever = _get_hybrid_retriever()
            results = retriever.search(query_vec, query, top_k=fetch_k)
            stage_parts.append("hybrid")
        else:
            from src.embeddings.indexer import search_qdrant
            results = search_qdrant(query_vec, top_k=fetch_k)
            stage_parts.append("bi")
    except Exception as e:
        logger.error("Retrieval failed (mode=%s): %s", mode, e)
        # Hybrid can fail if the BM25 index is missing — fall back to dense
        # so the bot stays responsive instead of returning nothing.
        if mode == "hybrid":
            try:
                from src.embeddings.indexer import search_qdrant
                results = search_qdrant(query_vec, top_k=fetch_k)
                stage_parts.append("bi-fallback")
                logger.warning("Hybrid retrieval unavailable, used dense fallback")
            except Exception as e2:
                logger.error("Dense fallback also failed: %s", e2)
                return []
        else:
            return []

    if do_rerank:
        results = rerank_chunks(query, results, top_k=top_k)
        stage_parts.append("cross")

    # Filter by relevance threshold (stage-1 score: dense cosine or hybrid
    # fused score, both in a comparable [0,1]-ish range).
    relevant = [r for r in results if r.get("score", 0) >= RELEVANCE_THRESHOLD]
    stage = "+".join(stage_parts)
    logger.info("Retrieved %d chunks [%s], %d above threshold %.2f",
                len(results), stage, len(relevant), RELEVANCE_THRESHOLD)
    return relevant


def _check_confidence(chunks: list[dict]) -> tuple[bool, float, str]:
    """Hallucination guard — is the best chunk confident enough to generate?

    Evaluates the top-1 retrieval score against a confidence floor. When
    cross-encoder re-ranking is enabled, the CE score (sigmoid-normalised) is
    preferred over the bi-encoder cosine because its wider dynamic range
    separates relevant from irrelevant passages more cleanly.

    Args:
        chunks: Non-empty list of retrieved chunks (already filtered by
            ``RELEVANCE_THRESHOLD`` in :func:`_retrieve`), sorted by score desc.

    Returns:
        ``(passes, top_score, signal_name)``:
          * ``passes`` — True if the top score clears the threshold.
          * ``top_score`` — the score that was evaluated.
          * ``signal_name`` — ``"ce"`` if the cross-encoder score was used,
            ``"bi"`` for the bi-encoder cosine.
    """
    top = chunks[0]
    # Prefer the cross-encoder score when available — it carries a far stronger
    # relevance signal (joint query-passage attention vs. independent encoding).
    # sentence-transformers CrossEncoder already sigmoid-normalises bge-reranker
    # logits to [0, 1], so we use rerank_score directly without re-applying.
    ce_score = top.get("rerank_score")
    if ce_score is not None and rerank_is_enabled():
        top_score = float(ce_score)
        passes = top_score >= CE_CONFIDENCE_THRESHOLD
        signal = "ce"
    else:
        top_score = float(top.get("score", 0))
        passes = top_score >= CONFIDENCE_THRESHOLD
        signal = "bi"

    if not passes:
        logger.warning(
            "Confidence guard triggered: top-1 %s score %.4f < threshold %.2f — "
            "refusing generation to prevent hallucination",
            signal, top_score,
            CE_CONFIDENCE_THRESHOLD if signal == "ce" else CONFIDENCE_THRESHOLD,
        )
    return passes, top_score, signal


def rerank_is_enabled() -> bool:
    """Thin wrapper so ``_check_confidence`` doesn't import at module top-level.

    Defers the ``config.RERANK_ENABLED`` lookup to call time, matching how
    :func:`_retrieve` already checks it (config can be hot-swapped in tests).
    """
    from src.rag.reranker import is_enabled
    return is_enabled()


def _build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("payload", {}).get("text", "").strip()
        url = chunk.get("payload", {}).get("source_url", "")
        section = chunk.get("payload", {}).get("section", "")
        score = chunk.get("score", 0)

        header = f"[{i}]"
        if section:
            header += f" Раздел: {section};"
        header += f" Релевантность: {score:.1%}"
        if url:
            header += f" Источник: {url}"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def _build_messages(query: str, context: str, history: list[dict] | None = None) -> list[dict]:
    """Build chat messages for the LLM.

    ``history`` is an optional list of prior turns (``{"role", "content"}``),
    used to give the model conversational memory (the last N user+assistant
    pairs). It is inserted between the system prompt and the current user turn.
    """
    user_content = f"Контекст:\n{context}\n\nВопрос: {query}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Cap at the last 4 messages (≈2 user+assistant pairs) to fit context.
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


def _extract_answer(data: dict) -> str:
    """Pull the assistant message out of an OpenAI-style chat response.

    Raises ``GenerationError`` if the response is structurally valid JSON but
    missing the expected fields (empty ``choices``, no ``message``, etc.), or
    if the generated content is blank.
    """
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise GenerationError(
            f"LM Studio returned malformed response (missing choices/message): {e}"
        )
    if not isinstance(answer, str):
        raise GenerationError(
            f"LM Studio returned non-string content: {type(answer).__name__}"
        )
    answer = answer.strip()
    if not answer:
        raise GenerationError("LM Studio returned an empty answer")
    return answer


# ── Answer post-processing (formatting) ─────────────────────────────────────
# Task t_8dd0b704. Pure, idempotent normalisation applied to the raw LLM output
# before it is returned to the user. Touches presentation only — never content
# or facts. Every transform is no-op on already-clean text, so existing tests
# and cached A/B answers are unaffected.

# A markdown code fence wrapping the WHOLE answer, optionally with a language
# tag (```ru\n...\n```). Matched only at the very start/end so inline `code` is
# left untouched.
_CODE_FENCE_RE = re.compile(r"\A```[a-zA-Z0-9+-]*\s*\n?([\s\S]*?)\n?```\s*\Z")
# A leading label the model sometimes emits: "Ответ:", "Бот:", "Assistant:".
# Stripped only when it opens the answer — never mid-sentence.
_LEADING_LABEL_RE = re.compile(
    r"\A\s*(?:Ответ|Бот|Ассистент|Assistant|As an assistant)\s*[:\-.–—]\s*",
    flags=re.IGNORECASE,
)
# Three or more consecutive newlines → two (keeps paragraphs readable).
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# Trailing spaces on a line (including tabs) — removed for clean diffing/render.
_LINE_TRAILING_WS_RE = re.compile(r"[ \t]+$", flags=re.MULTILINE)

# Per the project's universal typographic rule: the short/en-dash (–, U+2013) is
# used everywhere; the long/em-dash (—, U+2014) is never used. We normalise any
# em-dash the model emits to an en-dash. (Hyphen-minus '-' in compound words and
# ranges like «220–260» is left as the model wrote it — only U+2014 is rewritten.)
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"


def _format_answer(text: str) -> str:
    """Normalise a generated answer for presentation.

    Pure and idempotent: clean prose in → same prose out. The transforms are
    presentation-only and applied in a fixed order:

      1. unwrap a whole-answer markdown code fence (```` ```...``` ````);
      2. strip a leading label (``Ответ:``, ``Бот:`` …) if present;
      3. rewrite em-dashes (U+2014) → en-dashes (U+2013);
      4. collapse 3+ blank lines to a single blank line;
      5. trim trailing per-line whitespace and the overall string.

    Content (facts, numbers, wording) is never altered. Returns the original
    string unchanged if it is already clean.
    """
    if not isinstance(text, str) or not text:
        return text

    # 1. Unwrap a code fence that wraps the entire answer.
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1)

    # 2. Drop a leading "Ответ:" / "Бot:" label.
    text = _LEADING_LABEL_RE.sub("", text, count=1)

    # 3. Em-dash → en-dash (universal typographic convention for this project).
    text = text.replace(_EM_DASH, _EN_DASH)

    # 4. Collapse runs of blank lines.
    text = _BLANK_RUN_RE.sub("\n\n", text)

    # 5. Trim trailing whitespace per line, then the whole string.
    text = _LINE_TRAILING_WS_RE.sub("", text).strip()
    return text


def _generate(messages: list[dict]) -> tuple[str, dict]:
    """Call LM Studio for answer generation, with retries and graceful failure.

    Strategy:
      * ``ConnectionError`` (server down) — fail fast, no retry; the server is
        almost certainly still down a second later.
      * Transient failures (timeout, 5xx, malformed/empty body) — retry with
        exponential backoff up to ``GENERATION_MAX_RETRIES`` extra attempts.
      * 4xx HTTP errors — no retry (client error won't fix itself).
      * After exhausting retries → raise :class:`GenerationError`.

    Returns ``(answer_text, usage_info)``.
    """
    import requests

    cfg = config.LM_STUDIO_CONFIG
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        # Gemma 4 is a reasoning model — disable the thinking phase so the
        # model answers directly instead of burning 300–500 reasoning tokens
        # per call (slow, noisy, and eats the context budget).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{cfg['api_base']}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}

    last_error: Optional[str] = None
    total_attempts = GENERATION_MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=GENERATION_TIMEOUT)
        except requests.exceptions.ConnectionError:
            # Server unreachable — no point retrying.
            logger.error("LM Studio not reachable at %s", url)
            raise ConnectionError(
                f"Не удалось подключиться к LM Studio ({url}). "
                "Убедитесь, что сервер запущен."
            )
        except requests.exceptions.Timeout:
            last_error = f"timeout after {GENERATION_TIMEOUT}s (attempt {attempt})"
            logger.warning("LM Studio request timed out: %s", last_error)
        except requests.exceptions.RequestException as e:
            last_error = f"network error: {e} (attempt {attempt})"
            logger.warning("LM Studio request error: %s", last_error)
        else:
            # Got an HTTP response — classify the status code.
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code} (attempt {attempt})"
                logger.warning("LM Studio server error: %s", last_error)
            elif resp.status_code >= 400:
                # Client error (bad request / auth) — won't improve on retry.
                logger.error("LM Studio client error HTTP %d: %s",
                             resp.status_code, resp.text[:300])
                raise GenerationError(
                    f"LM Studio отклонил запрос (HTTP {resp.status_code})"
                )
            else:
                # 2xx — parse JSON and extract the answer.
                try:
                    data = resp.json()
                except ValueError as e:
                    last_error = f"invalid JSON: {e} (attempt {attempt})"
                    logger.warning("LM Studio returned non-JSON body: %s", last_error)
                else:
                    try:
                        answer = _extract_answer(data)
                    except GenerationError as e:
                        last_error = str(e)
                        logger.warning("LM Studio unusable response: %s", last_error)
                    else:
                        usage = data.get("usage", {})
                        return answer, usage

        # Back off before the next attempt (skip after the final attempt).
        if attempt < total_attempts:
            delay = GENERATION_RETRY_BACKOFF ** attempt
            logger.info("Retrying in %.1fs (attempt %d/%d)...",
                        delay, attempt + 1, total_attempts)
            time.sleep(delay)

    raise GenerationError(
        f"LM Studio failed after {total_attempts} attempts: {last_error}"
    )


def _format_sources(chunks: list[dict]) -> list[dict]:
    """Extract source info from retrieved chunks."""
    sources = []
    seen_urls = set()
    for chunk in chunks:
        payload = chunk.get("payload", {})
        url = payload.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "section": payload.get("section", ""),
                "title": payload.get("source_title", ""),
                "score": round(chunk.get("score", 0), 4),
            })
    return sources


def answer_question(
    query: str,
    top_k: int = MAX_CONTEXT_CHUNKS,
    history: list[dict] | None = None,
) -> dict:
    """Answer a question using RAG (retrieval + generation).

    The pipeline degrades gracefully at every stage rather than crashing:

      * invalid / empty / non-text input -> canned prompt to rephrase;
      * embedding-model failure -> PIPELINE_FAILURE_ANSWER (nothing to retrieve);
      * vector-store failure or no relevant chunks -> NO_CONTEXT_ANSWER;
      * chunks found but top-1 score below CONFIDENCE_THRESHOLD (hallucination
        guard, task t_737c1b5e) -> LOW_CONFIDENCE_ANSWER with sources surfaced
        so the user can judge for themselves;
      * generation failure after retries (GenerationError) -> LLM_FAILURE_ANSWER,
        but retrieved sources are still returned so the user isn't empty-handed;
      * ConnectionError (LM Studio fully offline) propagates so the API layer
        can answer 503 -- a distinct "backend offline" signal.

    Args:
        query: User's question in Russian.
        top_k: Max chunks to retrieve.

    Returns:
        {
            "query": str,
            "answer": str,
            "sources": [{"url", "section", "title", "score"}],
            "context_chunks": int,
            "retrieval_scores": [float],
            "generation_time": float,
            "usage": dict,
            "confidence_score": float,   # top-1 score used by the guard
            "confidence_signal": str,    # "bi" or "ce"
            "confidence_guarded": bool,  # True if the guard blocked generation
        }
    """
    start = time.time()

    def _empty(answer: str) -> dict:
        return {
            "query": query if isinstance(query, str) else "",
            "answer": answer,
            "sources": [],
            "context_chunks": 0,
            "retrieval_scores": [],
            "generation_time": 0,
            "usage": {},
            "confidence_score": 0.0,
            "confidence_signal": "",
            "confidence_guarded": False,
        }

    # 0. Validate + sanitise input.
    clean_query, error = _validate_query(query)
    if error:
        return _empty(error)

    # Normalisation: abbreviation expansion + text normalisation (whitespace,
    # punctuation, keyboard-layout fix). See _normalize_query.
    clean_query = _normalize_query(clean_query)

    # 1b. Conversational query rewriting: if the user's question is short or
    #     contains pronouns, it likely refers to the previous turn. Prepend
    #     the last user question from history so the retriever gets context.
    #     The *generation* still sees the original question (in _build_messages),
    #     only the *retrieval* query is enriched.
    retrieval_query = clean_query
    if history:
        last_user_msgs = [m["content"] for m in history if m.get("role") == "user"]
        if last_user_msgs:
            prev_q = last_user_msgs[-1].strip()
            # Heuristic: short question or pronoun-heavy → needs context
            needs_ctx = (
                len(clean_query) < 30
                or any(w in clean_query.lower() for w in
                       ["он ", "она ", "оно ", "они ", "там", "это", "туда",
                        "ему", "её ", "его ", "их ", "а ", "а ещё", "кстати"])
            )
            if needs_ctx:
                retrieval_query = f"{prev_q} {clean_query}"

    # 1. Encode query. A failure here (model won't load) is unrecoverable.
    try:
        query_vec = _encode_query(retrieval_query)
    except Exception as e:
        logger.error("Query encoding failed: %s", e)
        elapsed = time.time() - start
        return {
            "query": clean_query,
            "answer": PIPELINE_FAILURE_ANSWER,
            "sources": [],
            "context_chunks": 0,
            "retrieval_scores": [],
            "generation_time": round(elapsed, 2),
            "usage": {},
            "confidence_score": 0.0,
            "confidence_signal": "",
            "confidence_guarded": False,
        }

    # 2. Retrieve. _retrieve() swallows vector-store errors and returns [].
    #    Pass clean_query so the cross-encoder (if enabled) can score passages
    #    against the raw question text.
    chunks = _retrieve(query_vec, top_k=top_k, query=retrieval_query)

    # 3. If no relevant chunks, return fallback.
    if not chunks:
        elapsed = time.time() - start
        return {
            "query": clean_query,
            "answer": NO_CONTEXT_ANSWER,
            "sources": [],
            "context_chunks": 0,
            "retrieval_scores": [],
            "generation_time": round(elapsed, 2),
            "usage": {},
            "confidence_score": 0.0,
            "confidence_signal": "",
            "confidence_guarded": False,
        }

    # 3b. Hallucination guard (task t_737c1b5e): if the best chunk's score is
    #     too low, refuse to generate rather than risk fabrication on weak
    #     context. Sources are still surfaced so the user can investigate.
    sources = _format_sources(chunks)
    scores = [round(c.get("score", 0), 4) for c in chunks]
    passes, conf_score, conf_signal = _check_confidence(chunks)
    if not passes:
        elapsed = time.time() - start
        return {
            "query": clean_query,
            "answer": LOW_CONFIDENCE_ANSWER,
            "sources": sources,
            "context_chunks": len(chunks),
            "retrieval_scores": scores,
            "generation_time": round(elapsed, 2),
            "usage": {},
            "confidence_score": round(conf_score, 4),
            "confidence_signal": conf_signal,
            "confidence_guarded": True,
        }

    # 4. Build context + prompt.
    context = _build_context(chunks)
    messages = _build_messages(clean_query, context, history=history)

    # 5. Generate. GenerationError -> degrade but keep sources; ConnectionError
    #    (server fully offline) propagates so the API can return 503.
    try:
        answer, usage = _generate(messages)
    except GenerationError as e:
        logger.error("Generation failed, degrading to fallback with sources: %s", e)
        elapsed = time.time() - start
        return {
            "query": clean_query,
            "answer": LLM_FAILURE_ANSWER,
            "sources": sources,
            "context_chunks": len(chunks),
            "retrieval_scores": scores,
            "generation_time": round(elapsed, 2),
            "usage": {},
            "confidence_score": round(conf_score, 4),
            "confidence_signal": conf_signal,
            "confidence_guarded": False,
        }

    elapsed = time.time() - start

    return {
        "query": clean_query,
        "answer": _format_answer(answer),
        "sources": sources,
        "context_chunks": len(chunks),
        "retrieval_scores": scores,
        "generation_time": round(elapsed, 2),
        "usage": usage,
        "confidence_score": round(conf_score, 4),
        "confidence_signal": conf_signal,
        "confidence_guarded": False,
    }


if __name__ == "__main__":
    import json
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="RAG answer test")
    parser.add_argument("query", nargs="?", default="Где находится главный корпус МИРЭА?")
    args = parser.parse_args()

    result = answer_question(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
