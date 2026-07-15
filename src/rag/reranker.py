"""Cross-encoder re-ranking of bi-encoder retrieval candidates.

Two-stage retrieval:

  1. **Bi-encoder** (``intfloat/multilingual-e5-small``) encodes the query and
     each corpus chunk *independently* into a 384-d vector, then Qdrant returns
     the top-N by cosine similarity. This is fast (one vector per chunk, ANN
     search) but coarse — query and passage never "see" each other through
     attention.

  2. **Cross-encoder** (``BAAI/bge-reranker-v2-m3``) takes the query and each
     short-listed passage *jointly* through a single transformer, so every
     token attends across both. This captures fine-grained query-passage
     interaction the bi-encoder cannot, yielding markedly higher precision at
     the top of the ranking — at ~100x the per-pair cost, which is why it only
     re-scores the N candidates the bi-encoder already surfaced.

This module is the re-ranking layer only. It is *gated* by
``config.RERANK_ENABLED`` and is a pure no-op (identical output ordering) when
the flag is off, so existing behaviour and metrics are preserved unless the
pipeline opts in.

Score semantics: ``bge-reranker-v2-m3`` emits a raw logit per (query, passage)
pair; higher = more relevant. We do **not** apply a sigmoid — the absolute
value is not comparable to the bi-encoder cosine scores, so the re-ranked
chunks keep their *original* bi-encoder score in ``chunk["score"]`` for display
purposes and carry the cross-encoder score in ``chunk["rerank_score"]``. Only
the ordering changes.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Lazy singleton — the cross-encoder is ~2.2 GB and slow to load on first use.
_reranker_model = None


def _get_reranker_model():
    """Load the cross-encoder lazily (singleton).

    Uses ``sentence_transformers.CrossEncoder`` so the model definition, token
    pooling and relevance head come straight from the model card — no bespoke
    forward pass to maintain.
    """
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        model_name = getattr(config, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        device = getattr(config, "RERANK_DEVICE", None)  # let ST auto-pick
        logger.info("Loading cross-encoder reranker: %s", model_name)
        kwargs = {"max_length": 512}
        if device:
            kwargs["device"] = device
        _reranker_model = CrossEncoder(model_name, **kwargs)
        logger.info("Cross-encoder ready (device=%s)", _reranker_model._device
                    if hasattr(_reranker_model, "_device") else "auto")
    return _reranker_model


def _chunk_text(chunk: dict) -> str:
    """Pull the passage text out of a retrieved chunk (Qdrant payload shape)."""
    payload = chunk.get("payload") or chunk.get("metadata") or {}
    text = payload.get("text", "")
    if not text:
        # numpy/eval path where the text lives alongside, not nested
        text = chunk.get("text", "")
    return text.strip()


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: Optional[int] = None,
) -> list[dict]:
    """Re-score ``chunks`` with the cross-encoder and return them re-sorted.

    Args:
        query: The user question (already validated/normalised).
        chunks: Bi-encoder candidates, each carrying ``payload.text`` (Qdrant
            shape) or a top-level ``text`` field. Must be non-empty.
        top_k: If given, return only the top_k after re-ranking. Defaults to
            ``config.RERANK_TOP_K`` (or all of them if unset).

    Returns:
        The same chunk dicts, reordered by cross-encoder relevance (desc),
        each annotated with ``rerank_score``. The original ``score`` field is
        left untouched so callers can still show the bi-encoder similarity.

    Notes:
      * Empty passages are de-ranked to the end (the cross-encoder cannot score
        meaningfully without text).
      * The model is loaded on first call (~3-8 s on GPU, ~30 s on CPU); all
        subsequent calls reuse the singleton.
      * If the cross-encoder fails to load or score, the input ordering is
        returned unchanged — re-ranking is an enhancement, never a blocker.
    """
    if not chunks:
        return []

    if top_k is None:
        top_k = getattr(config, "RERANK_TOP_K", len(chunks))

    # Build (query, passage) pairs; remember which slot had empty text so we
    # can push those to the bottom regardless of model output.
    pairs = []
    empty_idx: list[int] = []
    for i, ch in enumerate(chunks):
        txt = _chunk_text(ch)
        if not txt:
            empty_idx.append(i)
            pairs.append((query, ""))  # placeholder, scored last anyway
        else:
            pairs.append((query, txt[:4096]))  # cap well beyond max_length

    try:
        model = _get_reranker_model()
        scores = model.predict(pairs, convert_to_numpy=True)
    except Exception as e:
        logger.error("Cross-encoder scoring failed, returning bi-encoder order: %s", e)
        # Return as-is (still honour top_k): re-rank must never crash retrieval.
        return chunks[:top_k]

    # Normalise scores to a 1-D float list.
    try:
        import numpy as _np
        if isinstance(scores, _np.ndarray):
            scores = scores.tolist()
    except Exception:
        pass
    if not isinstance(scores, list):
        scores = list(scores)
    # CrossEncoder.predict may return [[score]] for a single pair.
    flat: list[float] = []
    for s in scores:
        if isinstance(s, (list, tuple)):
            flat.append(float(s[-1]))
        else:
            flat.append(float(s))

    # Annotate + order. Empty-text chunks go to the very bottom.
    annotated = list(enumerate(chunks))
    for (i, ch), sc in zip(annotated, flat):
        ch["rerank_score"] = None if i in empty_idx else round(sc, 6)

    def _sort_key(item):
        i, ch = item
        if i in empty_idx:
            return (0, 0.0)             # empty text → bottom (sorts last, reverse=True)
        return (1, ch["rerank_score"])  # real text  → top bucket, by CE score desc

    ordered = [ch for _, ch in sorted(annotated, key=_sort_key, reverse=True)]
    return ordered[:top_k]


def is_enabled() -> bool:
    """True iff cross-encoder re-ranking is turned on in config."""
    return bool(getattr(config, "RERANK_ENABLED", False))
