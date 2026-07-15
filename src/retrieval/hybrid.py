"""Hybrid retrieval — fuse sparse (BM25) and dense (E5/Qdrant) signals.

Dense embeddings capture semantics (paraphrase, synonyms); BM25 captures
exact lexical overlap (codes, names, numbers like "260 баллов"). Hybrid
retrieval combines both so a query that phrasing-shifts a fact still hits
the right chunk, and a query with a rare proper noun isn't drowned by
semantic neighbours.

Two fusion strategies (both emit scores normalised to [0,1] so the
production ``RELEVANCE_THRESHOLD`` still gates the answer pipeline):

  * **RRF** (Reciprocal Rank Fusion, Cormack et al. 2009) — rank-based,
    parameter-light, robust to score-scale mismatch between backends.
    ``score(d) = Σ_i w_i / (k + rank_i(d))``, then min-max normalised.

  * **Weighted** (convex combination) — each backend's raw scores are
    min-max normalised to [0,1] over the candidate pool, then summed with
    weights. More expressive (you can bias toward dense), but sensitive to
    the normalisation choice.

Architecture: a small ``HybridRetriever`` owns one BM25 index + one dense
search callable and returns the same ``[{id, score, payload}]`` shape that
``src.embeddings.indexer.search_qdrant`` produces, so it drops into
``src.rag.answer._retrieve`` unchanged.

Usage:
    from src.retrieval.hybrid import HybridRetriever
    r = HybridRetriever.from_config()
    hits = r.search(query_vec, query_text="проходной балл", top_k=5)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Optional

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from src.retrieval.bm25 import BM25Index, default_index_path, tokenize_ru  # noqa: E402

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Score normalisation helpers
# --------------------------------------------------------------------------- #

def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Normalise a {id: score} mapping to [0, 1] via min-max.

    Degenerate cases: empty → {}; single point or constant → all 1.0 (so the
    lone candidate is never discarded by a downstream threshold).
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# --------------------------------------------------------------------------- #
# Fusion functions (operate on pre-computed hit lists)
# --------------------------------------------------------------------------- #

HitList = list[tuple[int, float]]
"""A ranked list of ``(doc_id, score)`` tuples from one backend."""


def fuse_rrf(
    rankings: list[list[int]],
    k: int = 60,
    weights: Optional[list[float]] = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked id-lists.

    Args:
        rankings: One ranked list of doc ids per backend (highest relevance
            first). Scores are NOT used — only rank position matters, which
            is why RRF tolerates incomparable score scales.
        k: RRF constant (default 60, the value from the original paper).
            Higher k dampens the advantage of top ranks; 60 is a robust
            default across corpora.
        weights: Per-backend weights (default uniform = 1/n). Lets you bias
            toward a stronger backend, e.g. [0.6, 0.4] for dense-heavy.

    Returns:
        ``(doc_id, fused_score)`` descending. Scores are min-max normalised
        to [0, 1] so the result is threshold-comparable with dense cosine.
    """
    n = len(rankings)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"len(weights)={len(weights)} != len(rankings)={n}")

    raw: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            raw[doc_id] = raw.get(doc_id, 0.0) + w / (k + rank)
    fused = sorted(_minmax(raw).items(), key=lambda kv: kv[1], reverse=True)
    return fused


def fuse_weighted(
    hits: list[HitList],
    weights: Optional[list[float]] = None,
    norm: str = "minmax",
) -> list[tuple[int, float]]:
    """Convex combination of per-backend score lists (normalised to [0,1]).

    Args:
        hits: One ranked ``(id, score)`` list per backend.
        weights: Per-backend weights (default uniform). Must sum to ~1.0 for
            the output to stay in [0,1]; the function normalises them anyway.
        norm: Score normalisation over the candidate pool — "minmax" (default)
            or "max" (divide by max, keeps 0 anchor).

    Returns:
        ``(doc_id, fused_score)`` descending, scores in [0, 1].
    """
    n = len(hits)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0 / n] * n
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()  # normalise to sum=1 defensively

    # Normalise each backend's scores over its own candidate set, then merge.
    merged: dict[int, float] = {}
    for backend_hits, w in zip(hits, weights):
        if not backend_hits:
            continue
        raw = {doc_id: score for doc_id, score in backend_hits}
        if norm == "minmax":
            normed = _minmax(raw)
        else:  # "max"
            mx = max(raw.values()) or 1.0
            normed = {d: s / mx for d, s in raw.items()}
        for d, s in normed.items():
            merged[d] = merged.get(d, 0.0) + float(w) * s
    fused = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    return fused


# --------------------------------------------------------------------------- #
# HybridRetriever — orchestrates dense + BM25 + fusion
# --------------------------------------------------------------------------- #

# Callable signature for a dense backend: (query_vec, top_k) -> HitList.
# Lets us plug Qdrant, FAISS, or numpy without coupling here.
DenseSearchFn = Callable[[np.ndarray, int], HitList]


class HybridRetriever:
    """Combine a dense vector backend with a BM25 lexical index.

    The dense backend is injected as a callable (DIP) so this class has no
    hard dependency on Qdrant — tests inject a numpy stub, production injects
    ``search_qdrant``. BM25 is owned directly (it's cheap and local).

    Over-retrieve-and-fuse: each backend returns ``pool_factor * top_k``
    candidates (default 3x), fusion is over the union, then we truncate to
    ``top_k``. This catches cases where the relevant chunk is rank 8 in one
    backend but rank 1 in the other.
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense_search: DenseSearchFn,
        method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        pool_factor: int = 3,
        metadata: Optional[list[dict]] = None,
    ) -> None:
        self.bm25 = bm25
        self.dense_search = dense_search
        self.method = method
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.pool_factor = pool_factor
        # Chunk metadata (text/url/section) keyed by positional id, used to
        # enrich fused results with payloads for the answer layer. Loaded
        # lazily from metadata.json when None.
        self._metadata = metadata

    @classmethod
    def from_config(cls, dense_search: Optional[DenseSearchFn] = None) -> "HybridRetriever":
        """Build from config defaults, loading the BM25 index from disk.

        Args:
            dense_search: Dense backend callable. If None, defaults to
                ``src.embeddings.indexer.search_qdrant`` (the production path).
        """
        if dense_search is None:
            from src.embeddings.indexer import search_qdrant
            dense_search = search_qdrant
        bm25 = BM25Index.load(default_index_path())
        meta_path = os.path.join(config.DATA_FAISS_DIR, "metadata.json")
        import json
        metadata = None
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        return cls(
            bm25=bm25,
            dense_search=dense_search,
            method=getattr(config, "HYBRID_FUSION", "rrf"),
            rrf_k=getattr(config, "HYBRID_RRF_K", 60),
            dense_weight=getattr(config, "HYBRID_DENSE_WEIGHT", 0.5),
            pool_factor=getattr(config, "HYBRID_POOL_FACTOR", 3),
            metadata=metadata,
        )

    @staticmethod
    def _as_tuples(hits: HitList | list[dict]) -> HitList:
        """Coerce a hit list (dicts OR tuples) into ``[(id, score)]`` tuples.

        ``search_qdrant`` returns ``[{"id", "score", "payload"}]``; BM25 and
        the numpy stub return ``[(id, score)]``. Fusion only needs id+score,
        so we normalise here once instead of branching inside each backend.
        """
        out: HitList = []
        for h in hits:
            if isinstance(h, dict):
                out.append((int(h["id"]), float(h.get("score", 0.0))))
            else:
                out.append((int(h[0]), float(h[1])))
        return out

    def fuse(
        self,
        dense_hits: HitList | list[dict],
        bm25_hits: HitList | list[dict],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Fuse two pre-computed hit lists and truncate to ``top_k``.

        Exposed separately from :meth:`search` so the eval harness can reuse
        cached backend results across many fusion parameter combinations
        without re-querying.
        """
        dense_t = self._as_tuples(dense_hits)
        bm25_t = self._as_tuples(bm25_hits)
        if self.method == "weighted":
            # dense_weight applies to the DENSE backend, BM25 gets the rest.
            dw = self.dense_weight
            fused = fuse_weighted(
                [dense_t, bm25_t],
                weights=[dw, 1.0 - dw],
                norm="minmax",
            )
        else:
            # RRF uses id-order from the already-ranked hit lists.
            fused = fuse_rrf(
                rankings=[
                    [d for d, _ in dense_t],
                    [d for d, _ in bm25_t],
                ],
                k=self.rrf_k,
                weights=[self.dense_weight, 1.0 - self.dense_weight],
            )
        return fused[:top_k]

    def search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Run dense + BM25 retrieval and return fused results.

        Args:
            query_vec: Pre-encoded dense query vector (E5, 384-d).
            query_text: Raw query string (tokenised for BM25).
            top_k: Final number of fused results.

        Returns:
            List of dicts ``{"id", "score", "payload"}`` mirroring
            ``search_qdrant``'s output shape so callers are backend-agnostic.
            Payloads come from the dense backend (Qdrant payloads) when
            available, falling back to the metadata.json loaded at init.
        """
        pool = max(top_k * self.pool_factor, top_k)
        dense_hits = self.dense_search(query_vec, pool)
        bm25_hits = self.bm25.search(query_text, top_k=pool)
        fused = self.fuse(dense_hits, bm25_hits, top_k)

        # Build payload map from dense results (they carry full payloads
        # from Qdrant).
        dense_payloads: dict[int, dict] = {}
        for item in dense_hits:
            # Dense hits can be either dicts (search_qdrant shape) or tuples.
            if isinstance(item, dict):
                dense_payloads[item["id"]] = item.get("payload", {})

        results = []
        for doc_id, score in fused:
            payload = dense_payloads.get(doc_id)
            if payload is None and self._metadata:
                if 0 <= doc_id < len(self._metadata):
                    payload = self._metadata[doc_id]
            if payload is None:
                payload = {}
            results.append({"id": doc_id, "score": score, "payload": payload})
        return results


__all__ = [
    "HybridRetriever",
    "fuse_rrf",
    "fuse_weighted",
    "_minmax",
]
