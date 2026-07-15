"""Retrieval metrics over the golden QA dataset (task t_8ab04e61).

Computes precision@k, recall@k, hit_rate@k and MRR for k=3 (production
TOP_K_RETRIEVAL) and k=5, against TWO retrieval backends:

  * qdrant : the PRODUCTION path — Qdrant local collection ``mirea_chunks``
             with Cosine distance, queried exactly like
             ``src.embeddings.qdrant_indexer.QdrantIndexer.search`` (one shared
             client for all queries to avoid per-call lock churn; functionally
             identical to ``src.embeddings.indexer.search_qdrant``).
  * numpy  : brute-force cosine over ``mirea_data/faiss/vectors.npy``
             (cross-check / baseline; mirrors ``scripts/eval_smoke.py``).

The PRIMARY metrics (precision/recall/MRR) are computed on raw top-k ranking —
NO relevance threshold is applied, because we are measuring ranking quality, not
the answer pipeline's filtering. A secondary stat reports how many queries would
degrade to the NO_CONTEXT fallback under production RELEVANCE_THRESHOLD=0.35.

Query expansion (abbreviation → full form) is applied by default via
``src.rag.query_expansion.expand_query``, mirroring production behaviour
after task t_4a4bd2e3.  Use ``--no-expand`` to run the baseline without
expansion.  When query expansion is active the report also contains an
``expansion_delta`` sub-report comparing with/without.

Relevance: a retrieved chunk id is a hit iff id ∈ item.relevant_chunk_ids.
For MRR, rank = 1-based position of the first relevant hit in the ranked list.

Outputs:
  mirea_data/eval/retrieval_report.json   machine-readable report
  stdout                                   human-readable summary

Usage:
  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --no-expand     # baseline only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import numpy as np

import config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_retrieval")


def _expand(question: str) -> str:
    """Apply production query expansion (abbreviation → full form).

    Imported lazily so the eval still runs if the module is temporarily broken
    (e.g. during development) — a failure falls back to the raw question.
    """
    try:
        from src.rag.query_expansion import expand_query
        return expand_query(question)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("query expansion failed (%r); using raw question", e)
        return question

GOLDEN_PATH = PROJ / "mirea_data/eval/golden_qa.json"
VECTORS_PATH = PROJ / "mirea_data/faiss/vectors.npy"
METADATA_PATH = PROJ / "mirea_data/faiss/metadata.json"
REPORT_PATH = PROJ / "mirea_data/eval/retrieval_report.json"

K_VALUES = [3, 5]
MAX_K = max(K_VALUES)
THRESHOLD = getattr(config, "RELEVANCE_THRESHOLD", None) or 0.35


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class NumpyBackend:
    """Brute-force cosine over the prebuilt vectors.npy."""

    name = "numpy"

    def __init__(self) -> None:
        vecs = np.load(VECTORS_PATH).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.vecs_n = (vecs / norms).astype(np.float32)
        self.model = None
        self.expand = False  # set by main() via set_expand()

    def set_model(self, model) -> None:
        self.model = model

    def set_expand(self, flag: bool) -> None:
        self.expand = flag

    def search(self, question: str, k: int = MAX_K) -> list[tuple[int, float]]:
        if self.expand:
            question = _expand(question)
        emb = self.model.encode(
            [f"query: {question}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        scores = (emb @ self.vecs_n.T)[0]
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]


class QdrantBackend:
    """Production path: local Qdrant collection with Cosine distance.

    Uses one shared QdrantClient for the whole run (QdrantIndexer.search opens a
    fresh client per call, which is both slow and lock-prone over 50 queries).
    The query itself is identical: query_points(collection, query=vec, limit).
    """

    name = "qdrant"

    def __init__(self) -> None:
        from qdrant_client import QdrantClient  # noqa: delayed import

        self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection = config.QDRANT_COLLECTION
        self.model = None
        self.expand = False

    def set_model(self, model) -> None:
        self.model = model

    def set_expand(self, flag: bool) -> None:
        self.expand = flag

    def search(self, question: str, k: int = MAX_K) -> list[tuple[int, float]]:
        if self.expand:
            question = _expand(question)
        # Production _encode_query uses normalize_embeddings=False; Qdrant Cosine
        # normalizes internally, so this is faithful to src/rag/answer.py.
        vec = self.model.encode(
            f"query: {question}",
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        res = self.client.query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            limit=k,
            with_payload=False,
        )
        return [(int(h.id), float(h.score)) for h in res.points]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def per_query(retrieved: list[tuple[int, float]], rel: set[int]) -> dict:
    """Return per-query metric values for all k in K_VALUES + reciprocal rank."""
    ids = [r[0] for r in retrieved]
    rank_first = next((i + 1 for i, cid in enumerate(ids) if cid in rel), None)
    out: dict = {"rr": (1.0 / rank_first) if rank_first else 0.0}
    for k in K_VALUES:
        topk = ids[:k]
        hits = sum(1 for cid in topk if cid in rel)
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / len(rel) if rel else 0.0
        out[f"hit@{k}"] = 1 if hits > 0 else 0
    return out


def aggregate(rows: list[dict], ks: list[int] = K_VALUES) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out: dict = {"n": n}
    out["mrr"] = round(sum(r["rr"] for r in rows) / n, 4)
    for k in ks:
        out[f"precision@{k}"] = round(sum(r[f"precision@{k}"] for r in rows) / n, 4)
        out[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in rows) / n, 4)
        out[f"hit_rate@{k}"] = round(sum(r[f"hit@{k}"] for r in rows) / n, 4)
    return out


def group_aggregate(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {g: aggregate(rs) for g, rs in sorted(groups.items())}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-qdrant", action="store_true", help="numpy baseline only")
    ap.add_argument("--no-expand", action="store_true",
                    help="disable abbreviation query expansion (baseline)")
    args = ap.parse_args()
    expand_on = not args.no_expand

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    all_items = golden["items"]
    # Filter out unreachable items (no relevant_chunk_ids in corpus)
    items = [it for it in all_items if it.get("relevant_chunk_ids") and not it.get("unreachable")]
    n_unreachable = len(all_items) - len(items)
    n_items = len(items)
    n_chunks = len(json.loads(METADATA_PATH.read_text(encoding="utf-8")))
    print(f"golden items: {len(all_items)} (reachable: {n_items}, unreachable: {n_unreachable}) | chunks in index: {n_chunks}")
    print(f"embedding model: {config.EMBEDDING_MODEL} | k values: {K_VALUES} | "
          f"threshold: {THRESHOLD}")

    # load E5 model once, share across backends
    try:
        from sentence_transformers import SentenceTransformer
        print(f"loading {config.EMBEDDING_MODEL} ...")
        model = SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"MODEL UNAVAILABLE: {e!r}")
        return 2

    backends: list = [NumpyBackend()]
    if not args.no_qdrant:
        try:
            backends.append(QdrantBackend())
        except Exception as e:
            print(f"[warn] Qdrant unavailable, skipping production backend: {e!r}")
    for b in backends:
        b.set_model(model)
        b.set_expand(expand_on)

    print(f"query expansion: {'ON' if expand_on else 'OFF'}")

    # run retrieval — one pass per backend, cache results by qid
    results: dict = {}           # backend -> list of (item, retrieved)
    for b in backends:
        print(f"running backend '{b.name}' ...")
        t0 = time.time()
        rows = [(it, b.search(it["question"], k=MAX_K)) for it in items]
        results[b.name] = rows
        print(f"  backend '{b.name}' done in {time.time()-t0:.1f}s")

    by_id = {b.name: {it["id"]: retrieved for it, retrieved in results[b.name]}
             for b in backends}

    # build per-query metric rows (keyed by qid for each backend)
    metric_rows: dict[str, list[dict]] = {}
    for b in backends:
        mrows = []
        for it, retrieved in results[b.name]:
            rel = set(it["relevant_chunk_ids"])
            m = per_query(retrieved, rel)
            m["id"] = it["id"]
            m["category"] = it["category"]
            m["difficulty"] = it["difficulty"]
            mrows.append(m)
        metric_rows[b.name] = mrows

    # overall + grouped aggregates per backend
    backends_report: dict = {}
    for b in backends:
        mrows = metric_rows[b.name]
        backends_report[b.name] = {
            "overall": aggregate(mrows),
            "by_category": group_aggregate(mrows, "category"),
            "by_difficulty": group_aggregate(mrows, "difficulty"),
        }

    # per-query detail + failures (use primary = qdrant if present else numpy)
    primary_name = "qdrant" if "qdrant" in results else "numpy"
    primary_rows = results[primary_name]
    per_query_detail = []
    failures = []          # relevant chunk NOT in top-3 (primary backend)
    below_threshold = 0    # top-1 score < THRESHOLD (would degrade to NO_CONTEXT)
    for it, retrieved in primary_rows:
        rel = set(it["relevant_chunk_ids"])
        ids = [r[0] for r in retrieved]
        rank_first = next((i + 1 for i, cid in enumerate(ids) if cid in rel), None)
        top1_score = retrieved[0][1] if retrieved else 0.0
        if top1_score < THRESHOLD:
            below_threshold += 1
        detail = {
            "id": it["id"],
            "question": it["question"],
            "category": it["category"],
            "difficulty": it["difficulty"],
            "relevant_chunk_ids": sorted(rel),
            "rank_first_relevant": rank_first,
            "top1_score": round(top1_score, 4),
        }
        # top-5 retrieved ids+scores for every backend
        for b in backends:
            retrieved_b = by_id[b.name].get(it["id"], [])
            detail[f"{b.name}_top5"] = [
                {"id": cid, "score": round(s, 4)} for cid, s in retrieved_b[:5]
            ]
        per_query_detail.append(detail)
        if not any(cid in rel for cid in ids[:3]):
            failures.append({
                "id": it["id"],
                "question": it["question"],
                "category": it["category"],
                "relevant": sorted(rel),
                f"{primary_name}_top3": ids[:3],
            })

    # ── Expansion delta (baseline vs expanded) on the primary backend ──────
    # When query expansion is ON, re-run the primary backend WITHOUT expansion
    # so the report quantifies exactly what expansion bought (or cost).
    expansion_delta = None
    changed_queries: list[dict] = []
    if expand_on:
        primary_backend = next(b for b in backends if b.name == primary_name)
        primary_backend.set_expand(False)
        print(f"running baseline (no-expand) pass on '{primary_name}' ...")
        t0 = time.time()
        base_rows = [(it, primary_backend.search(it["question"], k=MAX_K))
                     for it in items]
        print(f"  baseline pass done in {time.time()-t0:.1f}s")
        primary_backend.set_expand(True)  # restore

        base_overall = aggregate([
            per_query(r, set(it["relevant_chunk_ids"]))
            for it, r in base_rows
        ])
        exp_overall = backends_report[primary_name]["overall"]
        # per-query RR comparison
        base_by_id = {it["id"]: r for it, r in base_rows}
        for it, retrieved in primary_rows:
            rel = set(it["relevant_chunk_ids"])
            rr_exp = per_query(retrieved, rel)["rr"]
            rr_base = per_query(base_by_id[it["id"]], rel)["rr"]
            if rr_exp != rr_base:
                changed_queries.append({
                    "id": it["id"],
                    "question": it["question"],
                    "rr_base": round(rr_base, 4),
                    "rr_expanded": round(rr_exp, 4),
                    "delta": round(rr_exp - rr_base, 4),
                })
        expansion_delta = {
            "baseline_overall": base_overall,
            "expanded_overall": exp_overall,
            "delta": {
                k: round(exp_overall[k] - base_overall[k], 4)
                for k in ("mrr", "precision@3", "recall@3", "hit_rate@3",
                          "precision@5", "recall@5", "hit_rate@5")
            },
            "changed_queries": changed_queries,
            "n_changed": len(changed_queries),
        }

    report = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "k_values": K_VALUES,
            "relevance_threshold": THRESHOLD,
            "n_items": n_items,
            "n_unreachable": len(all_items) - n_items,
            "n_chunks": n_chunks,
            "primary_backend": primary_name,
            "query_expansion": expand_on,
        },
        "backends": backends_report,
        "secondary": {
            "below_threshold_count": below_threshold,
            "below_threshold_frac": round(below_threshold / n_items, 4),
            "note": "queries whose top-1 score < relevance_threshold would "
                    "degrade to NO_CONTEXT_ANSWER in production",
        },
        "failures_top3": failures,
        "per_query": per_query_detail,
    }
    if expansion_delta is not None:
        report["expansion_delta"] = expansion_delta

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")

    # ---- human-readable summary ----
    print("\n" + "=" * 64)
    print("RETRIEVAL METRICS  (mean over {} questions)".format(n_items))
    print("=" * 64)
    for b in backends:
        o = backends_report[b.name]["overall"]
        print(f"\n[{b.name}]")
        print(f"  MRR          = {o['mrr']:.4f}")
        for k in K_VALUES:
            print(f"  Precision@{k} = {o[f'precision@{k}']:.4f}   "
                  f"Recall@{k} = {o[f'recall@{k}']:.4f}   "
                  f"HitRate@{k} = {o[f'hit_rate@{k}']:.4f}")
    print(f"\nprimary backend: {primary_name}")
    print(f"below threshold (<{THRESHOLD}, would degrade to NO_CONTEXT): "
          f"{below_threshold}/{n_items}")
    print(f"failures (relevant not in top-3, {primary_name}): {len(failures)}")
    for f in failures:
        print(f"  {f['id']} [{f['category']}] {f['question'][:55]}")
        print(f"       relevant={f['relevant']} top3={f[f'{primary_name}_top3']}")

    # by-category table for primary backend
    print(f"\nby category ({primary_name}):")
    print(f"  {'category':<16} {'n':>3} {'P@3':>6} {'R@3':>6} {'HR@3':>6} {'MRR':>6}")
    for cat, agg in backends_report[primary_name]["by_category"].items():
        print(f"  {cat:<16} {agg['n']:>3} {agg['precision@3']:>6.3f} "
              f"{agg['recall@3']:>6.3f} {agg['hit_rate@3']:>6.3f} {agg['mrr']:>6.3f}")
    print(f"\nby difficulty ({primary_name}):")
    for diff, agg in backends_report[primary_name]["by_difficulty"].items():
        print(f"  {diff:<10} {agg['n']:>3} P@3={agg['precision@3']:.3f} "
              f"R@3={agg['recall@3']:.3f} HR@3={agg['hit_rate@3']:.3f} MRR={agg['mrr']:.3f}")

    # expansion delta summary
    if expansion_delta is not None:
        d = expansion_delta["delta"]
        bo = expansion_delta["baseline_overall"]
        eo = expansion_delta["expanded_overall"]
        print("\n" + "-" * 64)
        print(f"QUERY EXPANSION DELTA  ({primary_name}, baseline → expanded)")
        print("-" * 64)
        print(f"  MRR          {bo['mrr']:.4f} → {eo['mrr']:.4f}  ({'+' if d['mrr']>=0 else ''}{d['mrr']:+.4f})")
        for k in K_VALUES:
            print(f"  Recall@{k}     {bo[f'recall@{k}']:.4f} → {eo[f'recall@{k}']:.4f}  "
                  f"({'+' if d[f'recall@{k}']>=0 else ''}{d[f'recall@{k}']:+.4f})   "
                  f"HitRate@{k} {bo[f'hit_rate@{k}']:.4f} → {eo[f'hit_rate@{k}']:.4f}  "
                  f"({'+' if d[f'hit_rate@{k}']>=0 else ''}{d[f'hit_rate@{k}']:+.4f})")
        print(f"  queries whose ranking changed: {expansion_delta['n_changed']}/{n_items}")
        for cq in sorted(expansion_delta["changed_queries"], key=lambda x: x["delta"]):
            tag = "↑" if cq["delta"] > 0 else "↓"
            print(f"    {tag} {cq['id']} [{cq['question'][:45]}]  "
                  f"{cq['rr_base']} → {cq['rr_expanded']}")

    for b in backends:
        if hasattr(b, "close"):
            b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
