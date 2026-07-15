"""Hybrid retrieval evaluation — dense vs BM25 vs fused (RRF / weighted).

Extends the dense-only eval (scripts/eval_retrieval.py) with a BM25 backend
and hybrid fusion, so all four backends are scored on the same golden set:

  * numpy  : brute-force cosine (dense baseline)
  * bm25   : lexical BM25Okapi (sparse baseline)
  * rrf    : Reciprocal Rank Fusion of dense + bm25 (rank-based)
  * weighted : convex combination of min-max-normalised dense + bm25 scores

Also sweeps the fusion hyper-parameters (dense weight, RRF k) to find the
best operating point, and reuses CACHED per-query backend hits so the sweep
doesn't re-encode or re-search.

Data consistency is validated at startup: metadata count must equal vectors
shape, and golden relevant_chunk_ids must be in range. If the corpus was
regenerated after the golden set was built (stale ids), only in-range
questions are scored and a warning is printed.

Usage:
    python scripts/eval_hybrid.py
    python scripts/eval_hybrid.py --sweep     # hyper-parameter sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import numpy as np

import config
from src.retrieval.bm25 import BM25Index, tokenize_ru
from src.retrieval.hybrid import fuse_rrf, fuse_weighted

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_hybrid")

GOLDEN_PATH = PROJ / "mirea_data/eval/golden_qa.json"
VECTORS_PATH = PROJ / "mirea_data/faiss/vectors.npy"
METADATA_PATH = PROJ / "mirea_data/faiss/metadata.json"
REPORT_PATH = PROJ / "mirea_data/eval/hybrid_report.json"

K_VALUES = [3, 5]
MAX_K = max(K_VALUES)
POOL = 15  # candidate pool per backend before fusion (over-retrieve)


# --------------------------------------------------------------------------- #
# Metrics (same definitions as eval_retrieval.py)
# --------------------------------------------------------------------------- #

def per_query(retrieved: list[int], rel: set[int]) -> dict:
    rank_first = next((i + 1 for i, cid in enumerate(retrieved) if cid in rel), None)
    out: dict = {"rr": (1.0 / rank_first) if rank_first else 0.0}
    for k in K_VALUES:
        topk = retrieved[:k]
        hits = sum(1 for cid in topk if cid in rel)
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / len(rel) if rel else 0.0
        out[f"hit@{k}"] = 1 if hits > 0 else 0
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out: dict = {"n": n}
    out["mrr"] = round(sum(r["rr"] for r in rows) / n, 4)
    for k in K_VALUES:
        out[f"precision@{k}"] = round(sum(r[f"precision@{k}"] for r in rows) / n, 4)
        out[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in rows) / n, 4)
        out[f"hit_rate@{k}"] = round(sum(r[f"hit@{k}"] for r in rows) / n, 4)
    return out


# --------------------------------------------------------------------------- #
# Backends (cached per query)
# --------------------------------------------------------------------------- #

def encode_queries(items, model):
    """Encode all golden questions once, return {qid: vec}."""
    qs = [f"query: {it['question']}" for it in items]
    embs = model.encode(qs, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return {it["id"]: embs[i] for i, it in enumerate(items)}


def dense_search_all(vecs_n, qvecs, k=POOL):
    """Dense cosine for every query. Returns {qid: [(id, score), ...]}."""
    out = {}
    for qid, qv in qvecs.items():
        scores = (qv @ vecs_n.T)
        idx = np.argsort(-scores)[:k]
        out[qid] = [(int(i), float(scores[i])) for i in idx]
    return out


def bm25_search_all(bm25_idx, items, k=POOL):
    """BM25 for every query. Returns {qid: [(id, score), ...]}."""
    out = {}
    for it in items:
        out[it["id"]] = bm25_idx.search(it["question"], top_k=k)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="sweep fusion hyper-parameters")
    args = ap.parse_args()

    # ---- Load + validate data consistency ----
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    vecs = np.load(VECTORS_PATH).astype(np.float32)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items_all = golden["items"]
    n_chunks = len(metadata)
    print(f"metadata: {n_chunks} chunks | vectors: {vecs.shape} | golden: {len(items_all)} items")

    if len(metadata) != vecs.shape[0]:
        print(f"FATAL: metadata ({len(metadata)}) != vectors ({vecs.shape[0]})")
        return 2

    # Filter golden to in-range questions (corpus may have been regenerated).
    total_rel = sum(len(it["relevant_chunk_ids"]) for it in items_all)
    items = [it for it in items_all if all(c < n_chunks for c in it["relevant_chunk_ids"])]
    n_dropped = len(items_all) - len(items)
    if n_dropped:
        print(f"WARNING: {n_dropped}/{len(items_all)} questions dropped — "
              f"relevant_chunk_ids out of range (corpus regenerated to {n_chunks} "
              f"chunks after golden set was built). Scoring {len(items)} questions.")
    if len(items) < 10:
        print(f"WARNING: only {len(items)} valid questions — metrics unreliable")

    # ---- Normalise vectors for cosine ----
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vecs_n = (vecs / norms).astype(np.float32)

    # ---- Load model + BM25 ----
    try:
        from sentence_transformers import SentenceTransformer
        print(f"loading {config.EMBEDDING_MODEL} ...")
        model = SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"MODEL UNAVAILABLE: {e!r}")
        return 2

    texts = [m.get("text", "") for m in metadata]
    print("building BM25 index ...")
    bm25_idx = BM25Index.build(texts)

    # ---- Run backends once (cached) ----
    print("encoding queries ...")
    qvecs = encode_queries(items, model)
    print("running dense + bm25 ...")
    t0 = time.time()
    dense_cache = dense_search_all(vecs_n, qvecs)
    bm25_cache = bm25_search_all(bm25_idx, items)
    print(f"  backends done in {time.time()-t0:.1f}s")

    # ---- Helper: score a fusion config over all queries ----
    def score_fusion(method, **kw):
        rows = []
        for it in items:
            rel = set(it["relevant_chunk_ids"])
            qid = it["id"]
            d_hits = dense_cache[qid]
            b_hits = bm25_cache[qid]
            if method == "dense":
                fused = d_hits[:MAX_K]
            elif method == "bm25":
                fused = b_hits[:MAX_K]
            elif method == "rrf":
                fused = fuse_rrf(
                    [[d for d, _ in d_hits], [d for d, _ in b_hits]],
                    k=kw.get("rrf_k", 60),
                    weights=[kw.get("dw", 0.5), 1 - kw.get("dw", 0.5)],
                )[:MAX_K]
            elif method == "weighted":
                fused = fuse_weighted(
                    [d_hits, b_hits],
                    weights=[kw.get("dw", 0.5), 1 - kw.get("dw", 0.5)],
                )[:MAX_K]
            retrieved = [d for d, _ in fused]
            m = per_query(retrieved, rel)
            m["id"] = qid
            rows.append(m)
        return aggregate(rows), rows

    # ---- Score all backends ----
    backends = {}
    for name in ("dense", "bm25"):
        agg, _ = score_fusion(name)
        backends[name] = agg
    for name in ("rrf", "weighted"):
        agg, _ = score_fusion(name, dw=0.5, rrf_k=60)
        backends[name] = agg

    # ---- Print results ----
    print("\n" + "=" * 64)
    print(f"HYBRID RETRIEVAL METRICS  (mean over {len(items)} questions, {n_chunks} chunks)")
    print("=" * 64)
    print(f"  {'backend':<12} {'MRR':>7} {'P@3':>7} {'R@3':>7} {'HR@3':>7} {'R@5':>7} {'HR@5':>7}")
    for name in ("dense", "bm25", "rrf", "weighted"):
        o = backends[name]
        print(f"  {name:<12} {o['mrr']:>7.4f} {o['precision@3']:>7.4f} "
              f"{o['recall@3']:>7.4f} {o['hit_rate@3']:>7.4f} "
              f"{o['recall@5']:>7.4f} {o['hit_rate@5']:>7.4f}")

    sweep_results = {}
    if args.sweep:
        print("\n--- RRF dense-weight sweep (k=60) ---")
        print(f"  {'dw':>5} {'MRR':>7} {'HR@3':>7} {'R@3':>7}")
        best_rrf = (0, None)
        for dw_int in range(0, 11):
            dw = dw_int / 10.0
            agg, _ = score_fusion("rrf", dw=dw, rrf_k=60)
            tag = f"rrf_dw{dw:.1f}"
            sweep_results[tag] = agg
            print(f"  {dw:>5.1f} {agg['mrr']:>7.4f} {agg['hit_rate@3']:>7.4f} {agg['recall@3']:>7.4f}")
            if agg["mrr"] > best_rrf[0]:
                best_rrf = (agg["mrr"], (dw, 60))

        print("\n--- weighted dense-weight sweep ---")
        print(f"  {'dw':>5} {'MRR':>7} {'HR@3':>7} {'R@3':>7}")
        best_w = (0, None)
        for dw_int in range(0, 11):
            dw = dw_int / 10.0
            agg, _ = score_fusion("weighted", dw=dw)
            tag = f"weighted_dw{dw:.1f}"
            sweep_results[tag] = agg
            print(f"  {dw:>5.1f} {agg['mrr']:>7.4f} {agg['hit_rate@3']:>7.4f} {agg['recall@3']:>7.4f}")
            if agg["mrr"] > best_w[0]:
                best_w = (agg["mrr"], dw)

        print(f"\nbest RRF: dw={best_rrf[1][0]:.1f}, k=60 → MRR={best_rrf[0]:.4f}")
        print(f"best weighted: dw={best_w[1]:.1f} → MRR={best_w[0]:.4f}")

    # ---- Save report ----
    report = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_chunks": n_chunks,
            "n_questions_total": len(items_all),
            "n_questions_scored": len(items),
            "n_dropped_stale": n_dropped,
            "embedding_model": config.EMBEDDING_MODEL,
            "pool": POOL,
        },
        "backends": backends,
    }
    if args.sweep:
        report["sweep"] = sweep_results
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
