"""Cross-encoder re-ranking evaluation (task t_007083df).

Measures the effect of adding a cross-encoder re-ranking stage on top of the
existing bi-encoder (multilingual-e5-small) retrieval, over the 50-question
golden QA set.

Experimental design — isolate the re-ranking effect:
  For every question we first pull the SAME bi-encoder shortlist of
  ``RERANK_CANDIDATES`` (default 20) chunks from Qdrant. We then produce TWO
  rankings from that identical candidate pool:

    * **baseline**  : ordered by bi-encoder cosine score      (current prod)
    * **rerank**    : ordered by cross-encoder (query,passage) logit

  Because the candidate pool is shared, any metric delta is attributable purely
  to the re-ordering — not to recall of the first stage. This is the standard
  way to evaluate a re-ranker (cf. Nogueira & Cho 2019, BGE reranker paper).

Metrics (computed for k=3,5; no relevance threshold applied — we measure
ranking quality): MRR, Precision@k, Recall@k, HitRate@k.

Diagnostics reported alongside the headline numbers:
  * ``ranking_changed``     — queries where re-ranking altered the order at all
  * ``failures_fixed``      — relevant chunk outside top-3 in baseline, inside
                              top-3 after re-ranking
  * ``failures_broken``     — the reverse (a regression introduced by re-rank)
  * per-query before/after top-5, with both bi-encoder and CE scores

Outputs:
  mirea_data/eval/rerank_report.json   machine-readable report
  stdout                                human-readable summary + delta table

Usage:
  python scripts/eval_rerank.py
  python scripts/eval_rerank.py --candidates 20 --no-qdrant   # numpy shortlist
"""
from __future__ import annotations

import argparse
import json
import logging
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

# Reuse the exact metric definitions from the retrieval eval so the numbers are
# directly comparable (same per_query / aggregate / K_VALUES).
from scripts.eval_retrieval import (  # noqa: E402
    GOLDEN_PATH, VECTORS_PATH, METADATA_PATH, K_VALUES, MAX_K, THRESHOLD,
    per_query, aggregate, group_aggregate,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_rerank")

REPORT_PATH = PROJ / "mirea_data/eval/rerank_report.json"
RERANK_MODEL = getattr(config, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


# --------------------------------------------------------------------------- #
# Shortlist backends (bi-encoder stage 1)
# --------------------------------------------------------------------------- #

class NumpyShortlist:
    """Bi-encoder cosine over vectors.npy (baseline shortlist source)."""
    name = "numpy"

    def __init__(self, vecs_n, texts, model):
        self.vecs_n = vecs_n
        self.texts = texts
        self.model = model

    def shortlist(self, question: str, n: int) -> list[dict]:
        emb = self.model.encode(
            [f"query: {question}"], convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        scores = (emb @ self.vecs_n.T)[0]
        idx = np.argsort(-scores)[:n]
        return [
            {"id": int(i), "score": float(scores[i]), "text": self.texts[int(i)]}
            for i in idx
        ]


class QdrantShortlist:
    """Production bi-encoder shortlist via Qdrant (one shared client)."""
    name = "qdrant"

    def __init__(self, texts, model):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection = config.QDRANT_COLLECTION
        self.texts = texts
        self.model = model

    def shortlist(self, question: str, n: int) -> list[dict]:
        vec = self.model.encode(
            f"query: {question}", convert_to_numpy=True,
            normalize_embeddings=False,   # faithful to src/rag/answer.py
        ).astype(np.float32)
        res = self.client.query_points(
            collection_name=self.collection, query=vec.tolist(),
            limit=n, with_payload=False,
        )
        out = []
        for h in res.points:
            out.append({"id": int(h.id), "score": float(h.score),
                        "text": self.texts[int(h.id)]})
        return out

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Re-ranking (stage 2)
# --------------------------------------------------------------------------- #

def rerank(shortlist: list[dict], question: str, cross_encoder) -> list[dict]:
    """Re-score the bi-encoder shortlist with the cross-encoder.

    Returns the same dicts (with ``rerank_score`` added) ordered by CE logit.
    """
    pairs = [(question, c["text"][:4096]) for c in shortlist]
    scores = cross_encoder.predict(pairs, convert_to_numpy=True,
                                   batch_size=len(pairs))
    scores = np.asarray(scores).reshape(-1)
    for c, s in zip(shortlist, scores):
        c["rerank_score"] = round(float(s), 6)
    return sorted(shortlist, key=lambda c: c["rerank_score"], reverse=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=int,
                    default=getattr(config, "RERANK_CANDIDATES", 20),
                    help="bi-encoder shortlist size fed to the cross-encoder")
    ap.add_argument("--no-qdrant", action="store_true",
                    help="use numpy shortlist instead of Qdrant")
    args = ap.parse_args()
    N = args.candidates

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden["items"]
    n_items = len(items)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    texts = [m.get("text", "") for m in metadata]
    n_chunks = len(metadata)

    print(f"golden items: {n_items} | chunks: {n_chunks} | shortlist N={N}")
    print(f"bi-encoder : {config.EMBEDDING_MODEL}")
    print(f"cross-enc  : {RERANK_MODEL}")
    print(f"k values   : {K_VALUES} | threshold: {THRESHOLD}")

    # ---- load bi-encoder ----
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print(f"\nloading bi-encoder {config.EMBEDDING_MODEL} ...")
    bi = SentenceTransformer(config.EMBEDDING_MODEL)

    # ---- load cross-encoder ----
    print(f"loading cross-encoder {RERANK_MODEL} ...")
    t0 = time.time()
    ce = CrossEncoder(RERANK_MODEL, max_length=512)
    print(f"  cross-encoder ready in {time.time()-t0:.1f}s"
          f" (device={getattr(ce, '_device', 'auto')})")

    # ---- build shortlist backend ----
    if args.no_qdrant:
        vecs = np.load(VECTORS_PATH).astype(np.float32)
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)
        backend = NumpyShortlist((vecs / norms).astype(np.float32), texts, bi)
    else:
        backend = QdrantShortlist(texts, bi)
    print(f"shortlist backend: {backend.name}")

    # ---- run per question ----
    bi_rows, ce_rows = [], []     # metric rows
    per_q = []                    # detailed per-query record
    fixed, broken, changed = [], [], []
    bi_total, ce_total = 0.0, 0.0

    print(f"\nevaluating {n_items} questions ...")
    for k, it in enumerate(items, 1):
        q = it["question"]
        rel = set(it["relevant_chunk_ids"])

        # stage 1: bi-encoder shortlist (shared candidate pool)
        t_bi = time.time()
        pool = backend.shortlist(q, N)
        bi_total += time.time() - t_bi
        baseline = sorted(pool, key=lambda c: c["score"], reverse=True)

        # stage 2: cross-encoder re-rank of the SAME pool
        t_ce = time.time()
        reranked = rerank(pool, q, ce)
        ce_total += time.time() - t_ce

        # metric rows (id,score) lists — metrics only need ids + order
        bi_ranked = [(c["id"], c["score"]) for c in baseline]
        ce_ranked = [(c["id"], c["rerank_score"]) for c in reranked]

        mb = per_query(bi_ranked, rel); mb["id"] = it["id"]
        mb["category"] = it["category"]; mb["difficulty"] = it["difficulty"]
        mc = per_query(ce_ranked, rel); mc["id"] = it["id"]
        mc["category"] = it["category"]; mc["difficulty"] = it["difficulty"]
        bi_rows.append(mb); ce_rows.append(mc)

        # diagnostics: top-3 membership change for the relevant set
        bi_top3 = {c["id"] for c in baseline[:3]}
        ce_top3 = {c["id"] for c in reranked[:3]}
        bi_hit = bool(bi_top3 & rel)
        ce_hit = bool(ce_top3 & rel)
        if not bi_hit and ce_hit:
            fixed.append(it["id"])
        elif bi_hit and not ce_hit:
            broken.append(it["id"])
        order_changed = [c["id"] for c in baseline[:5]] != [c["id"] for c in reranked[:5]]
        if order_changed:
            changed.append(it["id"])

        per_q.append({
            "id": it["id"], "question": q,
            "category": it["category"], "difficulty": it["difficulty"],
            "relevant_chunk_ids": sorted(rel),
            "baseline_top5": [{"id": c["id"], "bi": round(c["score"], 4)} for c in baseline[:5]],
            "rerank_top5":   [{"id": c["id"], "bi": round(c["score"], 4),
                               "ce": c["rerank_score"]} for c in reranked[:5]],
            "baseline_rank1rel": mb["rr"] == 1.0,
            "rerank_rank1rel":   mc["rr"] == 1.0,
        })
        if k % 10 == 0:
            print(f"  {k}/{n_items}")

    # ---- aggregate ----
    report = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "t_007083df",
        "config": {
            "bi_encoder": config.EMBEDDING_MODEL,
            "cross_encoder": RERANK_MODEL,
            "shortlist_n": N,
            "shortlist_backend": backend.name,
            "k_values": K_VALUES,
            "relevance_threshold": THRESHOLD,
            "n_items": n_items,
            "n_chunks": n_chunks,
        },
        "overall": {
            "baseline": aggregate(bi_rows),
            "rerank": aggregate(ce_rows),
        },
        "by_category": {
            "baseline": group_aggregate(bi_rows, "category"),
            "rerank": group_aggregate(ce_rows, "category"),
        },
        "by_difficulty": {
            "baseline": group_aggregate(bi_rows, "difficulty"),
            "rerank": group_aggregate(ce_rows, "difficulty"),
        },
        "diagnostics": {
            "ranking_changed_count": len(changed),
            "ranking_changed_ids": changed,
            "failures_fixed_count": len(fixed),      # outside->inside top-3
            "failures_fixed_ids": fixed,
            "failures_broken_count": len(broken),    # inside->outside top-3
            "failures_broken_ids": broken,
            "bi_encoder_time_s": round(bi_total, 2),
            "cross_encoder_time_s": round(ce_total, 2),
            "bi_per_query_ms": round(bi_total / n_items * 1000, 1),
            "ce_per_query_ms": round(ce_total / n_items * 1000, 1),
        },
        "per_query": per_q,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")

    # ---- human-readable summary ----
    bo = report["overall"]["baseline"]
    ro = report["overall"]["rerank"]
    print("\n" + "=" * 70)
    print(f"RE-RANKING EVALUATION  (mean over {n_items} questions, N={N} shortlist)")
    print("=" * 70)
    hdr = f"  {'metric':<14} {'baseline':>10} {'+rerank':>10} {'delta':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    def row(name, b, r):
        d = r - b
        sign = "+" if d >= 0 else ""
        print(f"  {name:<14} {b:>10.4f} {r:>10.4f} {sign}{d:>8.4f}")
    row("MRR", bo["mrr"], ro["mrr"])
    for k in K_VALUES:
        row(f"P@{k}", bo[f"precision@{k}"], ro[f"precision@{k}"])
        row(f"R@{k}", bo[f"recall@{k}"], ro[f"recall@{k}"])
        row(f"HR@{k}", bo[f"hit_rate@{k}"], ro[f"hit_rate@{k}"])

    dg = report["diagnostics"]
    print("\ndiagnostics:")
    print(f"  ranking order changed (top-5) : {dg['ranking_changed_count']}/{n_items}")
    print(f"  failures fixed   (out->in top3): {dg['failures_fixed_count']}  {dg['failures_fixed_ids']}")
    print(f"  failures broken  (in->out top3): {dg['failures_broken_count']}  {dg['failures_broken_ids']}")
    print(f"  bi-encoder  latency/query : {dg['bi_per_query_ms']} ms")
    print(f"  cross-enc   latency/query : {dg['ce_per_query_ms']} ms"
          f"  (x{dg['ce_per_query_ms']/max(dg['bi_per_query_ms'],1e-3):.1f})")

    # by-category delta for MRR@3
    print(f"\nby category (HR@3 baseline -> rerank):")
    print(f"  {'category':<16} {'n':>3} {'base':>6} {'rerank':>7} {'dHR@3':>7}")
    bc, rc = report["by_category"]["baseline"], report["by_category"]["rerank"]
    for cat in sorted(bc):
        b = bc[cat]; r = rc[cat]
        d = r["hit_rate@3"] - b["hit_rate@3"]
        sign = "+" if d >= 0 else ""
        print(f"  {cat:<16} {b['n']:>3} {b['hit_rate@3']:>6.3f} "
              f"{r['hit_rate@3']:>7.3f} {sign}{d:>6.3f}")

    if hasattr(backend, "close"):
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
