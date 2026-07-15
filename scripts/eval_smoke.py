"""Smoke-test the golden dataset against the real embedding index.

Loads the pre-built vectors.npy (155 x 384, multilingual-e5-small), encodes
each golden question with the E5 ``query:`` prefix, and reports whether the
pinned ``relevant_chunk_ids`` surface in top-3 / top-5.

This is QA for the golden dataset itself — it does NOT compute final metrics
(that's task t_8ab04e61). It only confirms the questions are findable.
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import numpy as np

golden = json.loads((PROJ / "mirea_data/eval/golden_qa.json").read_text(encoding="utf-8"))
items = golden["items"]

vecs = np.load(PROJ / "mirea_data/faiss/vectors.npy").astype(np.float32)
# normalise
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-10)
vecs_n = (vecs / norms).astype(np.float32)
print("vectors:", vecs.shape)

try:
    from sentence_transformers import SentenceTransformer
    import config
    print("loading model", config.EMBEDDING_MODEL, "...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
except Exception as e:
    print("MODEL UNAVAILABLE, aborting smoke test:", repr(e))
    raise SystemExit(0)

def search(q, k=5):
    emb = model.encode([f"query: {q}"], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores = emb @ vecs_n.T
    idx = np.argsort(-scores[0])[:k]
    return idx.tolist(), scores[0][idx].tolist()

hit3 = hit5 = 0
mrr_sum = 0.0
worst = []
for it in items:
    idx, scores = search(it["question"], k=5)
    rel = set(it["relevant_chunk_ids"])
    rank = next((r + 1 for r, i in enumerate(idx) if i in rel), None)
    in3 = any(i in rel for i in idx[:3])
    in5 = any(i in rel for i in idx[:5])
    hit3 += int(in3)
    hit5 += int(in5)
    if rank:
        mrr_sum += 1.0 / rank
    if not in3:
        worst.append((it["id"], it["question"][:60], rel, idx[:3]))

n = len(items)
print(f"\nRecall@3 = {hit3}/{n} = {hit3/n:.2%}")
print(f"Recall@5 = {hit5}/{n} = {hit5/n:.2%}")
print(f"MRR      = {mrr_sum/n:.3f}")
print(f"\n{len(worst)} questions whose relevant chunk is NOT in top-3:")
for qid, q, rel, top in worst:
    print(f"  {qid}: {q}")
    print(f"      relevant={sorted(rel)} got top3={top}")
