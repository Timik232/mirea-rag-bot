"""Quick golden QA retrieval evaluation."""
import json, sys, os
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer
from src.embeddings.indexer import search_qdrant
import config

_model = None
def encode_query(text):
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model.encode(f"query: {text}", convert_to_numpy=True).astype('float32')

with open("mirea_data/eval/golden_qa.json", encoding="utf-8") as f:
    golden = json.load(f)
items = golden["items"] if isinstance(golden, dict) else golden
total = len(items)
hit_at_3 = 0
hit_at_5 = 0
mrr_scores = []

for item in items:
    q = item["question"]
    relevant = set(item.get("relevant_chunk_ids", []))
    if not relevant:
        continue
    vec = encode_query(q)
    results = search_qdrant(vec, top_k=5)
    retrieved = [r["id"] for r in results]

    top3 = set(retrieved[:3])
    top5 = set(retrieved[:5])

    if relevant & top3:
        hit_at_3 += 1
    if relevant & top5:
        hit_at_5 += 1

    rr = 0.0
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            rr = 1.0 / i
            break
    mrr_scores.append(rr)

print(f"Golden set: {total} questions")
print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3/total:.1%}")
print(f"Hit@5: {hit_at_5}/{total} = {hit_at_5/total:.1%}")
print(f"MRR:   {sum(mrr_scores)/total:.3f}")
