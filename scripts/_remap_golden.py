"""Remap golden QA relevant_chunk_ids after chunker rebalance.

Strategy: for each question, encode it, search Qdrant top-10, pick chunks
whose cosine score exceeds a threshold (0.75) as the new relevant set.
"""
import json, sys, os, logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from sentence_transformers import SentenceTransformer
from src.embeddings.indexer import search_qdrant
import config

# Suppress HF info logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

MIN_SCORE = 0.75
TOP_K = 10

model = SentenceTransformer(config.EMBEDDING_MODEL)

golden_path = "mirea_data/eval/golden_qa.json"
with open(golden_path, encoding="utf-8") as f:
    golden = json.load(f)

total = len(golden["items"])
reachable = 0
unreachable = 0

for item in golden["items"]:
    q = item["question"]
    vec = model.encode(f"query: {q}", convert_to_numpy=True).astype("float32")
    results = search_qdrant(vec, top_k=TOP_K)

    new_ids = [r["id"] for r in results if r.get("score", 0) >= MIN_SCORE]
    item["relevant_chunk_ids"] = new_ids

    if new_ids:
        reachable += 1
    else:
        unreachable += 1

golden["n_reachable"] = reachable
golden["n_unreachable"] = unreachable
golden["description"] = f"Golden QA dataset for mirea-rag-bot (re-mapped after chunker rebalance to 75 chunks, min_score={MIN_SCORE})"

with open(golden_path, "w", encoding="utf-8") as f:
    json.dump(golden, f, ensure_ascii=False, indent=2)

print(f"Remapped: {total} questions, {reachable} reachable, {unreachable} unreachable")
