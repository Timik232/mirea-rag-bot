#!/usr/bin/env python3
"""Consistency check: verify faiss metadata, qdrant points, and golden_qa
relevant_chunk_ids are all mutually aligned.

Usage:
    python scripts/check_consistency.py

Exits 1 on drift.
"""
import json
import os
import sys

PROJECT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT)
import config


def main():
    errors = []

    # Load all sources
    chunks = json.load(
        open(os.path.join(config.DATA_PROCESSED_DIR, "chunks.json"), encoding="utf-8")
    )
    meta = json.load(
        open(os.path.join(config.DATA_FAISS_DIR, "metadata.json"), encoding="utf-8")
    )

    import numpy as np
    vecs = np.load(os.path.join(config.DATA_FAISS_DIR, "vectors.npy"))

    from qdrant_client import QdrantClient
    client = QdrantClient(path=config.QDRANT_PATH)
    qinfo = client.get_collection(collection_name=config.QDRANT_COLLECTION)
    qdrant_count = qinfo.points_count
    client.close()

    golden_path = os.path.join("mirea_data", "eval", "golden_qa.json")
    max_golden_id = 0
    if os.path.exists(golden_path):
        golden = json.load(open(golden_path, encoding="utf-8"))
        items = golden.get("items", golden if isinstance(golden, list) else [])
        if items:
            max_golden_id = max(
                max(q.get("relevant_chunk_ids", [0])) for q in items if q.get("relevant_chunk_ids")
            )

    # Checks
    chunk_n = len(chunks)
    meta_n = len(meta)
    vec_n = vecs.shape[0]

    print(f"chunks.json:    {chunk_n}")
    print(f"faiss metadata: {meta_n}")
    print(f"faiss vectors:  {vec_n} (dim={vecs.shape[1]})")
    print(f"qdrant points:  {qdrant_count}")
    print(f"golden max_id:  {max_golden_id}")

    if chunk_n != meta_n:
        errors.append(f"chunks({chunk_n}) != metadata({meta_n})")
    if chunk_n != vec_n:
        errors.append(f"chunks({chunk_n}) != vectors({vec_n})")
    if chunk_n != qdrant_count:
        errors.append(f"chunks({chunk_n}) != qdrant({qdrant_count})")
    if max_golden_id >= chunk_n:
        errors.append(f"golden max_id({max_golden_id}) >= chunk_count({chunk_n})")

    if errors:
        print(f"\n❌ DRIFT DETECTED:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print(f"\n✅ ALL CONSISTENT: {chunk_n} chunks, golden IDs in range")
        sys.exit(0)


if __name__ == "__main__":
    main()
