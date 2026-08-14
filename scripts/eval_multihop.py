"""
Test harness for RAG improvement hypotheses.

Runs the multi-hop test set through different pipeline configurations
and reports retrieval + generation quality for each.

Usage (inside Docker):
    python scripts/eval_multihop.py --config baseline
    python scripts/eval_multihop.py --config filtered
    python scripts/eval_multihop.py --config decomposed
"""
import argparse
import json
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LITELLM_API_KEY", "")

import config
from src.rag.answer import (
    _encode_query, _normalize_query, _retrieve, _build_context,
    _build_messages, _generate, _format_answer, _check_confidence,
    MAX_CONTEXT_CHUNKS,
)
from src.embeddings.indexer import search_qdrant

# The eval harness runs as a separate process next to the live API. Since the
# API switched to a long-lived embedded-Qdrant client, the storage is locked
# for the process lifetime — point the harness at a copy (EVAL_QDRANT_PATH)
# or stop the API before running.
config.QDRANT_PATH = os.environ.get("EVAL_QDRANT_PATH", config.QDRANT_PATH)

logger = logging.getLogger(__name__)

# Sections that are real informational content (not VIKA conversational noise)
INFO_SECTIONS = {
    "applicants", "sveden", "contacts", "dormitories", "employment",
    "extracurricular", "education", "announcements", "qa_pairs",
    "student_portal",
}


def load_test_set():
    path = os.path.join(config.DATA_FAISS_DIR, "..", "eval", "multihop_test.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieve_filtered(query_vec, query_text, top_k=5):
    """Hypothesis #1: filter out VIKA noise, retrieve only informational chunks."""
    # Over-fetch, then filter by section, then take top_k
    raw = search_qdrant(query_vec, top_k=top_k * 5)
    filtered = [r for r in raw
                if r.get("payload", {}).get("section", "") in INFO_SECTIONS]
    return filtered[:top_k] if filtered else raw[:top_k]


def decompose_query(query, llm_config=None):
    """Hypothesis #3: decompose a multi-hop query into sub-questions via LLM."""
    if llm_config is None:
        llm_config = config.LM_STUDIO_CONFIG

    import requests
    prompt = (
        "Разбей вопрос абитуриента на простые под-вопросы. "
        "Каждый под-вопрос должен требовать одного факта. "
        "Верни ТОЛЬКО список под-вопросов, по одному на строку, без нумерации.\n\n"
        f"Вопрос: {query}\n\nПод-вопросы:"
    )
    try:
        resp = requests.post(
            f"{llm_config['api_base']}/chat/completions",
            headers={"Authorization": f"Bearer {llm_config['api_key']}"},
            json={
                "model": llm_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        subqs = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n")
                 if line.strip() and len(line.strip()) > 5]
        return subqs if subqs else [query]
    except Exception as e:
        logger.warning("Decomposition failed: %s", e)
        return [query]


def retrieve_decomposed(query, top_k=5, llm_config=None):
    """Hypothesis #3: decompose + retrieve per sub-question + merge."""
    subqs = decompose_query(query, llm_config)
    logger.info("Decomposed into %d sub-questions: %s", len(subqs), subqs)

    all_chunks = {}
    for sq in subqs:
        nq = _normalize_query(sq)
        vec = _encode_query(nq)
        chunks = retrieve_filtered(vec, nq, top_k=3)
        for c in chunks:
            cid = c["id"]
            if cid not in all_chunks or c["score"] > all_chunks[cid]["score"]:
                all_chunks[cid] = c

    merged = sorted(all_chunks.values(), key=lambda c: c["score"], reverse=True)
    return merged[:top_k]


def run_test(question, hyp="baseline", top_k=5, llm_config=None):
    """Run one question through the specified hypothesis configuration."""
    nq = _normalize_query(question)
    t0 = time.time()

    if hyp == "baseline":
        vec = _encode_query(nq)
        chunks = _retrieve(vec, top_k=top_k, query=nq)
    elif hyp == "filtered":
        vec = _encode_query(nq)
        chunks = retrieve_filtered(vec, nq, top_k=top_k)
    elif hyp == "decomposed":
        chunks = retrieve_decomposed(question, top_k=top_k, llm_config=llm_config)
    elif hyp == "filtered_decomposed":
        chunks = retrieve_decomposed(question, top_k=top_k, llm_config=llm_config)
        # Already filtered inside retrieve_decomposed via retrieve_filtered
    else:
        raise ValueError(f"Unknown hypothesis: {hyp}")

    retrieval_time = time.time() - t0

    # Match the production contract: _retrieve may return its full over-fetched
    # pool depending on the hypothesis path, but the LLM context never sees
    # more than MAX_CONTEXT_CHUNKS chunks.
    chunks = chunks[:MAX_CONTEXT_CHUNKS]

    if not chunks:
        return {
            "question": question,
            "hypothesis": hyp,
            "chunks": 0,
            "sections": [],
            "answer": "NO CHUNKS RETRIEVED",
            "retrieval_time": round(retrieval_time, 2),
        }

    sections = [c["payload"].get("section", "?") for c in chunks]
    scores = [round(c["score"], 3) for c in chunks]

    # Generate answer
    context = _build_context(chunks)
    messages = _build_messages(question, context)
    answer_text = ""
    try:
        answer_text, usage = _generate(messages) if llm_config is None else _generate_litellm(messages, llm_config)
        answer_text = _format_answer(answer_text)
    except Exception as e:
        answer_text = f"GENERATION FAILED: {e}"

    return {
        "question": question,
        "hypothesis": hyp,
        "chunks": len(chunks),
        "sections": sections,
        "scores": scores,
        "answer": answer_text[:500],
        "retrieval_time": round(retrieval_time, 2),
    }


def _generate_litellm(messages, llm_config):
    """Generate using LiteLLM proxy instead of LM Studio."""
    import requests
    resp = requests.post(
        f"{llm_config['api_base']}/chat/completions",
        headers={"Authorization": f"Bearer {llm_config['api_key']}"},
        json={
            "model": llm_config["model"],
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    return answer.strip(), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline",
                    choices=["baseline", "filtered", "decomposed", "filtered_decomposed"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm", default="lmstudio", choices=["lmstudio", "litellm"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)

    test_data = load_test_set()
    questions = test_data["questions"]

    llm_config = None
    if args.llm == "litellm":
        # External endpoint must come from the environment — never hardcode a
        # default here (this file is published).
        api_base = os.environ.get("LITELLM_API_BASE", "")
        if not api_base:
            sys.exit("--llm litellm requires LITELLM_API_BASE to be set "
                     "(plus optionally LITELLM_API_KEY / LITELLM_MODEL)")
        llm_config = {
            "api_base": api_base,
            "api_key": os.environ.get("LITELLM_API_KEY", ""),
            "model": os.environ.get("LITELLM_MODEL", "gpt-4o-mini"),
        }

    results = []
    for q in questions:
        print(f"\n{'='*70}")
        print(f"[{q['id']}] {q['question']}")
        print(f"Type: {q['type']}")
        r = run_test(q["question"], hyp=args.config, top_k=args.top_k, llm_config=llm_config)
        r["id"] = q["id"]
        r["type"] = q["type"]
        results.append(r)
        print(f"  Chunks: {r['chunks']}, Sections: {r['sections']}")
        print(f"  Scores: {r.get('scores', [])}")
        print(f"  Answer: {r['answer'][:200]}")
        print(f"  Time: {r['retrieval_time']}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({args.config}, top_k={args.top_k}, llm={args.llm}):")
    for r in results:
        print(f"  [{r['id']}] chunks={r['chunks']} | {r['answer'][:80]}...")

    out_path = os.path.join(config.DATA_FAISS_DIR, "..", "eval",
                            f"multihop_{args.config}_{args.llm}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
