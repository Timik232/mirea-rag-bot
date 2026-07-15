#!/usr/bin/env python3
"""Comprehensive data quality verification for the MIREA RAG pipeline.

Checks raw data, cleaned data, chunks, and Qdrant index for:
- HTML tags / markdown artifacts in text fields
- Empty or too-short fields
- Duplicate records (content_hash and near-duplicate)
- Encoding issues (mojibake, replacement chars)
- Size outliers
- Chunk quality (distribution, fragments, dupes)
- Qdrant index integrity
- Semantic sanity (test queries, near-duplicate vectors)

Usage:
    python scripts/verify_quality.py
    python scripts/verify_quality.py --skip-semantic   # skip embedding-based checks (faster)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_quality")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
MARKDOWN_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
REPLACEMENT_CHAR_RE = re.compile(r"[\ufffd]")
NON_BMP_RE = re.compile(r"[\U00010000-\U0001FFFF]")

# Known boilerplate / placeholder titles that carry no real content
BOILERPLATE_TITLES = {
    "Найдётся все",
    "Найдётся всё",
    "Главная",
    "Меню",
    "Карта сайта",
    "",
}

TEXT_FIELDS = ("text", "content", "title")
MIN_TEXT_LENGTH = 20

CHECKS_PASSED = 0
CHECKS_FAILED = 0
CHECKS_WARNED = 0


def _result(name: str, status: str, detail: str = "", issues: list | None = None):
    """Print and tally a check result."""
    global CHECKS_PASSED, CHECKS_FAILED, CHECKS_WARNED
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(status, "  ")
    print(f"  {icon} [{status}] {name}")
    if detail:
        print(f"          {detail}")
    if issues:
        for issue in issues[:10]:
            print(f"          • {issue}")
        if len(issues) > 10:
            print(f"          … and {len(issues) - 10} more")
    if status == "PASS":
        CHECKS_PASSED += 1
    elif status == "FAIL":
        CHECKS_FAILED += 1
    elif status == "WARN":
        CHECKS_WARNED += 1


def _load_json(path: str | Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_hash(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text)


# ---------------------------------------------------------------------------
# 1. Raw data checks
# ---------------------------------------------------------------------------

def check_raw_data(raw_dir: Path):
    print("\n" + "=" * 70)
    print("1. RAW DATA CHECKS")
    print("=" * 70)

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        _result("raw files exist", "FAIL", f"No JSON files in {raw_dir}")
        return

    total_items = 0
    all_html_issues = []
    all_empty_issues = []
    all_dupes = []
    all_encoding_issues = []
    all_boilerplate = []
    all_short = []
    hash_seen: dict[str, str] = {}

    for fpath in raw_files:
        data = _load_json(fpath)
        items = data if isinstance(data, list) else [data]
        total_items += len(items)
        file_name = fpath.name

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            combined_text = " ".join(
                str(item.get(k, "")) for k in TEXT_FIELDS
            )

            # HTML tags
            for key in TEXT_FIELDS:
                val = str(item.get(key, ""))
                if HTML_TAG_RE.search(val):
                    match = HTML_TAG_RE.search(val)
                    all_html_issues.append(
                        f"{file_name}[{i}].{key}: tag <{match.group()}>"
                    )

            # Empty / missing text
            text_val = str(item.get("text", "") or item.get("content", ""))
            if not text_val.strip():
                all_empty_issues.append(f"{file_name}[{i}]: empty text/content")
            elif len(text_val.strip()) < MIN_TEXT_LENGTH:
                all_short.append(
                    f"{file_name}[{i}]: text={len(text_val.strip())} chars"
                )

            # Encoding issues
            for key in TEXT_FIELDS:
                val = str(item.get(key, ""))
                if REPLACEMENT_CHAR_RE.search(val):
                    all_encoding_issues.append(
                        f"{file_name}[{i}].{key}: contains U+FFFD replacement char"
                    )
                if NON_BMP_RE.search(val) and key != "text":
                    # Emoji in titles is suspicious for a university site
                    all_encoding_issues.append(
                        f"{file_name}[{i}].{key}: contains non-BMP character"
                    )

            # Boilerplate titles
            title = str(item.get("title", "")).strip()
            if title in BOILERPLATE_TITLES:
                all_boilerplate.append(f"{file_name}[{i}]: title='{title}'")

            # Duplicates by content hash
            h = _norm_hash(combined_text)
            if h in hash_seen:
                all_dupes.append(
                    f"{file_name}[{i}] duplicates {hash_seen[h]}"
                )
            else:
                hash_seen[h] = f"{file_name}[{i}]"

    _result(
        f"raw files loaded ({len(raw_files)} files, {total_items} items)",
        "PASS",
    )

    _result(
        "no HTML tags in text fields",
        "FAIL" if all_html_issues else "PASS",
        issues=all_html_issues or None,
    )

    _result(
        "no empty text/content fields",
        "FAIL" if all_empty_issues else "PASS",
        issues=all_empty_issues or None,
    )

    _result(
        "no very short text (<20 chars)",
        "WARN" if all_short else "PASS",
        f"{len(all_short)} short records" if all_short else "",
        issues=all_short or None,
    )

    _result(
        "no encoding issues (U+FFFD, mojibake)",
        "FAIL" if all_encoding_issues else "PASS",
        issues=all_encoding_issues or None,
    )

    _result(
        "no boilerplate placeholder titles",
        "WARN" if all_boilerplate else "PASS",
        f"{len(all_boilerplate)} boilerplate titles" if all_boilerplate else "",
        issues=all_boilerplate or None,
    )

    _result(
        "no exact duplicate records",
        "WARN" if all_dupes else "PASS",
        f"{len(all_dupes)} duplicates" if all_dupes else "",
        issues=all_dupes or None,
    )

    return total_items


# ---------------------------------------------------------------------------
# 2. Cleaned data checks
# ---------------------------------------------------------------------------

def check_cleaned_data(processed_dir: Path, raw_item_count: int):
    print("\n" + "=" * 70)
    print("2. CLEANED DATA CHECKS")
    print("=" * 70)

    cleaned_path = processed_dir / "cleaned.json"
    if not cleaned_path.exists():
        _result("cleaned.json exists", "FAIL", f"Not found at {cleaned_path}")
        return

    data = _load_json(cleaned_path)
    items = data if isinstance(data, list) else [data]

    _result(f"cleaned.json loaded ({len(items)} items)", "PASS")

    # Retention rate
    if raw_item_count > 0:
        retention = len(items) / raw_item_count * 100
        if retention < 50:
            _result(
                f"retention rate (raw→cleaned)",
                "WARN",
                f"{len(items)}/{raw_item_count} = {retention:.1f}% — over half the data was dropped",
            )
        else:
            _result(
                f"retention rate",
                "PASS",
                f"{len(items)}/{raw_item_count} = {retention:.1f}%",
            )

    # Markdown artifacts (common after scraping)
    md_bold_issues = []
    md_link_issues = []
    html_issues = []
    no_text_issues = []
    no_hash_issues = []

    for i, item in enumerate(items):
        text = str(item.get("text", ""))
        title = str(item.get("title", ""))

        if MARKDOWN_BOLD_RE.search(text):
            count = len(MARKDOWN_BOLD_RE.findall(text))
            md_bold_issues.append(f"[{i}] title='{title[:40]}': {count} **bold** markers")

        if MARKDOWN_LINK_RE.search(text):
            count = len(MARKDOWN_LINK_RE.findall(text))
            md_link_issues.append(f"[{i}] title='{title[:40]}': {count} [md links]()")

        if HTML_TAG_RE.search(text):
            html_issues.append(f"[{i}] title='{title[:40]}'")

        if not text.strip():
            no_text_issues.append(f"[{i}]: empty text")

        if "content_hash" not in item:
            no_hash_issues.append(f"[{i}]")

    _result(
        "no markdown **bold** artifacts",
        "WARN" if md_bold_issues else "PASS",
        f"{len(md_bold_issues)} items with ** markers" if md_bold_issues else "",
        issues=md_bold_issues or None,
    )

    _result(
        "no markdown [link]() artifacts",
        "WARN" if md_link_issues else "PASS",
        f"{len(md_link_issues)} items with md links" if md_link_issues else "",
        issues=md_link_issues or None,
    )

    _result(
        "no HTML tags after cleaning",
        "FAIL" if html_issues else "PASS",
        issues=html_issues or None,
    )

    _result(
        "no records without text",
        "FAIL" if no_text_issues else "PASS",
        issues=no_text_issues or None,
    )

    _result(
        "all records have content_hash",
        "WARN" if no_hash_issues else "PASS",
        issues=no_hash_issues or None,
    )

    # Section distribution
    sections = collections.Counter(item.get("section", "?") for item in items)
    print(f"\n  📊 Section distribution ({len(items)} items):")
    for section, count in sections.most_common():
        print(f"          {section:30s} {count:5d}")

    return len(items)


# ---------------------------------------------------------------------------
# 3. Chunks checks
# ---------------------------------------------------------------------------

def check_chunks(processed_dir: Path, cleaned_count: int):
    print("\n" + "=" * 70)
    print("3. CHUNKS CHECKS")
    print("=" * 70)

    chunks_path = processed_dir / "chunks.json"
    if not chunks_path.exists():
        _result("chunks.json exists", "FAIL", f"Not found at {chunks_path}")
        return

    data = _load_json(chunks_path)
    chunks = data if isinstance(data, list) else [data]

    _result(f"chunks.json loaded ({len(chunks)} chunks)", "PASS")

    # Size distribution
    sizes = []
    empty_chunks = []
    fragment_chunks = []  # no ending punctuation, very short
    no_punct_end = []
    required_keys = {"text", "source_url", "source_title", "section", "chunk_index"}
    missing_keys_issues = []

    for i, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        wc = _word_count(text)
        cc = _char_count(text)
        sizes.append((wc, cc))

        if not text.strip():
            empty_chunks.append(f"[{i}]")
        elif cc < 50:
            fragment_chunks.append(f"[{i}]: '{text[:50]}'")

        # Check for ending punctuation (rough heuristic for fragments)
        if text.strip() and text.strip()[-1] not in ".!?»:\"');]" and wc < 20:
            no_punct_end.append(f"[{i}]: '{text[:60]}'")

        # Required keys
        chunk_keys = set(chunk.keys())
        if not required_keys.issubset(chunk_keys):
            missing = required_keys - chunk_keys
            missing_keys_issues.append(f"[{i}] missing: {missing}")

    if sizes:
        word_counts = [s[0] for s in sizes]
        char_counts = [s[1] for s in sizes]
        word_counts_sorted = sorted(word_counts)
        median_wc = word_counts_sorted[len(word_counts_sorted) // 2]
        avg_wc = sum(word_counts) / len(word_counts)

        print(f"\n  📊 Chunk size distribution:")
        print(f"          Words:  min={min(word_counts)}, max={max(word_counts)}, "
              f"median={median_wc}, avg={avg_wc:.1f}")
        print(f"          Chars:  min={min(char_counts)}, max={max(char_counts)}, "
              f"avg={sum(char_counts)/len(char_counts):.0f}")

        # Distribution histogram (buckets)
        buckets = [0, 10, 30, 50, 100, 200, 500, 1000, 99999]
        bucket_labels = ["<10w", "10-30w", "30-50w", "50-100w", "100-200w",
                         "200-500w", "500-1000w", ">1000w"]
        bucket_counts = [0] * len(bucket_labels)
        for wc in word_counts:
            for j in range(len(buckets) - 1):
                if buckets[j] <= wc < buckets[j + 1]:
                    bucket_counts[j] += 1
                    break
        print(f"          Histogram:")
        for label, count in zip(bucket_labels, bucket_counts):
            bar = "█" * min(40, count // max(1, max(bucket_counts) // 40 + 1))
            print(f"            {label:12s} {count:5d} {bar}")

        # Target range check
        target_min, target_max = 50, 500
        out_of_range = sum(1 for wc in word_counts if wc < target_min or wc > target_max)
        if out_of_range / len(word_counts) > 0.3:
            _result(
                f"chunk sizes in target range ({target_min}-{target_max} words)",
                "WARN",
                f"{out_of_range}/{len(word_counts)} ({out_of_range/len(word_counts)*100:.1f}%) out of range",
            )
        else:
            _result(
                f"chunk sizes in target range ({target_min}-{target_max} words)",
                "PASS",
                f"{out_of_range}/{len(word_counts)} out of range",
            )

    _result(
        "no empty chunks",
        "FAIL" if empty_chunks else "PASS",
        issues=empty_chunks or None,
    )

    _result(
        "no fragment chunks (<50 chars)",
        "WARN" if fragment_chunks else "PASS",
        f"{len(fragment_chunks)} fragments" if fragment_chunks else "",
        issues=fragment_chunks or None,
    )

    _result(
        "all chunks have required keys",
        "FAIL" if missing_keys_issues else "PASS",
        issues=missing_keys_issues or None,
    )

    # Duplicate chunks (exact text match)
    text_counts = collections.Counter(chunk.get("text", "") for chunk in chunks)
    exact_dupes = {t: c for t, c in text_counts.items() if c > 1 and t.strip()}
    if exact_dupes:
        total_dupe_chunks = sum(c - 1 for c in exact_dupes.values())
        sample = list(exact_dupes.keys())[:5]
        _result(
            "no exact-duplicate chunks",
            "WARN",
            f"{total_dupe_chunks} duplicate chunks across {len(exact_dupes)} unique texts",
            issues=[f"'{t[:60]}...' ×{c}" for t, c in list(exact_dupes.items())[:5]],
        )
    else:
        _result("no exact-duplicate chunks", "PASS")

    # Chunks per source distribution
    source_counts = collections.Counter(
        chunk.get("source_url", "?") for chunk in chunks
    )
    print(f"\n  📊 Chunks per source (top 10):")
    for url, count in source_counts.most_common(10):
        print(f"          {count:5d}  {url[:60]}")

    return len(chunks)


# ---------------------------------------------------------------------------
# 4. Qdrant index checks
# ---------------------------------------------------------------------------

def check_qdrant(qdrant_dir: Path, chunk_count: int):
    print("\n" + "=" * 70)
    print("4. QDRANT INDEX CHECKS")
    print("=" * 70)

    if not qdrant_dir.exists():
        _result("Qdrant directory exists", "FAIL", f"Not found at {qdrant_dir}")
        return

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        _result("qdrant-client import", "FAIL", "pip install qdrant-client")
        return

    try:
        client = QdrantClient(path=str(qdrant_dir))
        info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
        points_count = info.points_count or 0
        status = info.status
        client.close()
    except Exception as e:
        _result("Qdrant collection accessible", "FAIL", str(e))
        return

    _result(
        f"Qdrant collection '{config.QDRANT_COLLECTION}' accessible",
        "PASS",
        f"status={status}, points={points_count}",
    )

    # Vector count matches chunk count
    if chunk_count > 0:
        if points_count == chunk_count:
            _result(
                "vector count == chunk count",
                "PASS",
                f"{points_count} == {chunk_count}",
            )
        else:
            _result(
                "vector count == chunk count",
                "FAIL",
                f"MISMATCH: qdrant={points_count}, chunks={chunk_count}",
            )

    # Sample vectors for null/zero check
    try:
        client = QdrantClient(path=str(qdrant_dir))
        sample = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=100,
            with_vectors=True,
            with_payload=False,
        )
        null_vectors = 0
        zero_vectors = 0
        dim_set = set()
        for point in sample[0]:
            vec = point.vector
            if vec is None:
                null_vectors += 1
                continue
            dim_set.add(len(vec))
            if all(v == 0.0 for v in vec):
                zero_vectors += 1
        client.close()

        _result(
            f"sampled vectors have correct dimension ({config.QDRANT_VECTOR_SIZE})",
            "PASS" if config.QDRANT_VECTOR_SIZE in dim_set else "FAIL",
            f"dimensions found: {dim_set}",
        )

        _result(
            "no null/zero vectors (sample of 100)",
            "PASS" if null_vectors == 0 and zero_vectors == 0 else "FAIL",
            f"null={null_vectors}, zero={zero_vectors}",
        )
    except Exception as e:
        _result("vector sampling", "WARN", f"Could not sample vectors: {e}")


# ---------------------------------------------------------------------------
# 5. Semantic checks (optional, requires model loading)
# ---------------------------------------------------------------------------

def check_semantic():
    print("\n" + "=" * 70)
    print("5. SEMANTIC CHECKS")
    print("=" * 70)

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from src.embeddings.indexer import search_qdrant
    except ImportError as e:
        _result("semantic dependencies", "FAIL", f"Import error: {e}")
        return

    print("  Loading embedding model (this may take a moment)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Test queries with expected keywords
    test_queries = [
        ("адрес главного корпуса", ["москва", "стромынк", "адрес", "корпус"]),
        ("проходной балл", ["балл", "проходн", "конкурс"]),
        ("какие институты есть", ["институт", "факультет"]),
        ("стипендия", ["стипенди", "выплат", "социальн"]),
        ("общежитие", ["общежит", "проживан"]),
    ]

    low_score_count = 0
    for query, keywords in test_queries:
        prefixed = f"query: {query}"
        vec = model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=False)
        vec = vec.astype(np.float32)
        results = search_qdrant(vec, top_k=3)

        if not results:
            _result(f"query '{query}' returns results", "FAIL", "No results")
            low_score_count += 1
            continue

        top_score = results[0]["score"]
        top_text = results[0]["payload"].get("text", "")[:80]

        hit = any(
            kw.lower() in results[0]["payload"].get("text", "").lower()
            for kw in keywords
        )

        if top_score < 0.3:
            low_score_count += 1
            _result(
                f"query '{query}' (score={top_score:.3f})",
                "WARN",
                f"low score, top: '{top_text}...'",
            )
        elif hit:
            _result(
                f"query '{query}' (score={top_score:.3f})",
                "PASS",
                f"keyword hit, top: '{top_text}...'",
            )
        else:
            _result(
                f"query '{query}' (score={top_score:.3f})",
                "WARN",
                f"no keyword hit in top-1, top: '{top_text}...'",
            )

    # Near-duplicate detection: sample N chunks, check pairwise cosine
    print("\n  Checking for near-duplicate chunks...")
    chunks_path = Path(config.DATA_PROCESSED_DIR) / "chunks.json"
    if chunks_path.exists():
        chunks = _load_json(chunks_path)
        import random
        sample_size = min(500, len(chunks))
        sampled = random.sample(chunks, sample_size) if len(chunks) > sample_size else chunks

        texts = [f"passage: {c['text'][:500]}" for c in sampled]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Compute pairwise cosine similarity (normalized dot product)
        sim_matrix = embeddings @ embeddings.T
        # Zero out diagonal
        np.fill_diagonal(sim_matrix, 0)

        near_dupes = []
        threshold = 0.95
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                if sim_matrix[i, j] > threshold:
                    near_dupes.append(
                        f"cos={sim_matrix[i, j]:.3f}: "
                        f"'{sampled[i]['text'][:40]}...' ≈ "
                        f"'{sampled[j]['text'][:40]}...'"
                    )

        if near_dupes:
            _result(
                f"near-duplicate chunks (cosine > {threshold}, sample {sample_size})",
                "WARN",
                f"{len(near_dupes)} near-duplicate pairs found",
                issues=near_dupes[:10],
            )
        else:
            _result(
                f"near-duplicate chunks (cosine > {threshold}, sample {sample_size})",
                "PASS",
            )

    if low_score_count == 0:
        _result("all semantic queries above threshold", "PASS")
    else:
        _result(
            "all semantic queries above threshold",
            "WARN",
            f"{low_score_count}/{len(test_queries)} queries had score < 0.3",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MIREA RAG data quality verification")
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip embedding-based semantic checks (faster)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MIREA RAG — Data Quality Report")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 70)

    raw_dir = Path(config.DATA_RAW_DIR)
    processed_dir = Path(config.DATA_PROCESSED_DIR)
    qdrant_dir = Path(config.QDRANT_PATH)

    raw_count = check_raw_data(raw_dir)
    cleaned_count = check_cleaned_data(processed_dir, raw_count or 0)
    chunk_count = check_chunks(processed_dir, cleaned_count or 0)
    check_qdrant(qdrant_dir, chunk_count or 0)

    if not args.skip_semantic:
        check_semantic()

    # Summary
    total = CHECKS_PASSED + CHECKS_FAILED + CHECKS_WARNED
    score = CHECKS_PASSED / total * 100 if total > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ✅ PASSED:  {CHECKS_PASSED}")
    print(f"  ❌ FAILED:  {CHECKS_FAILED}")
    print(f"  ⚠️  WARNED:  {CHECKS_WARNED}")
    print(f"  Score: {score:.1f}% ({CHECKS_PASSED}/{total} checks clean)")
    print("=" * 70)

    return 1 if CHECKS_FAILED > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
