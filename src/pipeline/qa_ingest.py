"""qa_ingest.py — Load and normalize Q&A pairs into the project's standard schema.

Usage:
    python -m src.pipeline.qa_ingest

Output:
    mirea_data/raw/qa_pairs.json
"""

import csv
import json
import logging
import os
import re
import sys

sys.path.insert(0, ".")
import config

logger = logging.getLogger(__name__)

MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 20


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return text


def clean_text(text: str) -> str:
    text = strip_html(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_standard_schema(question: str, answer: str) -> dict[str, str]:
    return {
        "title": question,
        "text": answer,
        "url": "",
        "section": "qa_pairs",
        "subsection": "curated",
    }


def load_json(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of objects")
    pairs: list[tuple[str, str]] = []
    for item in data:
        q = item.get("question") or item.get("title") or ""
        a = item.get("answer") or item.get("text") or ""
        pairs.append((q, a))
    return pairs


def load_csv(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row.get("question", ""), row.get("answer", "")))
    return pairs


def load_tsv(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        first_row = next(reader, None)
        if first_row is None:
            return pairs
        header_lower = [c.lower().strip() for c in first_row]
        if "question" in header_lower and "answer" in header_lower:
            pass
        else:
            if len(first_row) >= 2 and (first_row[0].strip() or first_row[1].strip()):
                pairs.append((first_row[0].strip(), "\t".join(first_row[1:]).strip()))
        for row in reader:
            if len(row) < 2:
                continue
            question = row[0].strip()
            answer = "\t".join(row[1:]).strip()
            if not question and not answer:
                continue
            pairs.append((question, answer))
    return pairs


def load_qa_pairs(path: str) -> list[tuple[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return load_tsv(path)
    try:
        return load_json(path)
    except (json.JSONDecodeError, ValueError):
        return load_csv(path)


def validate_and_transform(pairs: list[tuple[str, str]]) -> tuple[list[dict[str, str]], int]:
    valid: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    skipped = 0
    for question, answer in pairs:
        q = clean_text(question)
        a = clean_text(answer)
        if not q or not a:
            skipped += 1
            continue
        if len(q) < MIN_QUESTION_LENGTH:
            logger.warning("Skipping short question (%d chars, min %d): %s",
                           len(q), MIN_QUESTION_LENGTH, q[:50])
            skipped += 1
            continue
        if len(a) < MIN_ANSWER_LENGTH:
            logger.warning("Skipping short answer (%d chars, min %d) for: %s",
                           len(a), MIN_ANSWER_LENGTH, q[:50])
            skipped += 1
            continue
        normalized = q.lower().strip()
        if normalized in seen_questions:
            logger.warning("Duplicate question skipped: %s", q[:50])
            skipped += 1
            continue
        seen_questions.add(normalized)
        valid.append(to_standard_schema(q, a))
    return valid, skipped


def main():
    input_path = config.QA_DATA_PATH
    output_dir = config.DATA_RAW_DIR
    output_path = os.path.join(output_dir, "qa_pairs.json")

    os.makedirs(output_dir, exist_ok=True)

    # Load Q&A pairs
    if os.path.exists(input_path):
        raw_pairs = load_qa_pairs(input_path)
        print(f"Loaded data from {input_path}")
    else:
        print(f"Input file '{input_path}' not found — FAILING (no silent demo fallback)")
        sys.exit(1)

    total = len(raw_pairs)

    # Validate and transform
    items, skipped = validate_and_transform(raw_pairs)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Loaded {total} items, skipped {skipped} invalid, saved {len(items)} items")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
