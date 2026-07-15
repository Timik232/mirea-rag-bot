"""Pipeline step: load cleaned data, split text into overlapping chunks, output chunks JSON."""

import json
import logging
import os
import re
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


_tokenizer = None  # None=unloaded, False=load failed, object=ready


def _get_tokenizer():
    """Lazy-load the E5 tokenizer (weights-free, ~2 MB).

    Uses the same model as the embedding encoder (multilingual-e5-small /
    XLM-R SentencePiece), so token counts match what the embedder actually sees.
    """
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            log.info("Loaded E5 tokenizer for chunking: %s", config.EMBEDDING_MODEL)
        except Exception as e:
            log.warning("Cannot load E5 tokenizer (%s); using word-count fallback", e)
            _tokenizer = False
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """Count real subword tokens via the E5 tokenizer.

    Falls back to word_count * 1.3 if the tokenizer is unavailable.
    """
    tok = _get_tokenizer()
    if tok:
        return len(tok.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def chunk_text(text: str) -> list[str]:
    """Split text into chunks respecting CHUNK_SIZE and CHUNK_OVERLAP."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        # Single sentence exceeds chunk size — emit it alone
        if sent_tokens > config.CHUNK_SIZE:
            # Flush current buffer first
            if current:
                chunks.append(" ".join(current))
                # Carry over overlap from end of previous chunk
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current):
                    st = estimate_tokens(s)
                    if overlap_tokens + st > config.CHUNK_OVERLAP:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += st
                current = list(overlap_sentences)
                current_tokens = overlap_tokens

            # Emit oversized sentence as its own chunk (split mid-sentence)
            words = sentence.split()
            sub_chunks: list[str] = []
            sub_current: list[str] = []
            sub_tokens = 0
            for word in words:
                sub_tokens += estimate_tokens(word)
                if sub_tokens > config.CHUNK_SIZE and sub_current:
                    sub_chunks.append(" ".join(sub_current))
                    # Overlap: keep last few words
                    keep: list[str] = []
                    keep_tokens = 0
                    for w in reversed(sub_current):
                        wt = estimate_tokens(w)
                        if keep_tokens + wt > config.CHUNK_OVERLAP:
                            break
                        keep.insert(0, w)
                        keep_tokens += wt
                    sub_current = list(keep)
                    sub_tokens = keep_tokens
                sub_current.append(word)
            if sub_current:
                sub_chunks.append(" ".join(sub_current))
            chunks.extend(sub_chunks)
            continue

        # Would adding this sentence exceed chunk size?
        if current_tokens + sent_tokens > config.CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            # Carry over overlap from end of previous chunk
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current):
                st = estimate_tokens(s)
                if overlap_tokens + st > config.CHUNK_OVERLAP:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += st
            current = list(overlap_sentences)
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def group_and_merge_blobs(
    items: list[dict],
    min_words: int,
) -> list[dict[str, str]]:
    """Group cleaned items by URL, then greedily merge small same-section blobs.

    Phase 1: concatenate all cleaned items sharing a URL into one blob.
    Phase 2: sort blobs by (section, url) so same-section blobs are adjacent,
             then merge any blob shorter than *min_words* with the next
             blob of the same section.

    Returns a list of blobs, each with: text, url, section, title.
    """
    # Phase 1 — group by URL
    by_url: dict[str, list[dict]] = {}
    for item in items:
        url = item.get("url", "")
        by_url.setdefault(url, []).append(item)

    blobs: list[dict[str, str]] = []
    for url, docs in by_url.items():
        text = " ".join(d.get("text", "").strip() for d in docs if d.get("text", "").strip())
        if not text:
            continue
        blobs.append({
            "text": text,
            "url": url,
            "section": docs[0].get("section", ""),
            "title": docs[0].get("title", ""),
        })

    # Phase 2 — greedy merge of small same-section blobs
    blobs.sort(key=lambda b: (b["section"], b["url"]))
    merged: list[dict[str, str]] = []
    for blob in blobs:
        if (merged
                and merged[-1]["section"] == blob["section"]
                and len(merged[-1]["text"].split()) < min_words):
            merged[-1]["text"] += " " + blob["text"]
        else:
            merged.append(dict(blob))  # shallow copy

    return merged


def main() -> None:
    input_path = os.path.join(config.DATA_PROCESSED_DIR, "cleaned.json")
    output_path = os.path.join(config.DATA_PROCESSED_DIR, "chunks.json")

    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    blobs = group_and_merge_blobs(items, config.CHUNK_MERGE_MIN_WORDS)

    all_chunks: list[dict[str, str | int]] = []
    for blob in blobs:
        raw_chunks = chunk_text(blob["text"])
        for idx, chunk_str in enumerate(raw_chunks):
            chunk_str = chunk_str.strip()
            if not chunk_str:
                continue
            all_chunks.append({
                "text": chunk_str,
                "source_url": blob["url"],
                "source_title": blob["title"],
                "section": blob["section"],
                "chunk_index": idx,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    log.info("Processed %d items → %d blobs → %d chunks", len(items), len(blobs), len(all_chunks))


if __name__ == "__main__":
    main()
