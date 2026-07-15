"""Pipeline step: load raw data, clean text, deduplicate, output cleaned JSON."""

import hashlib
import json
import logging
import os
import re
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.scraper._binary_guard import has_binary_content, _MAX_TEXT_LENGTH

_RAW_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _RAW_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _RAW_PROJECT_ROOT)

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_FILES = [
    "education.json",
    "applicants.json",
    "sveden.json",
    "contacts.json",
    "qa_pairs.json",
    "deeper.json",
    "student_portal.json",
]

BOILERPLATE_PHRASES = [
    "Главная",
    "Карта сайта",
    "Версия для слабовидящих",
    "©",
    "Все права защищены",
    "Перейти к контенту",
    "Меню",
]

YEAR_PATTERN = re.compile(r"\b(20[0-2]\d)\b")
MIN_TEXT_LENGTH = 20

# Titles that are navigation labels rather than content descriptions
BOILERPLATE_TITLES = {
    "главная", "home", "menu", "меню", "навигация", "карта сайта",
}

# Patterns to strip from subsection to derive a title
_SUBSECTION_PREFIX_RE = re.compile(
    r"^(?:arrow_forward\s*|Читать далее\s*)+",
    re.UNICODE,
)
_SUBSECTION_SUFFIX_RE = re.compile(
    r"\s*\([^)]*\)\s*$",
    re.UNICODE,
)


def generate_title(item: dict, cleaned_text: str) -> str:
    """Derive a meaningful title when the raw title is empty or boilerplate.

    Strategy:
    1. Clean subsection — strip 'arrow_forward', 'Читать далее', '(подразделы)'.
    2. If that yields something usable, return it.
    3. Otherwise, take the first sentence/heading from the cleaned text.
    4. Fallback: use the section name.
    """
    # Try subsection first
    raw_sub = item.get("subsection", "")
    sub = clean_text(raw_sub)
    sub = _SUBSECTION_PREFIX_RE.sub("", sub).strip()
    sub = _SUBSECTION_SUFFIX_RE.sub("", sub).strip()
    if len(sub) >= 3 and sub.lower() not in ("главная",):
        return sub[:120]

    # Extract first meaningful phrase from text (up to first period or newline)
    first_line = cleaned_text.split("\n")[0] if cleaned_text else ""
    # Trim to first sentence (generous limit for long Russian phrases)
    m = re.match(r"(.{5,120}?)(?:[.!?\n]|$)", first_line)
    if m:
        return m.group(1).strip()[:120]

    # Fallback: first ~8 words of text
    words = first_line.split()
    if len(words) >= 3:
        return " ".join(words[:8]).strip()[:120]

    # Fallback to section
    section = item.get("section", "")
    return section if section else "Без названия"


def strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&quot;", '"')]:
        cleaned = cleaned.replace(entity, char)
    return cleaned


def strip_markdown(text: str) -> str:
    """Remove markdown formatting artifacts: bold, italic, headings, links,
    images, tables, blockquotes, horizontal rules, code blocks."""
    cleaned = text
    # Headings: ### Title -> Title
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # Images: ![alt](url) -> alt
    cleaned = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", cleaned)
    # Links: [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    # Bold+italic: ***text*** or ___text___ -> text
    cleaned = re.sub(r"\*{3}([^*]+)\*{3}", r"\1", cleaned)
    cleaned = re.sub(r"_{3}([^_]+)_{3}", r"\1", cleaned)
    # Bold: **text** or __text__ -> text
    cleaned = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", cleaned)
    cleaned = re.sub(r"_{2}([^_]+)_{2}", r"\1", cleaned)
    # Italic: *text* or _text_ -> text
    cleaned = re.sub(r"\*([^*\n]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", cleaned)
    # Inline code: `code` -> code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Code blocks: ```...```
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    # Tables: | col | col | and |---|---|
    cleaned = re.sub(r"^\|.*\|$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\|[-:| ]+\|$", "", cleaned, flags=re.MULTILINE)
    # Horizontal rules: ---, ***, ___
    cleaned = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", cleaned, flags=re.MULTILINE)
    # Blockquotes: > text -> text
    cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)
    # Strikethrough: ~~text~~ -> text
    cleaned = re.sub(r"~~([^~]+)~~", r"\1", cleaned)
    return cleaned


def remove_boilerplate(text: str) -> str:
    removed = text
    for phrase in BOILERPLATE_PHRASES:
        removed = removed.replace(phrase, " ")
    return removed


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    return normalize_whitespace(remove_boilerplate(strip_markdown(strip_html(text))))


def compute_content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_years(text: str) -> list[str]:
    return sorted(set(YEAR_PATTERN.findall(text)))


def load_raw_data(raw_dir: str) -> list[dict]:
    items: list[dict] = []
    for filename in RAW_FILES:
        filepath = os.path.join(raw_dir, filename)
        if not os.path.isfile(filepath):
            log.warning("File not found, skipping: %s", filepath)
            continue
        with open(filepath, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                log.warning("Failed to parse %s: %s", filepath, exc)
                continue
        if isinstance(data, list):
            items.extend(data)
        else:
            items.append(data)
        log.info("Loaded %d items from %s", len(data) if isinstance(data, list) else 1, filename)
    return items


def process_items(items: list[dict]) -> tuple[list[dict], int, int]:
    seen_hashes: set[str] = set()
    cleaned: list[dict] = []
    duplicates = 0
    skipped = 0
    binary_dropped = 0

    for item in items:
        raw_text = item.get("text", "")

        # Binary content guard — drop garbage before it enters the pipeline
        if has_binary_content(raw_text):
            binary_dropped += 1
            log.warning("Dropping binary content from %s (section=%s)",
                        item.get("url", "?"), item.get("section", "?"))
            continue
        if len(raw_text) > _MAX_TEXT_LENGTH:
            binary_dropped += 1
            log.warning("Dropping oversized text (%d chars) from %s",
                        len(raw_text), item.get("url", "?"))
            continue

        cleaned_text = clean_text(raw_text)

        if len(cleaned_text) < MIN_TEXT_LENGTH:
            skipped += 1
            continue

        h = compute_content_hash(cleaned_text)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)

        cleaned_title = clean_text(item.get("title", ""))

        # Fill in empty or boilerplate titles from subsection / text context
        if not cleaned_title.strip() or cleaned_title.lower().strip() in BOILERPLATE_TITLES:
            cleaned_title = generate_title(item, cleaned_text)

        cleaned.append({
            "title": cleaned_title,
            "text": cleaned_text,
            "url": item.get("url", ""),
            "section": item.get("section", ""),
            "subsection": item.get("subsection", ""),
            "content_hash": h,
            "year": extract_years(cleaned_text + " " + cleaned_title),
        })

    return cleaned, duplicates, skipped, binary_dropped


def run():
    raw_dir = config.DATA_RAW_DIR
    processed_dir = config.DATA_PROCESSED_DIR

    os.makedirs(processed_dir, exist_ok=True)

    items = load_raw_data(raw_dir)
    loaded_count = len(items)
    log.info("Total items loaded: %d", loaded_count)

    cleaned, duplicates, skipped, binary_dropped = process_items(items)

    output_path = os.path.join(processed_dir, "cleaned.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, ensure_ascii=False, indent=2)

    log.info(
        "Summary: Loaded %d, removed %d duplicates, %d skipped (< %d chars), "
        "%d binary dropped, output %d items → %s",
        loaded_count, duplicates, skipped, MIN_TEXT_LENGTH,
        binary_dropped, len(cleaned), output_path,
    )
    return cleaned


if __name__ == "__main__":
    run()
