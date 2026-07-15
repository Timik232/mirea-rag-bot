import os
import re
import sys
import time

import litellm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def clean_html(html: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", html)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&quot;", '"')]:
        cleaned = cleaned.replace(entity, char)
    # Strip markdown artifacts
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\*{3}([^*]+)\*{3}", r"\1", cleaned)
    cleaned = re.sub(r"_{3}([^_]+)_{3}", r"\1", cleaned)
    cleaned = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", cleaned)
    cleaned = re.sub(r"_{2}([^_]+)_{2}", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*\n]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"^\|.*\|$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\|[-:| ]+\|$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"~~([^~]+)~~", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def llm_summarize(text: str, config: dict = None) -> str:
    cfg = config
    if cfg is None:
        from config import LITELLM_CONFIG
        cfg = LITELLM_CONFIG
    try:
        response = litellm.completion(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "Clean and summarize the following text. Return only the cleaned text, no commentary."},
                {"role": "user", "content": text},
            ],
            api_base=cfg["api_base"],
            api_key=cfg["api_key"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        result = response.choices[0].message.content.strip()
        if not result:
            return clean_html(text)
        return clean_html(result)
    except Exception:
        return clean_html(text)


def clean_item(item: dict, use_llm: bool = True) -> dict:
    raw_text = item.get("content") or item.get("text", "")
    cleaned_text = clean_html(raw_text)
    if use_llm:
        cleaned_text = llm_summarize(cleaned_text)
    cleaned_title = clean_html(item.get("title", ""))
    result = {k: v for k, v in item.items() if k not in ("text", "content", "title")}
    result["title"] = cleaned_title
    result["text"] = cleaned_text
    return result


def clean_batch(items: list[dict], use_llm: bool = True) -> list[dict]:
    results = []
    for i, item in enumerate(items):
        results.append(clean_item(item, use_llm=use_llm))
        if use_llm and i < len(items) - 1:
            time.sleep(0.3)
    return results
