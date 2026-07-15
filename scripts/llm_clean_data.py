#!/usr/bin/env python3
"""LLM-clean all raw JSON data files.

- Backs up originals to mirea_data/raw_backup/ (skipped if backup exists)
- Runs clean_batch(use_llm=True) on each file — handles HTML cleaning + LLM summarization with fallback
- Verifies no HTML tags remain in text/content fields after cleaning
"""

import json
import os
import re
import shutil
import sys
import time

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.pipeline.llm_cleaner import clean_batch

RAW_DIR = os.path.join(PROJECT_ROOT, "mirea_data", "raw")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "mirea_data", "raw_backup")


def create_backup():
    """Copy all JSON files from raw/ to raw_backup/. Skip if backup already has files."""
    if os.path.isdir(BACKUP_DIR) and os.listdir(BACKUP_DIR):
        print(f"[SKIP] Backup already exists at {BACKUP_DIR}")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".json"):
            shutil.copy2(os.path.join(RAW_DIR, fname), os.path.join(BACKUP_DIR, fname))
    print(f"[BACKUP] Copied raw JSON files to {BACKUP_DIR}")


def clean_all_files():
    """Process each JSON file: load, clean_batch, save back."""
    json_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))
    if not json_files:
        print("[ERROR] No JSON files found in", RAW_DIR)
        return

    print(f"\n=== Cleaning {len(json_files)} files ===\n")
    for fname in json_files:
        fpath = os.path.join(RAW_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                items = json.load(f)

            if not isinstance(items, list):
                print(f"  [{fname}] SKIP — not a list ({type(items).__name__})")
                continue

            print(f"  [{fname}] {len(items)} items — cleaning...", end=" ", flush=True)
            cleaned = clean_batch(items, use_llm=True)

            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)

            print(f"OK ({len(cleaned)} items saved)")
        except Exception as e:
            print(f"FAIL — {e}")
            continue

        time.sleep(0.5)

    print("\n=== Cleaning complete ===\n")


def verify():
    """Check that no HTML tags remain in text/content fields (URL fields excluded)."""
    html_pattern = re.compile(r"<[a-zA-Z/][^>]*>")
    issues = []

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            items = json.load(f)
        for i, item in enumerate(items):
            for key in ("text", "content", "title", "section", "subsection"):
                val = item.get(key, "")
                if not isinstance(val, str):
                    continue
                if html_pattern.search(val):
                    issues.append(f"  {fname}[{i}].{key}: found HTML tag")

    if issues:
        print(f"[VERIFY] FAIL — {len(issues)} HTML tags found:")
        for issue in issues:
            print(issue)
    else:
        print("[VERIFY] PASS — no HTML tags in content fields")


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] Raw data directory not found: {RAW_DIR}")
        sys.exit(1)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DIR}")
    print(f"Config model: {config.LITELLM_CONFIG['model']}")
    print(f"Config api:   {config.LITELLM_CONFIG['api_base']}")

    create_backup()
    clean_all_files()
    verify()


if __name__ == "__main__":
    main()
