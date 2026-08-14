"""Re-run the entire data pipeline: scrape → ingest QA → clean → chunk → encode → Qdrant upsert.

Usage:
    python scripts/update_data.py          # full pipeline
    python scripts/update_data.py --dry-run # preview without modifying data
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add project root so config imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402

SCRAPER_MODULES = [
    ("education", "src.scraper.education"),
    ("applicants", "src.scraper.applicants"),
    ("sveden", "src.scraper.sveden"),
    ("contacts", "src.scraper.contacts"),
    ("deeper", "src.scraper.deeper_scraper"),
    ("student_portal", "src.scraper.student_portal"),
]

PIPELINE_MODULES = [
    ("qa_ingest", "src.pipeline.qa_ingest"),
    ("cleaner", "src.pipeline.cleaner"),
    ("chunker", "src.pipeline.chunker"),
]



def _count_json_lines_or_items(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 1
    except Exception:
        return 0


def _run_module(module_name: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [DRY-RUN] would run: python -m {module_name}")
        return True
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print(f"  FAILED: {module_name} timed out after 3600s")
        return False
    if result.returncode != 0:
        print(f"  FAILED: {module_name} exited with code {result.returncode}")
        return False
    return True


def _current_point_count() -> int | None:
    """Points in the live Qdrant collection, or None if unavailable."""
    try:
        from src.embeddings.qdrant_indexer import QdrantIndexer

        info = QdrantIndexer().get_collection_info(config.QDRANT_PATH, config.QDRANT_COLLECTION)
        return int(info.get("points_count", 0))
    except Exception as exc:  # noqa: BLE001 — informational only
        print(f"  (could not read current Qdrant point count: {exc})")
        return None


def _backup_qdrant(qdrant_dir: str, dry_run: bool):
    """Move the live Qdrant storage to a SIBLING backup directory.

    The backup must live OUTSIDE the live path: _build_qdrant_atomic replaces
    the whole qdrant directory, so a backup stored inside it (the old
    ``qdrant/backup/`` scheme) was destroyed by the very rebuild it was
    supposed to protect.

    Returns the backup directory path, or None when there was nothing to back up.
    """
    qdrant_path = Path(qdrant_dir)
    if not qdrant_path.exists():
        print("  No existing Qdrant data to back up.")
        return None
    backup_dir = qdrant_path.parent / (qdrant_path.name + "_backup")
    if dry_run:
        print(f"  [DRY-RUN] would back up {qdrant_dir} → {backup_dir}")
        return backup_dir
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.move(str(qdrant_path), str(backup_dir))
    print(f"  Backed up existing Qdrant data to {backup_dir}")
    return backup_dir


def _build_qdrant_atomic(dry_run: bool) -> tuple[int, int, int] | None:
    """Encode chunks and build Qdrant index into a temp dir, then atomically move.

    Returns (n_items, n_chunks, n_vectors) or None on failure.
    """
    from src.embeddings.encoder import encode_chunks
    from src.embeddings.indexer import build_qdrant_index

    n_items = _count_json_lines_or_items(os.path.join(config.DATA_PROCESSED_DIR, "cleaned.json"))
    n_chunks = _count_json_lines_or_items(os.path.join(config.DATA_PROCESSED_DIR, "chunks.json"))

    if dry_run:
        print(f"  [DRY-RUN] would encode {n_chunks} chunks and upsert to Qdrant collection '{config.QDRANT_COLLECTION}'")
        return n_items, n_chunks, 0

    # Encode into a temp dir, build Qdrant index there, then atomic swap
    with tempfile.TemporaryDirectory(prefix="qdrant_build_", dir=os.path.dirname(config.QDRANT_PATH)) as tmp_dir:
        print(f"  Encoding {n_chunks} chunks into temp dir {tmp_dir} ...")
        chunks_path = os.path.join(config.DATA_PROCESSED_DIR, "chunks.json")
        embeddings, metadata = encode_chunks(chunks_path=chunks_path, output_dir=tmp_dir)
        n_vectors = embeddings.shape[0]

        print(f"  Building Qdrant index with {n_vectors} vectors ...")
        vectors_path = os.path.join(tmp_dir, "vectors.npy")
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        info = build_qdrant_index(
            vectors_path=vectors_path,
            metadata_path=metadata_path,
            qdrant_path=tmp_dir,
            collection_name=config.QDRANT_COLLECTION,
            vector_size=config.QDRANT_VECTOR_SIZE,
        )

        # Atomic swap: replace final Qdrant dir with temp dir contents.
        # The previous storage was moved away by _backup_qdrant (sibling
        # directory), so a failure here leaves BOTH the old backup and this
        # temp build intact — no total loss window.
        target = Path(config.QDRANT_PATH)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(tmp_dir, target)
        # Keep the dense-metadata sidecar next to the vectors in sync: the
        # hybrid retriever reads payload fallbacks and BM25 source data from
        # DATA_FAISS_DIR/metadata.json. Without this copy it stays stale
        # after every rebuild (hybrid then mixes a new corpus with an old
        # positional id contract).
        faiss_meta = Path(config.DATA_FAISS_DIR) / "metadata.json"
        faiss_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata_path, faiss_meta)
        print(f"  Qdrant index atomically moved to {config.QDRANT_PATH}")
        print(f"  metadata.json refreshed at {faiss_meta}")

    return n_items, n_chunks, n_vectors


def main():
    parser = argparse.ArgumentParser(description="Re-run the full data pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing")
    args = parser.parse_args()

    dry_run = args.dry_run
    print("=" * 60)
    print("MIREA Data Pipeline Update")
    if dry_run:
        print("*** DRY RUN — no data will be modified ***")
    print("=" * 60)

    step = 0

    step += 1
    print(f"\n--- Step {step}: Scraping ---")
    for name, module in SCRAPER_MODULES:
        print(f"  [{name}]")
        if not _run_module(module, dry_run):
            print(f"\nABORTED at scraper '{name}'. Existing data not modified.")
            sys.exit(1)
        time.sleep(0.5)

    raw_dir = config.DATA_RAW_DIR
    total_raw = 0
    for fname in ("education.json", "applicants.json", "sveden.json", "contacts.json",
                  "deeper.json", "student_portal.json"):
        fpath = os.path.join(raw_dir, fname)
        cnt = _count_json_lines_or_items(fpath)
        print(f"  {fname}: {cnt} items")
        total_raw += cnt

    step += 1
    print(f"\n--- Step {step}: Q&A Ingestion ---")
    for name, module in PIPELINE_MODULES[:1]:
        print(f"  [{name}]")
        if not _run_module(module, dry_run):
            print(f"\nABORTED at '{name}'. Existing data not modified.")
            sys.exit(1)
    qa_count = _count_json_lines_or_items(os.path.join(raw_dir, "qa_pairs.json"))
    print(f"  qa_pairs.json: {qa_count} pairs")

    step += 1
    print(f"\n--- Step {step}: Cleaning ---")
    for name, module in PIPELINE_MODULES[1:2]:
        print(f"  [{name}]")
        if not _run_module(module, dry_run):
            print(f"\nABORTED at '{name}'. Existing data not modified.")
            sys.exit(1)

    step += 1
    print(f"\n--- Step {step}: Chunking ---")
    for name, module in PIPELINE_MODULES[2:]:
        print(f"  [{name}]")
        if not _run_module(module, dry_run):
            print(f"\nABORTED at '{name}'. Existing data not modified.")
            sys.exit(1)

    step += 1
    print(f"\n--- Step {step}: Build Qdrant Index ---")
    old_points = _current_point_count()
    if old_points:
        print(f"  Current collection holds {old_points} points.")
    backup_dir = _backup_qdrant(config.QDRANT_PATH, dry_run)

    try:
        result = _build_qdrant_atomic(dry_run)
    except Exception as exc:
        print(f"\nABORTED during Qdrant index build: {exc}")
        result = None
    if result is None:
        if backup_dir:
            print(f"  Backup preserved at {backup_dir} — restore it manually if needed.")
        sys.exit(1)

    n_items, n_chunks, n_vectors = result

    if not dry_run and old_points and n_vectors < old_points:
        print(f"\n  WARNING: point count dropped ({old_points} → {n_vectors}).")
        print("  This rebuild indexes chunks.json ONLY — any extra collections")
        print("  ingested separately (e.g. VIKA QA chunks) are NOT included and")
        print(f"  must be re-ingested. The previous data is in {backup_dir}.")

    step += 1
    print(f"\n--- Step {step}: Rebuild BM25 Index ---")
    # BM25 shares the positional doc-id contract with metadata.json/Qdrant;
    # after a corpus rebuild it MUST be rebuilt or hybrid retrieval joins a
    # new corpus against a stale lexical index.
    bm25_script = os.path.join(os.path.dirname(__file__), "build_bm25.py")
    if dry_run:
        print(f"  [DRY-RUN] would run: python {bm25_script}")
    else:
        try:
            r = subprocess.run(
                [sys.executable, bm25_script],
                cwd=os.path.join(os.path.dirname(__file__), ".."),
                timeout=1800,
            )
            if r.returncode != 0:
                print(f"  FAILED: build_bm25.py exited with code {r.returncode}")
                print("  Hybrid retrieval will fall back to dense-only until fixed.")
        except subprocess.TimeoutExpired:
            print("  FAILED: build_bm25.py timed out")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"  Updated {total_raw} scraped items + {qa_count} QA pairs")
    print(f"  Produced {n_chunks} chunks from {n_items} cleaned items")
    print(f"  Indexed {n_vectors} vectors in Qdrant")
    if dry_run:
        print("  (dry run — nothing was actually changed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
