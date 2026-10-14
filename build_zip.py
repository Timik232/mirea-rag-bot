"""Build a clean distributable zip of the mirea-rag-bot project."""
import zipfile, os, sys

EXCLUDE_DIRS = {
    "models", "__pycache__", ".pytest_cache", ".deepeval", ".git",
    ".venv", "venv", "node_modules",
    # stale/poisoned artifacts — not for distribution
    "qdrant_old_75chunks", "mirea_data_backup_20261006",
}
EXCLUDE_FILES = {"mirea-rag-bot.zip", "build_zip.py", "_verify_layout.py", "requests.db"}
# Eval artifacts computed on the OLD (pre-2026-10, partially poisoned) corpus.
# Misleading for consumers; keep only reports regenerated after the rebuild
# (retrieval/hybrid/rerank/golden_qa, October 2026).
EXCLUDE_STALE_EVAL = {
    "mirea_data/eval/ab_answers_A.json",
    "mirea_data/eval/ab_answers_B.json",
    "mirea_data/eval/ab_answers_C.json",
    "mirea_data/eval/ab_answers_D.json",
    "mirea_data/eval/ab_report.json",
    "mirea_data/eval/ab_report_bd.json",
    "mirea_data/eval/ab_run.log",
    "mirea_data/eval/build_vika_audit.py",
    "mirea_data/eval/confidence_guard_report.json",
    "mirea_data/eval/judge_answers.json",
    "mirea_data/eval/judge_report.json",
    "mirea_data/eval/multihop_baseline_lmstudio.json",
    "mirea_data/eval/multihop_test.json",
    "mirea_data/eval/vika_audit.json",
    "mirea_data/vika_chunks.json",
    "mirea_data/vika_embeddings.npy",
    "benchmark_results.json",
}
EXCLUDE_EXT = {".pyc", ".pyo"}


def should_skip(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    for p in parts:
        if p in EXCLUDE_DIRS or p.endswith(".egg-info"):
            return True
    base = os.path.basename(rel_path)
    if base in EXCLUDE_FILES:
        return True
    if rel_path.replace("\\", "/") in EXCLUDE_STALE_EVAL:
        return True
    if os.path.splitext(rel_path)[1] in EXCLUDE_EXT:
        return True
    return False


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out = "mirea-rag-bot.zip"
    if os.path.exists(out):
        os.remove(out)
    count = 0
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if not should_skip(os.path.join(root, d))]
            for f in files:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, ".")
                if should_skip(rel):
                    continue
                z.write(fp, rel)
                count += 1
                total += os.path.getsize(fp)
    arch = os.path.getsize(out)
    print("files: %d" % count)
    print("uncompressed: %.1f MB" % (total / 1024 / 1024))
    print("archive: %.1f MB" % (arch / 1024 / 1024))


if __name__ == "__main__":
    main()
