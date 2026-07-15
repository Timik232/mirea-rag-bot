"""Build a clean distributable zip of the mirea-rag-bot project."""
import zipfile, os, sys

EXCLUDE_DIRS = {
    "models", "__pycache__", ".pytest_cache", ".deepeval", ".git",
    ".venv", "venv", "node_modules",
}
EXCLUDE_FILES = {"mirea-rag-bot.zip", "build_zip.py", "_verify_layout.py"}
EXCLUDE_EXT = {".pyc", ".pyo"}


def should_skip(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    for p in parts:
        if p in EXCLUDE_DIRS or p.endswith(".egg-info"):
            return True
    base = os.path.basename(rel_path)
    if base in EXCLUDE_FILES:
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
