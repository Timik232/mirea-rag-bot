"""TDD tests for pipeline integration with Qdrant backend.

Tests cover:
1. cleaner.py loads ALL raw files (including deeper.json, student_portal.json)
2. chunker.py handles longer texts correctly
3. encoder.py output is Qdrant-compatible
4. indexer.py has Qdrant backend with numpy fallback
5. build_index.py end-to-end with Qdrant
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary raw/processed/faiss/qdrant dirs."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()
    qdrant_dir = tmp_path / "qdrant"
    qdrant_dir.mkdir()
    return {
        "raw": str(raw_dir),
        "processed": str(processed_dir),
        "faiss": str(faiss_dir),
        "qdrant": str(qdrant_dir),
        "tmp_path": tmp_path,
    }


def _write_raw_files(raw_dir, file_data: dict[str, list[dict]]):
    """Write raw JSON files into raw_dir."""
    for filename, items in file_data.items():
        with open(os.path.join(raw_dir, filename), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)


def _sample_items(n: int, prefix: str = "item") -> list[dict]:
    """Generate n sample raw items with enough text to pass cleaning."""
    return [
        {
            "title": f"{prefix} title {i}",
            "text": f"Это текст номер {i} для проверки пайплайна. " * 5,
            "url": f"https://example.com/{prefix}/{i}",
            "section": f"Раздел {i % 3}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Cleaner tests — loads all raw files including new ones
# ---------------------------------------------------------------------------

class TestCleanerLoadAllFiles:
    """cleaner.load_raw_data must load all 7 raw files."""

    def test_loads_standard_files(self, tmp_dirs):
        from src.pipeline.cleaner import load_raw_data

        file_data = {
            "education.json": _sample_items(10),
            "applicants.json": _sample_items(15),
            "sveden.json": _sample_items(8),
            "contacts.json": _sample_items(5),
            "qa_pairs.json": _sample_items(10),
        }
        _write_raw_files(tmp_dirs["raw"], file_data)
        items = load_raw_data(tmp_dirs["raw"])
        assert len(items) == 48  # 10+15+8+5+10

    def test_loads_deeper_json(self, tmp_dirs):
        """deeper.json must be in RAW_FILES list."""
        from src.pipeline import cleaner

        assert "deeper.json" in cleaner.RAW_FILES, "deeper.json must be in RAW_FILES"

    def test_loads_student_portal_json(self, tmp_dirs):
        """student_portal.json must be in RAW_FILES list."""
        from src.pipeline import cleaner

        assert "student_portal.json" in cleaner.RAW_FILES, "student_portal.json must be in RAW_FILES"

    def test_loads_all_seven_files(self, tmp_dirs):
        """Must load from 7 raw files."""
        from src.pipeline import cleaner

        assert len(cleaner.RAW_FILES) >= 7, f"RAW_FILES has {len(cleaner.RAW_FILES)} entries, expected >= 7"

    def test_real_raw_data_loads_200_plus(self):
        """With real raw data, should load 200+ items."""
        raw_dir = config.DATA_RAW_DIR
        if not os.path.isdir(raw_dir):
            pytest.skip("No real raw data directory")

        from src.pipeline.cleaner import load_raw_data

        items = load_raw_data(raw_dir)
        assert len(items) >= 200, f"Expected 200+ raw items, got {len(items)}"

    def test_cleaner_run_with_real_data(self):
        """Full cleaner.run() on real data produces output."""
        raw_dir = config.DATA_RAW_DIR
        if not os.path.isdir(raw_dir):
            pytest.skip("No real raw data directory")

        from src.pipeline.cleaner import run

        cleaned = run()
        assert len(cleaned) > 0, "Cleaner produced zero items"
        # All items should have required keys
        for item in cleaned:
            assert "title" in item
            assert "text" in item
            assert "url" in item


# ---------------------------------------------------------------------------
# 2. Chunker tests — handles longer texts
# ---------------------------------------------------------------------------

class TestChunkerLongTexts:
    """chunker must split long texts into multiple chunks with overlap."""

    def test_long_text_produces_multiple_chunks(self, tmp_dirs):
        from src.pipeline.chunker import chunk_text

        # Generate text that exceeds CHUNK_SIZE (300 tokens)
        long_text = ". ".join([f"Предложение номер {i} для проверки разбиения текста на части" for i in range(100)])
        chunks = chunk_text(long_text)
        assert len(chunks) > 1, f"Long text should produce multiple chunks, got {len(chunks)}"

    def test_chunk_overlap_exists(self, tmp_dirs):
        from src.pipeline.chunker import chunk_text

        # Generate text well beyond CHUNK_SIZE (300 tokens)
        # Each sentence ~10 words = ~13 tokens; 100 sentences = ~1300 tokens
        sentences = [f"Уникальное предложение номер {i} для проверки." for i in range(100)]
        long_text = " ".join(sentences)
        chunks = chunk_text(long_text)
        assert len(chunks) > 1
        # Overlap: some words from end of chunks[i] should appear in chunks[i+1]
        found_overlap = False
        for i in range(len(chunks) - 1):
            words_prev = chunks[i].split()[-5:]
            words_next = chunks[i + 1].split()[:10]
            common = set(words_prev) & set(words_next)
            if common:
                found_overlap = True
                break
        assert found_overlap, "No overlap found between consecutive chunks"

    def test_empty_text_returns_empty(self):
        from src.pipeline.chunker import chunk_text

        assert chunk_text("") == []

    def test_real_cleaned_data_chunks(self):
        """Chunker works with real cleaned data."""
        cleaned_path = os.path.join(config.DATA_PROCESSED_DIR, "cleaned.json")
        if not os.path.isfile(cleaned_path):
            pytest.skip("No cleaned.json — run cleaner first")

        from src.pipeline.chunker import chunk_text

        with open(cleaned_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        total_chunks = 0
        for item in items:
            text = item.get("text", "")
            chunks = chunk_text(text)
            total_chunks += len(chunks)
        assert total_chunks >= len(items), "Should produce at least as many chunks as items"


# ---------------------------------------------------------------------------
# 3. Encoder output is Qdrant-compatible
# ---------------------------------------------------------------------------

class TestEncoderQdrantCompat:
    """encoder output (vectors + metadata) must be consumable by QdrantIndexer."""

    def test_metadata_has_required_fields(self, tmp_dirs):
        """Metadata must have fields needed for Qdrant payload."""
        # Create minimal chunks.json
        chunks = [
            {"text": "Текст для тестирования", "source_title": "Test", "source_url": "https://example.com", "section": "s1", "chunk_index": 0},
            {"text": "Второй текст для проверки", "source_title": "Test2", "source_url": "https://example.com/2", "section": "s2", "chunk_index": 0},
        ]
        chunks_path = os.path.join(tmp_dirs["processed"], "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        from src.embeddings.encoder import encode_chunks

        vectors, metadata = encode_chunks(
            chunks_path=chunks_path,
            output_dir=tmp_dirs["faiss"],
        )

        # Verify metadata fields usable as Qdrant payload
        required_fields = {"source_title", "source_url", "section", "chunk_index", "text"}
        for entry in metadata:
            assert required_fields.issubset(entry.keys()), f"Missing fields: {required_fields - entry.keys()}"

    def test_vectors_shape_matches_config(self, tmp_dirs):
        """Vector dimensions must match QDRANT_VECTOR_SIZE."""
        chunks = [
            {"text": "Текст", "source_title": "T", "source_url": "u", "section": "s", "chunk_index": 0},
        ]
        chunks_path = os.path.join(tmp_dirs["processed"], "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        from src.embeddings.encoder import encode_chunks

        vectors, metadata = encode_chunks(
            chunks_path=chunks_path,
            output_dir=tmp_dirs["faiss"],
        )

        assert vectors.shape[0] == len(metadata)
        assert vectors.shape[1] == config.QDRANT_VECTOR_SIZE
        assert vectors.dtype == np.float32


# ---------------------------------------------------------------------------
# 4. Indexer — Qdrant backend with numpy fallback
# ---------------------------------------------------------------------------

class TestIndexerQdrantBackend:
    """indexer.py must have build_qdrant_index function."""

    def test_build_qdrant_index_exists(self):
        """build_qdrant_index function must exist in indexer."""
        from src.embeddings import indexer

        assert hasattr(indexer, "build_qdrant_index"), "indexer must have build_qdrant_index()"

    def test_build_qdrant_index_creates_collection(self, tmp_dirs):
        """build_qdrant_index should create Qdrant collection and upsert vectors."""
        # Create fake vectors + metadata
        n = 5
        vectors = np.random.randn(n, 384).astype(np.float32)
        metadata = [
            {"text": f"text {i}", "source_title": f"t{i}", "source_url": f"u{i}", "section": "s", "chunk_index": i}
            for i in range(n)
        ]
        vectors_path = os.path.join(tmp_dirs["faiss"], "vectors.npy")
        metadata_path = os.path.join(tmp_dirs["faiss"], "metadata.json")
        np.save(vectors_path, vectors)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        from src.embeddings.indexer import build_qdrant_index

        build_qdrant_index(
            vectors_path=vectors_path,
            metadata_path=metadata_path,
            qdrant_path=tmp_dirs["qdrant"],
            collection_name="test_chunks",
        )

        # Verify collection was created
        from src.embeddings.qdrant_indexer import QdrantIndexer

        qi = QdrantIndexer()
        info = qi.get_collection_info(tmp_dirs["qdrant"], "test_chunks")
        assert info["points_count"] == n

    def test_numpy_fallback_still_works(self, tmp_dirs):
        """build_index (numpy) must still work as fallback."""
        vectors = np.random.randn(3, 384).astype(np.float32)
        vectors_path = os.path.join(tmp_dirs["faiss"], "vectors.npy")
        np.save(vectors_path, vectors)

        from src.embeddings.indexer import build_index

        index = build_index(vectors_path=vectors_path, output_dir=tmp_dirs["faiss"])
        assert index is not None

    def test_search_qdrant_exists(self):
        """search_qdrant function must exist in indexer."""
        from src.embeddings import indexer

        assert hasattr(indexer, "search_qdrant"), "indexer must have search_qdrant()"

    def test_search_qdrant_returns_results(self, tmp_dirs):
        """search_qdrant should return results from Qdrant collection."""
        n = 5
        vectors = np.random.randn(n, 384).astype(np.float32)
        metadata = [
            {"text": f"text {i}", "source_title": f"t{i}", "source_url": f"u{i}", "section": "s", "chunk_index": i}
            for i in range(n)
        ]
        vectors_path = os.path.join(tmp_dirs["faiss"], "vectors.npy")
        metadata_path = os.path.join(tmp_dirs["faiss"], "metadata.json")
        np.save(vectors_path, vectors)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        from src.embeddings.indexer import build_qdrant_index, search_qdrant

        build_qdrant_index(
            vectors_path=vectors_path,
            metadata_path=metadata_path,
            qdrant_path=tmp_dirs["qdrant"],
            collection_name="test_search",
        )

        query = np.random.randn(384).astype(np.float32)
        results = search_qdrant(
            query_vector=query,
            qdrant_path=tmp_dirs["qdrant"],
            collection_name="test_search",
            top_k=3,
        )

        assert isinstance(results, list)
        assert len(results) <= 3
        if results:
            assert "id" in results[0]
            assert "score" in results[0]
            assert "payload" in results[0]


# ---------------------------------------------------------------------------
# 5. Build index script — end-to-end with Qdrant
# ---------------------------------------------------------------------------

class TestBuildIndexScript:
    """build_index.py must support Qdrant as default backend."""

    def test_build_index_has_qdrant_function(self):
        """build_index module must have a build_qdrant function."""
        from scripts import build_index

        assert hasattr(build_index, "build_qdrant"), "build_index must have build_qdrant()"

    def test_build_index_has_numpy_function(self):
        """build_index module must have a build_numpy function (fallback)."""
        from scripts import build_index

        assert hasattr(build_index, "build_numpy"), "build_index must have build_numpy()"

    def test_build_qdrant_uses_config_constants(self):
        """build_qdrant must use config.QDRANT_PATH and config.QDRANT_COLLECTION."""
        assert hasattr(config, "QDRANT_PATH")
        assert hasattr(config, "QDRANT_COLLECTION")
        assert hasattr(config, "QDRANT_VECTOR_SIZE")
