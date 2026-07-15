"""Tests for Qdrant config constants."""
import config


def test_config_has_qdrant_path():
    assert hasattr(config, "QDRANT_PATH")
    assert isinstance(config.QDRANT_PATH, str)
    assert config.QDRANT_PATH == "mirea_data/qdrant"


def test_config_has_qdrant_collection():
    assert hasattr(config, "QDRANT_COLLECTION")
    assert isinstance(config.QDRANT_COLLECTION, str)
    assert config.QDRANT_COLLECTION == "mirea_chunks"


def test_config_has_qdrant_vector_size():
    assert hasattr(config, "QDRANT_VECTOR_SIZE")
    assert isinstance(config.QDRANT_VECTOR_SIZE, int)
    assert config.QDRANT_VECTOR_SIZE == 384


def test_existing_constants_unchanged():
    """Existing constants must remain exactly as they were."""
    assert config.CHUNK_SIZE == 300
    assert config.CHUNK_OVERLAP == 50
    assert config.TOP_K_RETRIEVAL == 3
    assert config.MAX_INPUT_LENGTH == 500
    assert config.EMBEDDING_MODEL == "intfloat/multilingual-e5-small"
    assert config.FAISS_DIMENSIONS == 384
    assert config.QA_DATA_PATH == "mirea_data/qa_pairs.json"
    assert config.DATA_RAW_DIR == "mirea_data/raw"
    assert config.DATA_PROCESSED_DIR == "mirea_data/processed"
    assert config.DATA_FAISS_DIR == "mirea_data/faiss"
