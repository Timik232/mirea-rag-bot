# config.py — Project Constants
import os

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHUNK_MERGE_MIN_WORDS = 100
TOP_K_RETRIEVAL = 3
MAX_INPUT_LENGTH = 500
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
FAISS_DIMENSIONS = 384
QA_DATA_PATH = "mirea_data/qa_pairs.json"
DATA_RAW_DIR = "mirea_data/raw"
DATA_PROCESSED_DIR = "mirea_data/processed"
DATA_FAISS_DIR = "mirea_data/faiss"

QDRANT_PATH = "mirea_data/qdrant"
QDRANT_COLLECTION = "mirea_chunks"
QDRANT_VECTOR_SIZE = 384

# --- Retrieval mode -------------------------------------------------------
# "dense"  : Qdrant cosine only (the original production path).
# "hybrid" : BM25 (lexical) + dense fusion — better recall on exact-match
#            queries (codes, numbers, proper nouns) while keeping semantic
#            match. See src/retrieval/hybrid.py.
RETRIEVAL_MODE = "hybrid"
BM25_INDEX_PATH = "mirea_data/bm25/bm25_index.pkl"
HYBRID_FUSION = "rrf"          # "rrf" (rank-based, robust) | "weighted"
HYBRID_RRF_K = 60              # RRF constant (Cormack et al. 2009 default)
HYBRID_DENSE_WEIGHT = 0.5      # dense share; BM25 gets (1 - this). 0.5 = equal
HYBRID_POOL_FACTOR = 3         # over-retrieve pool_factor*top_k per backend
BM25_K1 = 1.5                  # BM25 term-frequency saturation
BM25_B = 0.75                  # BM25 length normalisation

# Cross-encoder re-ranking (two-stage retrieval).
# When RERANK_ENABLED is True, _retrieve first pulls RERANK_CANDIDATES chunks
# from Qdrant (bi-encoder) and then re-sorts them with a cross-encoder, keeping
# the top RERANK_TOP_K. Off by default — re-ranking is a measurable enhancement
# (see mirea_data/eval/rerank_report.json), not a correctness fix.
RERANK_ENABLED = False
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # multilingual (100+ langs, RU incl.)
RERANK_CANDIDATES = 20                      # bi-encoder shortlist size
RERANK_TOP_K = 3                            # chunks kept after re-ranking

# Hallucination guard — confidence threshold (task t_737c1b5e).
# After retrieval, the best chunk's score is checked against this floor. If the
# top-1 score falls below it, the pipeline refuses to generate and returns
# LOW_CONFIDENCE_ANSWER instead of risking a hallucination on weak context.
#
# Calibrated from the golden-set score distribution: bi-encoder (E5-small)
# cosine scores for the 50 in-domain questions span [0.84, 0.95] (mean 0.90,
# std 0.03). A threshold of 0.55 sits well below that range (0% false refusals
# on the golden set) yet above the noise floor for genuinely out-of-domain
# queries (unrelated Russian text pairs typically score 0.30–0.60 on E5-small).
#
# When RERANK_ENABLED, the cross-encoder score (already in [0, 1] —
# sentence-transformers applies sigmoid for bge-reranker-v2-m3) is used as
# the confidence signal instead — it has a far wider dynamic range ([0.001,
# 1.0] on the golden set) and separates relevant from irrelevant chunks
# cleanly.
CONFIDENCE_THRESHOLD = 0.55
CE_CONFIDENCE_THRESHOLD = 0.5

# Query normalisation stages (task t_bd93c958). Each stage is independent and
# can be toggled here so it is benchmarked in isolation on the golden set.
# See src/rag/query_normalization.py for the implementation and rationale.
NORMALIZE_WHITESPACE = True      # collapse multiple spaces / nbsp / newlines
NORMALIZE_PUNCTUATION = True     # "???" -> "?", "!!!" -> "!"
NORMALIZE_KEYBOARD_LAYOUT = True # fix Russian typed on English QWERTY layout
NORMALIZE_CASE = False           # lowercase — off: e5 is case-aware, proper
                                 # nouns (МИРЭА, institute names) carry signal

# Request logging (SQLite)
REQUEST_LOG_DB = "mirea_data"

_litellm_api_key = os.environ.get("LITELLM_API_KEY", "")

LITELLM_CONFIG = {
    "model": os.environ.get("LITELLM_MODEL", "openai/mistral/magistral-small-latest"),
    "api_base": os.environ.get("LITELLM_API_BASE", ""),
    "api_key": _litellm_api_key,
    "temperature": 0.3,
    "max_tokens": 2048,
}

# LM Studio — local Gemma 4 E4B for answer generation (OpenAI-compatible API)
# Download GGUF (Q4_K_M ~5GB) from lmstudio-community/gemma-4-e4b-it-GGUF
# Start server in LM Studio: Developer → Start Server (port 1234)
LM_STUDIO_CONFIG = {
    "api_base": os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1"),
    "api_key": "lm-studio",  # LM Studio accepts any non-empty key
    "model": "google/gemma-4-e2b",  # identifier shown in LM Studio
    "temperature": 0.3,
    "max_tokens": 1024,
    "context_length": 8192,  # fits 8GB VRAM budget
}
