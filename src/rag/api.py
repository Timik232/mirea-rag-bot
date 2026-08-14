#!/usr/bin/env python3
"""FastAPI server for the MIREA RAG bot.

Endpoints:
    POST /ask        — {"question": str} → {"answer": str, "sources": [...], ...}
    GET  /health     — service health check
    GET  /analytics  — request statistics (query param: days=7)

Usage:
    uvicorn src.rag.api:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from starlette.concurrency import run_in_threadpool

import config
from src.rag.answer import answer_question, _get_embedding_model
from src.rag.request_log import log_request, get_analytics

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="МИРЭА RAG Bot",
    description="API для ответов на вопросы о РТУ МИРЭА",
    version="1.0.0",
)

# Track startup time
_startup_time = time.time()
_model_ready = False


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Вопрос о МИРЭА на русском языке")
    history: list[dict] = Field(default_factory=list,
                                description="История диалога: [{'role':'user','content':...},{'role':'assistant','content':...}]")


class Source(BaseModel):
    url: str = ""
    section: str = ""
    title: str = ""
    score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    context_chunks: int = 0
    retrieval_scores: list[float] = []
    generation_time: float = 0.0
    confidence: float = 0.0


@app.on_event("startup")
async def startup_event():
    """Pre-load embedding model on startup."""
    global _model_ready
    logger.info("Pre-loading embedding model...")
    try:
        _get_embedding_model()
        _model_ready = True
        logger.info("Embedding model ready")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)


@app.get("/health")
async def health():
    """Service health check."""
    # Check Qdrant
    qdrant_ok = False
    qdrant_points = 0
    try:
        from src.embeddings.qdrant_indexer import QdrantIndexer
        qi = QdrantIndexer()
        info = qi.get_collection_info(config.QDRANT_PATH, config.QDRANT_COLLECTION)
        qdrant_ok = info["status"] in ("green", "yellow")
        qdrant_points = info.get("points_count", 0)
    except Exception:
        pass

    # Check LM Studio
    lm_studio_ok = False
    try:
        import requests
        resp = requests.get(
            f"{config.LM_STUDIO_CONFIG['api_base']}/models",
            timeout=3,
        )
        lm_studio_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if (_model_ready and qdrant_ok) else "degraded",
        "model_loaded": _model_ready,
        "qdrant": {"ok": qdrant_ok, "points": qdrant_points},
        "lm_studio": {"ok": lm_studio_ok, "url": config.LM_STUDIO_CONFIG["api_base"]},
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Answer a question about MIREA using RAG. Every request is logged."""
    try:
        # answer_question is fully synchronous (E5 encode + Qdrant search +
        # up to 3 LM Studio HTTP calls with 60s timeouts each). Running it on
        # the event loop would freeze ALL concurrent requests (including
        # /health) for the duration of generation — offload to the threadpool.
        result = await run_in_threadpool(answer_question, req.question, history=req.history)
        # Confidence = top retrieval score (heuristic)
        scores = result.get("retrieval_scores", [])
        confidence = scores[0] if scores else 0.0

        response = AskResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result.get("sources", [])],
            context_chunks=result.get("context_chunks", 0),
            retrieval_scores=scores,
            generation_time=result.get("generation_time", 0),
            confidence=round(confidence, 4),
        )

        # Log the request off the event loop too (synchronous SQLite write;
        # errors suppressed inside log_request)
        await run_in_threadpool(
            log_request,
            question=req.question,
            answer=result["answer"],
            sources=result.get("sources", []),
            context_chunks=result.get("context_chunks", 0),
            retrieval_scores=scores,
            generation_time=result.get("generation_time", 0),
            confidence=round(confidence, 4),
        )

        return response
    except ConnectionError as e:
        # Log the error too
        log_request(question=req.question, error=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Error answering question: %s", e, exc_info=True)
        log_request(question=req.question, error=str(e)[:500])
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}")


@app.get("/analytics")
async def analytics(days: int = Query(default=7, ge=1, le=365)):
    """Request statistics for the last N days.

    Returns total requests, avg generation time, avg confidence,
    top questions, low-confidence queries, hourly/daily volume.
    """
    return get_analytics(days=days)


@app.get("/")
async def chat_ui():
    """Serve the chat UI."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(404, "UI not found")
