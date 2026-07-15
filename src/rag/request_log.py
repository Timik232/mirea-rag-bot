"""Request logging and analytics for the RAG bot.

Persists every /ask request to a SQLite database (mirea_data/requests.db).
Provides query helpers for the /analytics endpoint.

Schema:
    requests_log:
        id          INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp   TEXT        (ISO 8601)
        question    TEXT
        answer      TEXT        (truncated to 500 chars)
        sources     TEXT        (JSON array)
        context_chunks  INTEGER
        retrieval_scores TEXT   (JSON array of floats)
        generation_time  REAL
        confidence  REAL
        error       TEXT        (NULL if success)

Analytics queries:
    - total_requests, avg_generation_time, avg_confidence
    - top_questions (most frequent)
    - question_volume over time (hourly / daily)
    - low-confidence queries (potential gaps)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(_PROJECT_ROOT, getattr(config, "REQUEST_LOG_DB", "mirea_data") + "/requests.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT,
    sources         TEXT,
    context_chunks  INTEGER DEFAULT 0,
    retrieval_scores TEXT,
    generation_time  REAL DEFAULT 0,
    confidence      REAL DEFAULT 0,
    error           TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests_log(timestamp);
"""


def _get_conn() -> sqlite3.Connection:
    """Return a connection to the requests DB, ensuring schema exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX)
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def log_request(
    question: str,
    answer: str = "",
    sources: Optional[list] = None,
    context_chunks: int = 0,
    retrieval_scores: Optional[list] = None,
    generation_time: float = 0,
    confidence: float = 0,
    error: Optional[str] = None,
) -> int:
    """Insert a request log entry. Returns the row id."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            """INSERT INTO requests_log
               (timestamp, question, answer, sources, context_chunks,
                retrieval_scores, generation_time, confidence, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat() + "Z",
                question[:2000],
                answer[:500] if answer else "",
                json.dumps(sources or [], ensure_ascii=False),
                context_chunks,
                json.dumps(retrieval_scores or []),
                generation_time,
                confidence,
                error,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        logger.debug("Logged request #%d: %s", row_id, question[:80])
        return row_id
    except Exception as e:
        logger.error("Failed to log request: %s", e)
        return -1


def get_analytics(days: int = 7) -> dict:
    """Compute analytics for the last N days.

    Returns a dict with:
        - total_requests
        - error_count
        - avg_generation_time
        - avg_confidence
        - top_questions: [{question, count}]
        - recent_requests: [{id, timestamp, question, confidence, generation_time}]
        - low_confidence: [{id, question, confidence}]  (confidence < 0.4)
        - hourly_volume: {hour: count}  (last 24h)
        - daily_volume: {date: count}  (last N days)
    """
    conn = _get_conn()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

    rows = conn.execute(
        "SELECT * FROM requests_log WHERE timestamp >= ? ORDER BY timestamp DESC",
        (since,),
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "total_requests": 0,
            "error_count": 0,
            "avg_generation_time": 0,
            "avg_confidence": 0,
            "top_questions": [],
            "recent_requests": [],
            "low_confidence": [],
            "hourly_volume": {},
            "daily_volume": {},
            "period_days": days,
        }

    total = len(rows)
    errors = sum(1 for r in rows if r["error"])
    gen_times = [r["generation_time"] for r in rows if r["generation_time"] and r["generation_time"] > 0]
    confs = [r["confidence"] for r in rows if r["confidence"] is not None and r["confidence"] > 0]

    # Top questions (normalized)
    q_counts: dict[str, int] = {}
    for r in rows:
        q = r["question"].strip().lower()
        q_counts[q] = q_counts.get(q, 0) + 1
    top_qs = sorted(q_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    # Group same question back to original case (use most recent occurrence)
    seen: dict[str, str] = {}
    for r in rows:
        norm = r["question"].strip().lower()
        if norm not in seen:
            seen[norm] = r["question"]
    top_questions = [
        {"question": seen[q], "count": c} for q, c in top_qs
    ]

    # Recent 20
    recent = [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "question": r["question"],
            "confidence": r["confidence"],
            "generation_time": r["generation_time"],
        }
        for r in rows[:20]
    ]

    # Low confidence
    low_conf = [
        {"id": r["id"], "question": r["question"], "confidence": r["confidence"]}
        for r in rows
        if r["confidence"] is not None and r["confidence"] < 0.4
    ][:20]

    # Hourly volume (last 24h)
    now = datetime.utcnow()
    hourly: dict[str, int] = {}
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        age = (now - ts).total_seconds()
        if age > 86400:
            continue
        hour_key = ts.strftime("%Y-%m-%dT%H:00")
        hourly[hour_key] = hourly.get(hour_key, 0) + 1

    # Daily volume
    daily: dict[str, int] = {}
    for r in rows:
        try:
            day_key = r["timestamp"][:10]
        except (TypeError, IndexError):
            continue
        daily[day_key] = daily.get(day_key, 0) + 1

    return {
        "total_requests": total,
        "error_count": errors,
        "avg_generation_time": round(statistics.mean(gen_times), 2) if gen_times else 0,
        "avg_confidence": round(statistics.mean(confs), 4) if confs else 0,
        "top_questions": top_questions,
        "recent_requests": recent,
        "low_confidence": low_conf,
        "hourly_volume": hourly,
        "daily_volume": daily,
        "period_days": days,
    }


def clear_logs(older_than_days: int = 90) -> int:
    """Delete log entries older than N days. Returns count deleted."""
    conn = _get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat() + "Z"
    cur = conn.execute("DELETE FROM requests_log WHERE timestamp < ?", (cutoff,))
    count = cur.rowcount
    conn.commit()
    conn.close()
    logger.info("Cleared %d log entries older than %d days", count, older_than_days)
    return count
