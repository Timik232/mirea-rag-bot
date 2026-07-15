"""Tests for src.rag.request_log — SQLite request logging + analytics."""

import json
import os
import sys
import sqlite3

import pytest

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from src.rag import request_log  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    """Redirect the DB to a temp file so tests don't touch production data."""
    db_path = str(tmp_path / "test_requests.db")
    # Patch module-level constant
    monkeypatch.setattr(request_log, "DB_PATH", db_path)
    monkeypatch.setattr(config, "REQUEST_LOG_DB", str(tmp_path))
    yield db_path
    # Cleanup — tmp_path handles removal automatically


class TestLogRequest:
    def test_basic_log(self):
        row_id = request_log.log_request(
            question="Где библиотека?",
            answer="Библиотека находится в главном корпусе.",
            sources=[{"url": "https://mirea.ru/lib", "score": 0.85}],
            context_chunks=3,
            retrieval_scores=[0.85, 0.72, 0.60],
            generation_time=1.23,
            confidence=0.85,
        )
        assert row_id > 0

        # Verify it's in the DB
        conn = sqlite3.connect(request_log.DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM requests_log WHERE id = ?", (row_id,)).fetchone()
        conn.close()

        assert row is not None
        assert row["question"] == "Где библиотека?"
        assert "Библиотека" in row["answer"]
        assert row["context_chunks"] == 3
        assert row["confidence"] == 0.85

    def test_error_log(self):
        row_id = request_log.log_request(
            question="test error",
            error="Connection refused",
        )
        assert row_id > 0

        conn = sqlite3.connect(request_log.DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM requests_log WHERE id = ?", (row_id,)).fetchone()
        conn.close()

        assert row["error"] == "Connection refused"
        assert row["answer"] == ""

    def test_long_answer_truncated(self):
        long_answer = "x" * 1000
        row_id = request_log.log_request(question="q", answer=long_answer)
        conn = sqlite3.connect(request_log.DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM requests_log WHERE id = ?", (row_id,)).fetchone()
        conn.close()
        assert len(row["answer"]) <= 500

    def test_sources_serialized(self):
        sources = [{"url": "http://a", "score": 0.9}]
        request_log.log_request(question="q", sources=sources)
        conn = sqlite3.connect(request_log.DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM requests_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        parsed = json.loads(row["sources"])
        assert len(parsed) == 1
        assert parsed[0]["url"] == "http://a"


class TestGetAnalytics:
    def _seed_data(self):
        """Insert a few rows for testing analytics."""
        now_iso = request_log.datetime.utcnow().isoformat() + "Z"
        # Ensure table exists (log_request normally does this, but we insert directly)
        request_log._get_conn().close()
        conn = sqlite3.connect(request_log.DB_PATH)
        rows = [
            (now_iso, "Где библиотека?", "В корпусе А", "[]", 3, "[0.85]", 1.2, 0.85, None),
            (now_iso, "Расписание занятий", "На сайте", "[]", 2, "[0.70]", 0.9, 0.70, None),
            (now_iso, "Где библиотека?", "В корпусе А", "[]", 3, "[0.82]", 1.1, 0.82, None),
            (now_iso, "Как поступить?", "Нет данных", "[]", 0, "[]", 0.5, 0.0, None),
            (now_iso, "Ошибка тест", "", "[]", 0, "[]", 0, 0, "some error"),
        ]
        conn.executemany(
            """INSERT INTO requests_log
               (timestamp, question, answer, sources, context_chunks,
                retrieval_scores, generation_time, confidence, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        conn.close()

    def test_empty_analytics(self):
        result = request_log.get_analytics(days=7)
        assert result["total_requests"] == 0
        assert result["top_questions"] == []
        assert result["avg_generation_time"] == 0

    def test_populated_analytics(self):
        self._seed_data()
        result = request_log.get_analytics(days=7)

        assert result["total_requests"] == 5
        assert result["error_count"] == 1
        assert result["avg_generation_time"] > 0
        assert result["period_days"] == 7

        # Top question: "Где библиотека?" appears 2x
        top = result["top_questions"]
        assert len(top) >= 1
        assert top[0]["count"] == 2
        assert "библиотека" in top[0]["question"].lower()

        # Low confidence: "Как поступить?" (0.0) + "Ошибка тест" (0.0)
        low = result["low_confidence"]
        assert len(low) >= 1

    def test_custom_days(self):
        self._seed_data()
        result = request_log.get_analytics(days=1)
        assert result["total_requests"] == 5
        assert result["period_days"] == 1

    def test_recent_requests_limited(self):
        self._seed_data()
        result = request_log.get_analytics(days=7)
        assert len(result["recent_requests"]) <= 20


class TestClearLogs:
    def test_clear_old(self):
        now = request_log.datetime.utcnow()
        old_ts = (now - request_log.timedelta(days=100)).isoformat() + "Z"
        new_ts = now.isoformat() + "Z"

        # Ensure table exists
        request_log._get_conn().close()

        conn = sqlite3.connect(request_log.DB_PATH)
        conn.execute(
            "INSERT INTO requests_log (timestamp, question) VALUES (?, ?)",
            (old_ts, "old question"),
        )
        conn.execute(
            "INSERT INTO requests_log (timestamp, question) VALUES (?, ?)",
            (new_ts, "new question"),
        )
        conn.commit()
        conn.close()

        deleted = request_log.clear_logs(older_than_days=90)
        assert deleted == 1

        conn = sqlite3.connect(request_log.DB_PATH)
        remaining = conn.execute("SELECT COUNT(*) FROM requests_log").fetchone()[0]
        conn.close()
        assert remaining == 1

    def test_clear_none(self):
        deleted = request_log.clear_logs(older_than_days=90)
        assert deleted == 0
