"""test_qa_ingest_phase2.py — TDD tests for enhanced qa_ingest (TSV, dedup, min-length)."""

import csv
import json
import os
import tempfile

import pytest

# Ensure project root is on sys.path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.qa_ingest import (
    load_json,
    load_csv,
    load_tsv,
    load_qa_pairs,
    validate_and_transform,
    clean_text,
    to_standard_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, pairs):
    """Write a list of {question, answer} dicts as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)


def _write_csv(path, pairs):
    """Write question,answer CSV with header."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for q, a in pairs:
            w.writerow([q, a])


def _write_tsv(path, pairs):
    """Write question<TAB>answer TSV with header."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["question", "answer"])
        for q, a in pairs:
            w.writerow([q, a])


def _make_pair(q_len=50, a_len=50):
    """Create a Q&A pair with guaranteed min lengths."""
    q = "Q" * q_len
    a = "A" * a_len
    return {"question": q, "answer": a}


# ---------------------------------------------------------------------------
# 1. TSV format support
# ---------------------------------------------------------------------------

class TestTSVFormat:
    """TSV (tab-separated) loading and auto-detection."""

    def test_load_tsv_basic(self, tmp_path):
        """TSV file with header loads correctly."""
        rows = [
            ("Какой институт лучший?", "ИИТ — лучший институт МИРЭА!"),
            ("Где находится МИРЭА?", "Москва, ул. Стромынка, д. 20"),
        ]
        tsv_file = tmp_path / "qa.tsv"
        _write_tsv(str(tsv_file), rows)

        pairs = load_tsv(str(tsv_file))
        assert len(pairs) == 2
        assert pairs[0] == rows[0]
        assert pairs[1] == rows[1]

    def test_load_tsv_empty_lines_skipped(self, tmp_path):
        """Empty rows in TSV are skipped."""
        path = str(tmp_path / "qa.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("question\tanswer\n")
            f.write("Вопрос номер один\tОтвет номер один\n")
            f.write("\n")
            f.write("\t\n")
            f.write("Вопрос номер два\tОтвет номер два\n")

        pairs = load_tsv(path)
        assert len(pairs) == 2

    def test_load_qa_pairs_detects_tsv_by_extension(self, tmp_path):
        """load_qa_pairs auto-detects .tsv extension."""
        rows = [("Вопрос про экзамены", "Ответ про экзамены подробный")]
        tsv_file = tmp_path / "data.tsv"
        _write_tsv(str(tsv_file), rows)

        pairs = load_qa_pairs(str(tsv_file))
        assert len(pairs) == 1
        assert pairs[0][0] == "Вопрос про экзамены"

    def test_load_qa_pairs_tsv_with_tabs_in_answer(self, tmp_path):
        """Tabs within answer text don't break TSV parsing."""
        path = str(tmp_path / "qa.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("question\tanswer\n")
            f.write("Вопрос\tОтвет\tс\tтабами\n")
            # answer = "Ответ\tс\tтабами" — extra tabs are part of answer

        pairs = load_tsv(path)
        assert len(pairs) == 1
        # The answer should include the extra tab-separated parts
        assert "Ответ" in pairs[0][1]

    def test_load_tsv_no_header(self, tmp_path):
        """TSV without header row still works (first row treated as data)."""
        path = str(tmp_path / "qa.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Вопрос без заголовка\tОтвет без заголовка\n")

        # Should still load — even if first row looks like header
        pairs = load_tsv(path)
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# 2. Duplicate question detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    """Duplicate questions are warned and first occurrence is kept."""

    def test_exact_duplicates_removed(self, tmp_path):
        """Exact same question appears twice — only first kept."""
        pairs = [
            ("Какие институты есть в МИРЭА?", "Ответ один про институты"),
            ("Какие институты есть в МИРЭА?", "Ответ два про институты"),
        ]
        _write_json(str(tmp_path / "qa.json"), [
            {"question": q, "answer": a} for q, a in pairs
        ])

        raw = load_json(str(tmp_path / "qa.json"))
        items, skipped = validate_and_transform(raw)
        assert len(items) == 1
        assert items[0]["title"] == "Какие институты есть в МИРЭА?"
        assert items[0]["text"] == "Ответ один про институты"

    def test_case_insensitive_dedup(self):
        """Questions differing only by case are treated as duplicates."""
        pairs = [
            ("Какие институты есть в МИРЭА?", "Первый ответ на этот вопрос"),
            ("КАКИЕ ИНСТИТУТЫ ЕСТЬ В МИРЭА?", "Второй ответ на этот вопрос"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 1
        assert items[0]["text"] == "Первый ответ на этот вопрос"

    def test_whitespace_normalized_dedup(self):
        """Questions differing only in whitespace are duplicates."""
        pairs = [
            ("Какие институты в МИРЭА?", "Первый ответ здесь достаточно"),
            ("  Какие   институты в МИРЭА?  ", "Второй ответ здесь достаточно"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 1
        assert items[0]["text"] == "Первый ответ здесь достаточно"

    def test_three_duplicates_keeps_first(self):
        """Three identical questions → only first kept."""
        pairs = [
            ("Как поступить в МИРЭА?", "Первый ответ подробный"),
            ("Как поступить в МИРЭА?", "Второй ответ подробный"),
            ("Как поступить в МИРЭА?", "Третий ответ подробный"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 1
        assert items[0]["text"] == "Первый ответ подробный"

    def test_no_false_positive_dedup(self):
        """Different questions are NOT deduped."""
        pairs = [
            ("Как поступить в МИРЭА?", "Ответ про поступление"),
            ("Какие специальности в МИРЭА?", "Ответ про специальности"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# 3. Min length validation
# ---------------------------------------------------------------------------

class TestMinLengthValidation:
    """Questions <10 chars or answers <20 chars are skipped."""

    def test_short_question_skipped(self):
        """Question with <10 chars is skipped."""
        pairs = [
            ("Короткий", "Это достаточно длинный ответ для проверки"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 0
        assert skipped == 1

    def test_short_answer_skipped(self):
        """Answer with <20 chars is skipped."""
        pairs = [
            ("Достаточно длинный вопрос?", "Короткий ответ"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 0
        assert skipped == 1

    def test_both_short_skipped(self):
        """Both question and answer too short → skipped."""
        pairs = [
            ("Что?", "Да"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 0
        assert skipped == 1

    def test_exactly_min_lengths_pass(self):
        """Exactly 10 char question and 20 char answer pass."""
        q = "1234567890"  # exactly 10 chars
        a = "12345678901234567890"  # exactly 20 chars
        pairs = [(q, a)]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 1
        assert skipped == 0

    def test_below_min_by_one_fails(self):
        """9 char question fails; 19 char answer fails."""
        pairs = [
            ("123456789", "12345678901234567890"),  # q=9, a=20
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 0

        pairs = [
            ("1234567890", "1234567890123456789"),  # q=10, a=19
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 0

    def test_whitespace_only_question_skipped(self):
        """Whitespace-only question fails min length after cleaning."""
        pairs = [
            ("     ", "Достаточно длинный ответ для пропуска"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 0
        assert skipped == 1


# ---------------------------------------------------------------------------
# 4. Large file handling (200+ pairs)
# ---------------------------------------------------------------------------

class TestLargeFiles:
    """Handles 200+ pairs without issues."""

    def test_200_json_pairs(self, tmp_path):
        """Load and process 200 JSON Q&A pairs."""
        pairs = [{"question": f"Question number {i:04d}?", "answer": f"Answer number {i:04d} is long enough"} for i in range(200)]
        path = str(tmp_path / "large.json")
        _write_json(path, pairs)

        raw = load_json(path)
        assert len(raw) == 200
        items, skipped = validate_and_transform(raw)
        assert len(items) == 200
        assert skipped == 0

    def test_500_tsv_pairs(self, tmp_path):
        """Load and process 500 TSV Q&A pairs."""
        rows = [("Вопрос номер " + str(i), "Ответ номер " + str(i) * 3) for i in range(500)]
        # Ensure min lengths
        rows = [(q.ljust(15, "Q"), a.ljust(25, "A")) for q, a in rows]
        path = str(tmp_path / "large.tsv")
        _write_tsv(path, rows)

        raw = load_tsv(path)
        assert len(raw) == 500
        items, skipped = validate_and_transform(raw)
        assert len(items) == 500
        assert skipped == 0

    def test_200_pairs_with_mixed_invalid(self, tmp_path):
        """200 pairs with some short/duplicate → correct counts."""
        pairs = []
        for i in range(100):
            pairs.append((f"Valid question {i:04d}?", f"Valid answer {i:04d} is long enough"))
        for i in range(50):
            pairs.append((f"Valid question {0:04d}?", f"Duplicate answer {i:04d} is long enough"))
        for i in range(50):
            pairs.append(("short", f"Short Q answer {i:04d} is long enough"))

        items, skipped = validate_and_transform(pairs)
        assert len(items) == 100
        assert skipped == 100


# ---------------------------------------------------------------------------
# 5. Output schema unchanged
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Output schema: {title, text, url, section, subsection}."""

    def test_schema_fields(self):
        """to_standard_schema produces correct fields."""
        result = to_standard_schema("Вопрос?", "Ответ!")
        assert set(result.keys()) == {"title", "text", "url", "section", "subsection"}
        assert result["title"] == "Вопрос?"
        assert result["text"] == "Ответ!"
        assert result["url"] == ""
        assert result["section"] == "qa_pairs"
        assert result["subsection"] == "curated"

    def test_validate_and_transform_schema(self):
        """validate_and_transform produces correct schema."""
        pairs = [
            ("Достаточно длинный вопрос?", "Достаточно длинный ответ!"),
        ]
        items, skipped = validate_and_transform(pairs)
        assert len(items) == 1
        item = items[0]
        assert "title" in item
        assert "text" in item
        assert item["url"] == ""
        assert item["section"] == "qa_pairs"
        assert item["subsection"] == "curated"


# ---------------------------------------------------------------------------
# 6. Existing functionality preserved
# ---------------------------------------------------------------------------

class TestExistingFunctionality:
    """JSON and CSV loading still works; demo data preserved."""

    def test_json_loading(self, tmp_path):
        """JSON loading works as before."""
        data = [
            {"question": "Вопрос один?", "answer": "Ответ один"},
            {"question": "Вопрос два?", "answer": "Ответ два"},
        ]
        path = str(tmp_path / "qa.json")
        _write_json(path, data)

        pairs = load_json(path)
        assert len(pairs) == 2

    def test_csv_loading(self, tmp_path):
        """CSV loading works as before."""
        rows = [
            ("Вопрос один?", "Ответ один"),
            ("Вопрос два?", "Ответ два"),
        ]
        path = str(tmp_path / "qa.csv")
        _write_csv(path, rows)

        pairs = load_csv(path)
        assert len(pairs) == 2

    def test_load_qa_pairs_json_fallback_to_csv(self, tmp_path):
        """If JSON fails, falls back to CSV parsing."""
        path = str(tmp_path / "data.csv")
        _write_csv(path, [("Вопрос про учебу", "Ответ про учебу")])

        pairs = load_qa_pairs(path)
        assert len(pairs) == 1

    def test_clean_text_html_stripping(self):
        """HTML tags are stripped."""
        assert clean_text("<b>Bold</b>") == "Bold"
        assert clean_text("A &amp; B") == "A & B"

    def test_clean_text_whitespace(self):
        """Excess whitespace is normalized."""
        assert clean_text("  hello   world  ") == "hello world"
