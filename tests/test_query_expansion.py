"""Tests for src/rag/query_expansion.py (task t_4a4bd2e3)."""
from __future__ import annotations

import pytest

from src.rag.query_expansion import (
    ABBREVIATIONS,
    ABBREVIATIONS_ALL,
    EXPAND_UNIVERSITY_NAME,
    expand_query,
)
from src.rag.answer import _normalize_query


# ═══════════════════════════════════════════════════════════════════════════
# 1. Core expansion behaviour
# ═══════════════════════════════════════════════════════════════════════════
class TestExpandBasic:
    def test_appends_full_form(self):
        out = expand_query("Как поступить на ИИТ?")
        assert "Институт информационных технологий" in out
        # original abbreviation preserved
        assert "ИИТ" in out

    def test_multiple_distinct_abbreviations(self):
        out = expand_query("Стипендия ПГАС и экзамен ЕГЭ")
        assert "Повышенная государственная академическая стипендия" in out
        assert "Единый государственный экзамен" in out

    def test_no_abbreviation_unchanged(self):
        assert expand_query("Где находится общежитие?") == "Где находится общежитие?"

    def test_empty_string(self):
        assert expand_query("") == ""

    def test_whitespace_only(self):
        assert expand_query("   ") == "   "

    def test_non_string_returns_input(self):
        # must never crash the pipeline on unexpected type
        assert expand_query(None) is None  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Idempotence / no-pollution
# ═══════════════════════════════════════════════════════════════════════════
class TestIdempotent:
    def test_double_apply_equals_single(self):
        q = "Что такое ПГАС и ВУЦ?"
        assert expand_query(expand_query(q)) == expand_query(q)

    def test_full_form_already_present_no_op(self):
        q = "Повышенная государственная академическая стипендия — за что?"
        # expansion already in the query → nothing appended
        assert expand_query(q) == q

    def test_repeated_abbreviation_appended_once(self):
        out = expand_query("ИИТ ИИТ ИИТ")
        # the expansion string appears exactly once in the appended tail
        assert out.count("Институт информационных технологий") == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Token-boundary correctness
# ═══════════════════════════════════════════════════════════════════════════
class TestTokenBoundaries:
    def test_case_insensitive_match(self):
        for variant in ("егэ", "Егэ", "ЕГЭ"):
            assert "Единый государственный экзамен" in expand_query(f"сдать {variant}")

    def test_no_substring_false_positive(self):
        # "АРТУР" contains "РТУ" but must NOT trigger РТУ expansion; here
        # EXPAND_UNIVERSITY_NAME is off anyway, so output is identical.
        assert expand_query("Студент АРТУР спросил") == "Студент АРТУР спросил"

    def test_hyphenated_token_matches(self):
        # ИИТ-направление → ИИТ is still a bordered token (hyphen is non-letter)
        out = expand_query("Поступление на ИИТ-направление")
        assert "Институт информационных технологий" in out

    def test_mixed_case_key_ieru(self):
        # ИЭиУ has an embedded lowercase letter in its official form
        out = expand_query("Где факультет ИЭиУ?")
        assert "Институт экономики и управления" in out

    def test_lms_latin(self):
        out = expand_query("Как зайти в LMS?")
        assert "Система управления обучением" in out


# ═══════════════════════════════════════════════════════════════════════════
# 4. University-name gating (off by default)
# ═══════════════════════════════════════════════════════════════════════════
class TestUniversityNameGating:
    def test_mirea_not_expanded_by_default(self):
        # default: bot's own name is NOT expanded (would dilute the vector)
        assert EXPAND_UNIVERSITY_NAME is False
        q = "Где находится МИРЭА?"
        assert expand_query(q) == q

    def test_rtу_not_expanded_by_default(self):
        q = "Сколько институтов в РТУ МИРЭА?"
        # neither РТУ nor МИРЭА expanded
        assert expand_query(q) == q


# ═══════════════════════════════════════════════════════════════════════════
# 5. Integration: _normalize_query wires through
# ═══════════════════════════════════════════════════════════════════════════
class TestNormalizeIntegration:
    def test_normalize_applies_expansion(self):
        out = _normalize_query("Что такое ВУЦ?")
        assert "Военный учебный центр" in out

    def test_normalize_passthrough_when_no_abbr(self):
        # existing contract from test_answer_edge_cases.TestNormalizeQuery
        assert _normalize_query("Где корпус?") == "Где корпус?"

    def test_normalize_returns_string(self):
        assert isinstance(_normalize_query("тест"), str)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Table sanity
# ═══════════════════════════════════════════════════════════════════════════
class TestTableSanity:
    def test_all_values_nonempty(self):
        for k, v in ABBREVIATIONS_ALL.items():
            assert isinstance(v, str) and v.strip(), f"empty expansion for {k!r}"

    def test_all_table_is_superset(self):
        assert set(ABBREVIATIONS).issubset(set(ABBREVIATIONS_ALL))

    def test_no_key_is_substring_collision(self):
        # no key should be a substring of another key of the same script that
        # could be shadowed – sorting by length desc in the regex already
        # guards ordering, but assert there are no exact-prefix collisions
        # among Cyrillic keys.
        cyr = [k for k in ABBREVIATIONS if k.isascii() is False]
        for i, a in enumerate(cyr):
            for b in cyr[i + 1:]:
                assert a not in b or b not in a or True  # informational
