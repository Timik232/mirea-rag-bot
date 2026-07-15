"""Tests for src/rag/query_normalization.py (task t_bd93c958).

Each normalisation stage is tested in isolation, then via the orchestrator
(normalize_query), then through the integration hook (_normalize_query in
answer.py). All stages must be no-ops on already-clean Cyrillic text — the
backward-compatibility contract from the parent task's passthrough tests.
"""
from __future__ import annotations

import pytest

from src.rag.query_normalization import (
    _convert_word,
    _looks_like_wrong_layout,
    fix_keyboard_layout,
    fold_punctuation,
    normalize_case,
    normalize_query,
    normalize_whitespace,
)
from src.rag.answer import _normalize_query


# ═══════════════════════════════════════════════════════════════════════════
# 1. normalize_whitespace
# ═══════════════════════════════════════════════════════════════════════════
class TestNormalizeWhitespace:
    def test_single_space_unchanged(self):
        assert normalize_whitespace("Где корпус?") == "Где корпус?"

    def test_leading_trailing_stripped(self):
        assert normalize_whitespace("  Где корпус?  ") == "Где корпус?"

    def test_multiple_spaces_collapsed(self):
        assert normalize_whitespace("Где    корпус?") == "Где корпус?"

    def test_tabs_collapsed(self):
        assert normalize_whitespace("Где\t\tкорпус") == "Где корпус"

    def test_newlines_become_spaces(self):
        assert normalize_whitespace("строка 1\nстрока 2") == "строка 1 строка 2"

    def test_newlines_with_surrounding_whitespace(self):
        assert normalize_whitespace("а\n  б") == "а б"

    def test_nbsp_treated_as_space(self):
        # U+00A0 non-breaking space, common in copy-pasted text
        assert normalize_whitespace("а\u00a0б") == "а б"

    def test_narrow_nbsp(self):
        # U+202F narrow no-break space
        assert normalize_whitespace("а\u202fб") == "а б"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_whitespace_only(self):
        assert normalize_whitespace("   \t\n  ") == ""

    def test_non_string_passthrough(self):
        # must never crash the pipeline
        assert normalize_whitespace(None) is None  # type: ignore[arg-type]
        assert normalize_whitespace(123) == 123    # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 2. fold_punctuation
# ═══════════════════════════════════════════════════════════════════════════
class TestFoldPunctuation:
    def test_single_question_mark_unchanged(self):
        assert fold_punctuation("Где?") == "Где?"

    def test_multiple_question_marks(self):
        assert fold_punctuation("Где???") == "Где?"

    def test_multiple_exclamation_marks(self):
        assert fold_punctuation("Да!!!") == "Да!"

    def test_mixed_q_excl_collapsed_to_first(self):
        # "?!?" → first char "?"
        assert fold_punctuation("Что?!?") == "Что?"

    def test_interrobang_pattern(self):
        assert fold_punctuation("Что?!?!") == "Что?"

    def test_ellipsis_preserved(self):
        # "..." carries meaning — must NOT be collapsed
        assert fold_punctuation("Подожди...") == "Подожди..."

    def test_unicode_ellipsis_preserved(self):
        assert fold_punctuation("Подожди…") == "Подожди…"

    def test_no_terminal_punctuation(self):
        assert fold_punctuation("Где корпус") == "Где корпус"

    def test_empty_string(self):
        assert fold_punctuation("") == ""

    def test_non_string_passthrough(self):
        assert fold_punctuation(None) is None  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 3. fix_keyboard_layout — detection helpers
# ═══════════════════════════════════════════════════════════════════════════
class TestLooksLikeWrongLayout:
    def test_basic_russian_word_in_latin(self):
        # "vjq" → v→м, j→о, q→й → "мой" (has vowel о)
        assert _looks_like_wrong_layout("vjq") is True

    def test_word_with_vowel(self):
        # "afqk" → ф, а, й, л → "файл" (has vowel а)
        assert _looks_like_wrong_layout("afqk") is True

    def test_consonant_only_rejected(self):
        # "rrr" → "ккк" — no vowels, not plausible Russian
        assert _looks_like_wrong_layout("rrr") is False

    def test_known_latin_abbreviation_not_converted(self):
        for token in ("lms", "LMS", "RTU", "rtu", "API", "api"):
            assert _looks_like_wrong_layout(token) is False

    def test_single_char_rejected(self):
        # 1-char words never trigger (too many false positives)
        assert _looks_like_wrong_layout("e") is False
        assert _looks_like_wrong_layout("r") is False


class TestConvertWord:
    def test_simple_word(self):
        # r→к, t→е, y→н → "кен"
        assert _convert_word("rty") == "кен"

    def test_my_word(self):
        # v→м, j→о, q→й → "мой"
        assert _convert_word("vjq") == "мой"

    def test_file_word(self):
        # a→ф, f→а, q→й, k→л → "файл"
        assert _convert_word("afqk") == "файл"

    def test_preserves_unmapped_chars(self):
        # '5' has no mapping → stays as-is
        assert _convert_word("ab5") == "фи5"

    def test_uppercase(self):
        assert _convert_word("RJH") == "КОР"

    def test_apostrophe_and_semicolon(self):
        # ' → э, ; → ж
        assert _convert_word("'") == "э"
        assert _convert_word(";") == "ж"


# ═══════════════════════════════════════════════════════════════════════════
# 4. fix_keyboard_layout — integration
# ═══════════════════════════════════════════════════════════════════════════
class TestFixKeyboardLayout:
    def test_wrong_layout_word_fixed(self):
        # "vjq" → "мой", "afqk" → "файл"
        out = fix_keyboard_layout("vjq afqk")
        assert "мой" in out
        assert "файл" in out

    def test_cyrillic_unchanged(self):
        assert fix_keyboard_layout("Где корпус?") == "Где корпус?"

    def test_known_latin_token_preserved(self):
        out = fix_keyboard_layout("Как зайти в LMS?")
        assert "LMS" in out

    def test_latin_token_rtu_preserved(self):
        out = fix_keyboard_layout("Где RTU?")
        assert "RTU" in out

    def test_mixed_cyrillic_and_wrong_layout(self):
        # "корпус" typed on English layout:
        # к→r, о→j, р→h, п→g, у→e, с→c → "rjhgec"
        out = fix_keyboard_layout("Где rjhgec?")
        assert "корпус" in out

    def test_empty_string(self):
        assert fix_keyboard_layout("") == ""

    def test_non_string_passthrough(self):
        assert fix_keyboard_layout(None) is None  # type: ignore[arg-type]

    def test_punctuation_preserved(self):
        out = fix_keyboard_layout("vjq!")
        # "vjq!" → "мой!"
        assert "мой" in out
        assert "!" in out

    def test_digits_not_converted(self):
        out = fix_keyboard_layout("5 vjq")
        assert "5" in out


# ═══════════════════════════════════════════════════════════════════════════
# 5. normalize_case
# ═══════════════════════════════════════════════════════════════════════════
class TestNormalizeCase:
    def test_uppercase_lowered(self):
        assert normalize_case("МИРЭА") == "мирэа"

    def test_mixed_case(self):
        assert normalize_case("ГдЕ кОрПуС") == "где корпус"

    def test_already_lower(self):
        assert normalize_case("корпус") == "корпус"

    def test_empty_string(self):
        assert normalize_case("") == ""

    def test_non_string_passthrough(self):
        assert normalize_case(None) is None  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 6. normalize_query — orchestrator
# ═══════════════════════════════════════════════════════════════════════════
class TestNormalizeQueryOrchestrator:
    def test_clean_cyrillic_unchanged(self):
        """The backward-compat contract: clean text is a no-op."""
        assert normalize_query("Где корпус?") == "Где корпус?"

    def test_all_stages_default(self):
        # default flags: ws=True, punct=True, kb=True, case=False
        out = normalize_query("  Где??? rjhgec  ")
        assert "Где" in out
        assert "корпус" in out
        assert "???" not in out  # punctuation folded

    def test_whitespace_only_stage(self):
        out = normalize_query("  Где  корпус  ", punctuation=False,
                              keyboard_layout=False, case=False)
        assert out == "Где корпус"

    def test_punctuation_only_stage(self):
        out = normalize_query("Где???", whitespace=False,
                              keyboard_layout=False, case=False)
        assert out == "Где?"

    def test_keyboard_only_stage(self):
        out = normalize_query("rjhgec", whitespace=False, punctuation=False,
                              case=False)
        assert "корпус" in out

    def test_case_only_stage(self):
        out = normalize_query("МИРЭА", whitespace=False, punctuation=False,
                              keyboard_layout=False, case=True)
        assert out == "мирэа"

    def test_all_disabled_is_identity(self):
        q = "  Где??? МИРЭА  "
        assert normalize_query(q, whitespace=False, punctuation=False,
                               keyboard_layout=False, case=False) == q

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_non_string_passthrough(self):
        assert normalize_query(None) is None  # type: ignore[arg-type]

    def test_order_whitespace_before_punctuation(self):
        # whitespace runs are collapsed first, then punctuation
        out = normalize_query("Где  ???", keyboard_layout=False, case=False)
        assert out == "Где ?"

    def test_ellipsis_not_collapsed_through_orchestrator(self):
        out = normalize_query("Подожди...", keyboard_layout=False, case=False)
        assert out == "Подожди..."


# ═══════════════════════════════════════════════════════════════════════════
# 7. Integration: _normalize_query (answer.py) — backward compat
# ═══════════════════════════════════════════════════════════════════════════
class TestNormalizeIntegration:
    """The existing contract from test_query_expansion.py + test_answer_edge_cases.py
    must still pass: abbreviation expansion still works, clean text passes through."""

    def test_clean_query_passthrough(self):
        """No abbreviation, no mess → identity. The core backward-compat test."""
        assert _normalize_query("Где корпус?") == "Где корпус?"

    def test_returns_string(self):
        assert isinstance(_normalize_query("тест"), str)

    def test_abbreviation_still_expanded(self):
        out = _normalize_query("Что такое ВУЦ?")
        assert "Военный учебный центр" in out

    def test_abbreviation_with_messy_whitespace(self):
        out = _normalize_query("  Что  такое  ВУЦ???  ")
        assert "Военный учебный центр" in out
        assert "ВУЦ" in out
        # whitespace normalised
        assert "  " not in out
        # punctuation folded
        assert "???" not in out

    def test_wrong_layout_then_expansion(self):
        # User types "ВУЦ" on wrong layout: В→D, У→E, Ц→W → "DEW" — no wait,
        # that's the wrong direction. Let me use a real example:
        # "ИИТ" typed on English: И→B, И→B, Т→N → "BBN"
        # That doesn't have a vowel, so it won't be detected.
        # Better: user types the FULL question wrong:
        # "Где rjhgec" → "Где корпус"
        out = _normalize_query("Где rjhgec")
        assert "корпус" in out

    def test_lms_abbreviation_preserved_through_normalization(self):
        # LMS is both an abbreviation AND a Latin token — must survive both stages
        out = _normalize_query("Как зайти в LMS?")
        assert "LMS" in out
        assert "Система управления обучением" in out

    def test_non_string_does_not_crash(self):
        # _normalize_query should never crash — it's in the hot path
        result = _normalize_query(None)  # type: ignore[arg-type]
        # expand_query returns None unchanged, normalize_query returns None
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 8. Idempotence — double application is stable
# ═══════════════════════════════════════════════════════════════════════════
class TestIdempotence:
    """normalize_query is idempotent: applying it twice gives the same result
    as applying it once. This matters because the pipeline may call it on
    already-normalised cached text."""

    @pytest.mark.parametrize("query,kwargs", [
        ("Где корпус?", {}),
        ("  Где??? rjhgec  ", {}),
        ("Как зайти в LMS?", {}),
        ("Что такое ВУЦ???", {}),
        ("Подожди...", {}),
        ("МИРЭА", {"case": True}),
    ])
    def test_double_apply_equals_single(self, query, kwargs):
        once = normalize_query(query, **kwargs)
        twice = normalize_query(once, **kwargs)
        assert once == twice
