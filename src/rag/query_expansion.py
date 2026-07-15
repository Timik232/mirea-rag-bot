"""Query expansion for МИРЭА abbreviations (task t_4a4bd2e3).

Users of a university FAQ bot frequently type abbreviations (ЕГЭ, ПГАС, ВУЦ,
ИИТ, …) whose *full* form – not the abbreviation – is what actually appears in
the knowledge-base chunks, or vice-versa.  The multilingual-e5-small embedder
is weak at bridging that vocabulary gap on its own, so we expand recognised
abbreviations to their full Russian form before the query is embedded.

Design goals
------------
* **Append, never replace** – the original token stays (it is still a strong
  signal for chunks that use the abbreviation verbatim); we only *add* the
  expansion so the embedding blends both forms.
* **Idempotent** – if the full form is already present in the query the
  abbreviation is skipped, so ``expand_query(expand_query(q)) == expand_query(q)``
  and a user who typed the full form is not polluted.
* **Token-safe** – matches respect letter boundaries for *both* Cyrillic and
  Latin scripts, so ``ИИТ`` matches but the ``ИИТ`` inside a longer token never
  does, and ``РТУ`` does not fire on the ``РТ`` inside ``РТВ``.
* **Conservative by default** – the bot's own name (``МИРЭА`` / ``РТУ``) is
  *not* expanded: it occurs in most chunks and most queries, so appending its
  long form only dilutes the vector.  Expansion targets domain vocabulary
  (institutes, exams, grants, platforms) where a real query↔chunk mismatch
  exists.  The full table is still exposed via :data:`ABBREVIATIONS_ALL` so the
  behaviour can be widened by flipping :data:`EXPAND_UNIVERSITY_NAME`.
* **Failure-safe** – any unexpected input returns the query unchanged; this
  module must never break the answer pipeline.

Public API
----------
.. autofunction:: expand_query
.. autodata:: ABBREVIATIONS
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Abbreviation table
# --------------------------------------------------------------------------- #
# Canonical key is the form most likely typed (uppercase, except ИЭиУ whose
# official short form mixes case).  Matching is case-insensitive (see below),
# so "егэ", "ЕГЭ", "Егэ" all resolve to the same entry.
#
# Each value is the *full official form* as it appears (or should appear) in
# the knowledge base, minus trailing punctuation.
ABBREVIATIONS: dict[str, str] = {
    # ── entrance exams / education levels ───────────────────────────────
    "ЕГЭ": "Единый государственный экзамен",
    "ИВТ": "Информатика и вычислительная техника",
    "ИКТ": "Информационно-коммуникационные технологии",
    "ФГБОУ": "Федеральное государственное бюджетное образовательное учреждение",
    # ── financial aid ───────────────────────────────────────────────────
    "ПГАС": "Повышенная государственная академическая стипендия",
    # ── military / platforms ────────────────────────────────────────────
    "ВУЦ": "Военный учебный центр",
    "LMS": "Система управления обучением",
    # ── documents / misc ────────────────────────────────────────────────
    "СНИЛС": "Страховой номер индивидуального лицевого счёта",
    "ВДНХ": "Выставка достижений народного хозяйства",
    "КВН": "Клуб весёлых и находчивых",
    "ММОО": "Московская молодёжная общественная организация",
    # ── institutes (source: chunk 78, official structure page) ──────────
    "ИИТ": "Институт информационных технологий",
    "ИРСС": "Институт радиотехники и систем связи",
    "ИКБ": "Институт кибербезопасности и криптографии",
    "ИИРС": "Институт интеллектуальных робототехнических систем",
    "ИОЭСТ": "Институт оптико-электронных систем и технологий",
    "ИЭБП": "Институт экологической безопасности и природопользования",
    "ИЭиУ": "Институт экономики и управления",
    "ИВО": "Институт высшего образования",
    "ИДВИ": "Институт дизайна и визуальных искусств",
}

# The bot's own name.  Off by default – see module docstring.
ABBREVIATIONS_ALL: dict[str, str] = {
    **ABBREVIATIONS,
    "МИРЭА": "Московский институт радиотехники, электроники и автоматики",
    "РТУ": "Российский технологический университет",
    "РФ": "Российская Федерация",
}

# Whether to also expand МИРЭА / РТУ / РФ.  Off by default: these tokens are
# already densely represented in both queries and chunks, and appending their
# long forms was measured to dilute retrieval (see eval/retrieval_report.json,
# task t_4a4bd2e3).  Flip to True to re-enable.
EXPAND_UNIVERSITY_NAME: bool = False


def _active_table() -> dict[str, str]:
    return ABBREVIATIONS_ALL if EXPAND_UNIVERSITY_NAME else ABBREVIATIONS


# --------------------------------------------------------------------------- #
# Matching machinery
# --------------------------------------------------------------------------- #
# A "letter" for boundary purposes = Unicode letter that is not a digit or
# underscore.  ``[^\W\d_]`` is exactly that, and works for Cyrillic AND Latin.
# We require the abbreviation to be bordered by non-letters (or string edges)
# so that "ИИТ" matches but "ИИТ-2024" still matches (hyphen is non-letter),
# while the "РТ" inside "АРТУР" never would.
_BORDER = r"(?:(?<=[^\W\d_])|(?<=\A))"   # preceded by letter OR start-of-string
# Simpler & robust: use lookarounds that forbid a letter on either side.
def _build_regex(table: dict[str, str]) -> re.Pattern[str]:
    # Sort longest-first so "ИОЭСТ" is tried before a shorter prefix match.
    keys = sorted(table, key=len, reverse=True)
    # Escape each key (handles the lowercase 'и' in ИЭиУ and Latin LMS).
    alt = "|".join(re.escape(k) for k in keys)
    # No letter immediately before/after the matched token.
    return re.compile(rf"(?<![^\W\d_])(?:{alt})(?![^\W\d_])", re.IGNORECASE)


def expand_query(query: str) -> str:
    """Return *query* with recognised МИРЭА abbreviations expanded.

    The original abbreviation tokens are preserved; each recognised
    abbreviation that is *not already accompanied by its full form* contributes
    its expansion, appended once at the end of the query.

    Examples
    --------
    >>> expand_query("Как поступить на ИИТ?")
    'Как поступить на ИИТ? Институт информационных технологий'
    >>> expand_query("Что такое ПГАС?")
    'Что такое ПГАС? Повышенная государственная академическая стипендия'
    >>> # full form already present -> no-op (idempotent)
    >>> expand_query("Повышенная государственная академическая стипендия?")
    'Повышенная государственная академическая стипендия?'
    >>> # no abbreviations -> unchanged
    >>> expand_query("Где общежитие?")
    'Где общежитие?'
    """
    if not isinstance(query, str) or not query.strip():
        return query  # type: ignore[return-value]

    table = _active_table()
    if not table:
        return query

    try:
        rx = _build_regex(table)
        # Look up by casefold so keys with embedded lowercase letters
        # (e.g. ИЭиУ) resolve correctly regardless of how they were typed.
        cf_table = {k.casefold(): v for k, v in table.items()}
        found = {cf_table[m.group(0).casefold()]
                 for m in rx.finditer(query)}
    except re.error as e:  # pragma: no cover - dict is static, regex is valid
        log.warning("query_expansion regex failed (%s); returning raw query", e)
        return query

    if not found:
        return query

    # Idempotence / no-pollution: drop an expansion if its full form is already
    # in the query (case-insensitive).  Also drop if it duplicates what we are
    # about to append.
    lowered = query.casefold()
    additions: list[str] = []
    seen: set[str] = set()
    for expansion in found:
        key = expansion.casefold()
        if key in lowered or key in seen:
            continue
        seen.add(key)
        additions.append(expansion)

    if not additions:
        return query
    return f"{query.rstrip()} {' '.join(additions)}"
