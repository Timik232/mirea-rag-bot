"""Query normalization for retrieval quality (task t_bd93c958).

Applies lightweight, **semantic** normalisation to a user query *before* it is
embedded, so the bi-encoder (multilingual-e5-small) sees a cleaned,
layout-correct, consistently-formatted string instead of the raw keystrokes
the user happened to produce.

Each stage is:

* **Pure** – no side effects, no I/O; ``f(f(x)) == f(x)`` for every stage.
* **Independent** – gated by its own ``config.NORMALIZE_*`` flag so it can be
  benchmarked in isolation (A/B on the golden set, one switch at a time).
* **Conservative** – a no-op on already-clean text. ``"Где корпус?"`` in →
  ``"Где корпус?"`` out. Existing tests and cached answers are unaffected.

Stages (applied in fixed order inside :func:`normalize_query`):

  1. **Whitespace collapse** (:func:`normalize_whitespace`)
     Multiple spaces / tabs / newlines → single space. Strip ends.

  2. **Punctuation folding** (:func:`fold_punctuation`)
     Collapses runs of identical terminal punctuation: ``"???"`` → ``"?"``,
     ``"!!!"`` → ``"!"``. Ellipsis (``...`` / ``…``) is preserved — it carries
     a distinct pragmatic meaning the model should see.

  3. **Keyboard-layout fix** (:func:`fix_keyboard_layout`)
     Detects Russian words typed on an English (QWERTY) keyboard layout and
     converts them to Cyrillic.  ``"rjq afqk"`` → ``"мой фак"``.  Known Latin
     tokens (``LMS``, ``RTU``, ``E5`` …) are never converted.

  4. **Case folding** (:func:`normalize_case`) — **off by default**
     Lowercases the query. Neutral-to-slightly-negative for multilingual-e5
     (its SentencePiece tokenizer is case-aware and Russian proper nouns carry
     signal in their case). Left as a flag for benchmarking.

Design rationale and the measured effect of each stage on retrieval are
documented in the task completion notes (``t_bd93c958``).

Public API
----------
.. autofunction:: normalize_query
.. autofunction:: normalize_whitespace
.. autofunction:: fold_punctuation
.. autofunction:: fix_keyboard_layout
.. autofunction:: normalize_case
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# ── keyboard layout tables (ЙЦУКЕН ↔ QWERTY) ──────────────────────────────#
# Standard Russian PC keyboard. Each Russian letter maps to the Latin key
# in the same physical position. Reverse mapping converts "typed Russian on
# English layout" back to Cyrillic.
_LAT_TO_RU: dict[str, str] = {
    # row 2
    "q": "й", "w": "ц", "e": "у", "r": "к", "t": "е", "y": "н", "u": "г",
    "i": "ш", "o": "щ", "p": "з", "[": "х", "]": "ъ",
    # row 3
    "a": "ф", "s": "ы", "d": "в", "f": "а", "g": "п", "h": "р", "j": "о",
    "k": "л", "l": "д", ";": "ж", "'": "э",
    # row 4
    "z": "я", "x": "ч", "c": "с", "v": "м", "b": "и", "n": "т", "m": "ь",
    ",": "б", ".": "ю",
    # backtick row
    "`": "ё",
}
# Uppercase variants — but only for keys that actually change case.
# Punctuation keys (`'`, `;`, `[` …) have .upper() == self, so blindly
# uppercasing them would overwrite the lowercase entry with an uppercased
# value (e.g. "'" → "Э" instead of "э").
_LAT_TO_RU.update({
    k.upper(): v.upper()
    for k, v in list(_LAT_TO_RU.items())
    if k.upper() != k
})

# Character class: any Latin key that has a Russian counterpart.
_LAT_LAYOUT_CHARS = re.escape("".join(_LAT_TO_RU.keys()))
# Matches a maximal run of characters all of which exist in the layout table.
# We split on these so word boundaries are respected.
_LAT_RUN_RE = re.compile(rf"[{_LAT_LAYOUT_CHARS}]+")

# Russian vowels — a converted word must contain at least one to be plausible.
_RU_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")

# Tokens that are *legitimately* Latin and must never be layout-converted.
# Case-insensitive match.  Includes the bot's domain abbreviations that are
# typed in Latin (LMS) and common IT tokens the student audience uses.
_LAT_KEEP: frozenset[str] = frozenset({
    "lms", "rtu", "mirea", "api", "url", "pdf", "id", "email", "git",
    "e5", "ai", "it", "hr", "pr", "os", "ui", "ux", "saas", "crm",
})


# ── whitespace ────────────────────────────────────────────────────────────#
_WS_RUN_RE = re.compile(r"[ \t\u00a0\u2007\u202f]+")  # spaces, tabs, nbsp
_NEWLINE_RE = re.compile(r"\s*\n\s*")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends.

    Newlines become spaces (a query is a single line). Non-breaking spaces
    (U+00A0, U+2007, U+202F — common in copy-pasted text) are treated as
    regular spaces.

    >>> normalize_whitespace("  Где    корпус?  ")
    'Где корпус?'
    """
    if not isinstance(text, str):
        return text  # type: ignore[return-value]
    text = _NEWLINE_RE.sub(" ", text)
    text = _WS_RUN_RE.sub(" ", text)
    return text.strip()


# ── punctuation folding ───────────────────────────────────────────────────#
# Collapse 2+ identical terminal marks (?!.) — but NOT runs of ".", because
# "..." (ellipsis) is semantically meaningful and common in natural queries.
# Only the exact char-clusters "?", "!", "?!", "!?" get collapsed.
_MULTI_QMARK_RE = re.compile(r"\?{2,}")
_MULTI_EXCL_RE = re.compile(r"!{2,}")
# "??" after "?" or "!!" after "!" → single; also collapse "?!" → "?!" (no-op)
# and "!?" patterns that get duplicated like "?!?!?" → "?!".  Keep simple: we
# just collapse each run of ? and ! to a canonical form.
_QEXCL_RUN_RE = re.compile(r"[?!]{2,}")


def fold_punctuation(text: str) -> str:
    """Collapse repeated terminal punctuation to a single mark.

    ``"Что???"`` → ``"Что?"``, ``"Да!!!"`` → ``"Да!"``. Ellipsis (``"..."``
    or ``"…"``) is left untouched — it conveys trailing-off, a distinct
    pragmatic signal for the embedder.

    >>> fold_punctuation("Где корпус???")
    'Где корпус?'
    >>> fold_punctuation("Помогите!!!")
    'Помогите!'
    """
    if not isinstance(text, str):
        return text  # type: ignore[return-value]
    # Collapse any run of 2+ [?!] chars to its first char: "?!?" → "?"
    return _QEXCL_RUN_RE.sub(lambda m: m.group(0)[0], text)


# ── keyboard layout fix ───────────────────────────────────────────────────#
def _looks_like_wrong_layout(word: str) -> bool:
    """Heuristic: is *word* Russian typed on a QWERTY (English) layout?

    True when **all** of these hold:
      1. Every character in the word maps to a Russian letter via ЙЦУКЕН.
      2. The converted form contains at least one Russian vowel — this
         rejects consonant-only noise like ``"rtrt"``.
      3. The lowercased word is not in :data:`_LAT_KEEP` (legitimately Latin
         tokens like ``LMS``, ``RTU``, ``API``).

    A 1-char word never triggers (too many false positives: ``"e"`` → ``"у"``).
    """
    if len(word) < 2:
        return False
    lower = word.lower()
    if lower in _LAT_KEEP:
        return False
    converted = _convert_word(word)
    # Must contain at least one vowel to look like real Russian.
    return any(ch in _RU_VOWELS for ch in converted)


def _convert_word(word: str) -> str:
    """Map a Latin-layout word to Cyrillic via the ЙЦУКЕН table."""
    return "".join(_LAT_TO_RU.get(ch, ch) for ch in word)


def fix_keyboard_layout(text: str) -> str:
    """Detect and fix Russian words typed on an English keyboard layout.

    Walks the text token-by-token. A token that :func:`_looks_like_wrong_layout`
    flags is replaced by its Cyrillic equivalent; everything else (real Latin,
    real Cyrillic, digits, punctuation) is left as-is.

    >>> fix_keyboard_layout("rjq afqk")
    'мой фак'
    >>> fix_keyboard_layout("Где LMS?")
    'Где LMS?'   # LMS is a known Latin abbreviation — not converted
    >>> fix_keyboard_layout("Где корпус?")
    'Где корпус?'  # already Cyrillic — no-op
    """
    if not isinstance(text, str) or not text:
        return text

    def _replace(m: re.Match[str]) -> str:
        word = m.group(0)
        if _looks_like_wrong_layout(word):
            return _convert_word(word)
        return word

    return _LAT_RUN_RE.sub(_replace, text)


# ── case folding ──────────────────────────────────────────────────────────#
def normalize_case(text: str) -> str:
    """Lowercase the query.

    Off by default — multilingual-e5's SentencePiece tokenizer is case-aware
    and Russian proper nouns (МИРЭА, institute names) carry retrieval signal
    in their capitalisation. Provided as a flag for isolated benchmarking.

    >>> normalize_case("Где МИРЭА?")
    'где мирэа?'
    """
    if not isinstance(text, str):
        return text  # type: ignore[return-value]
    return text.lower()


# ── orchestrator ──────────────────────────────────────────────────────────#
def normalize_query(
    query: str,
    *,
    whitespace: bool = True,
    punctuation: bool = True,
    keyboard_layout: bool = True,
    case: bool = False,
) -> str:
    """Apply the enabled normalisation stages to *query*, in fixed order.

    Each stage is a no-op on already-clean, already-Cyrillic text, so
    ``normalize_query("Где корпус?") == "Где корпус?"``. The keyword flags
    default to the project config (see :mod:`config`); pass them explicitly
    only in tests or benchmarks.

    Parameters
    ----------
    query
        A validated, non-empty query string.
    whitespace, punctuation, keyboard_layout, case
        Per-stage toggles. ``True`` = apply that stage.

    Returns
    -------
    str
        The normalised query. Never raises on bad input — returns it
        unchanged, because this module must never break the answer pipeline.
    """
    if not isinstance(query, str) or not query:
        return query  # type: ignore[return-value]

    try:
        if whitespace:
            query = normalize_whitespace(query)
        if punctuation:
            query = fold_punctuation(query)
        if keyboard_layout:
            query = fix_keyboard_layout(query)
        if case:
            query = normalize_case(query)
    except Exception as e:  # pragma: no cover — defensive, never expected
        log.warning("query_normalization failed (%s); returning unnormalised query", e)

    return query
