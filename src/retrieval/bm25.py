"""BM25 (Okapi) sparse retrieval — pure numpy, no external dependency.

BM25 is the standard lexical retrieval baseline: it scores documents by a
saturation-weighted term frequency (k1) modulated by document-length
normalisation (b). For Russian text we lowercase, split on word boundaries,
and drop a compact stopword list so frequent function words don't dominate.

The index is built once from the chunk corpus and pickled to disk; at query
time it is a cheap dictionary lookup + numpy dot, suitable for online use.

Algorithm (BM25Okapi, Robertson & Zaragoza 2009):

    score(q, d) = Σ_{t∈q} IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1-b+b·|d|/avgdl))

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))   # the non-negative form

Usage:
    idx = BM25Index.build(corpus_texts, tokenizer=tokenize_ru)
    idx.save("mirea_data/bm25/bm25_index.pkl")
    idx = BM25Index.load("mirea_data/bm25/bm25_index.pkl")
    hits = idx.search("проходной балл программная инженерия", top_k=5)
    # -> [(doc_id, score), ...]
"""
from __future__ import annotations

import logging
import os
import pickle
import re
import sys
from typing import Callable, Iterable

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Russian tokeniser
# --------------------------------------------------------------------------- #

# Compact Russian function-word list. Deliberately small — BM25's IDF already
# down-weights common words, but removing stopwords shrinks the vocabulary and
# speeds up scoring. Curated for university-admissions domain (mirea.ru).
_RU_STOPWORDS = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтобы без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между лих
такие тех которое которых этого этом этих каждому какая какие каком какой
каким каких ей ее еще ещё около через более менее теперь
""".split())

# Word tokens: Cyrillic (with ё) + Latin + digits. Strips markdown (**, #),
# punctuation, and everything else. Case-folded.
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize_ru(text: str, remove_stopwords: bool = True, min_len: int = 2) -> list[str]:
    """Lowercase + split into alphanumeric tokens, optionally drop stopwords.

    Args:
        text: Raw chunk/query text (may contain markdown, punctuation).
        remove_stopwords: Drop Russian function words (default True).
        min_len: Minimum token length in characters (default 2 drops bare
            digits/letters that carry little signal).

    Returns:
        Ordered list of tokens. Repeats are preserved (they feed term
        frequency), but order is not used by BM25 scoring.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        return [t for t in tokens if len(t) >= min_len and t not in _RU_STOPWORDS]
    return [t for t in tokens if len(t) >= min_len]


# --------------------------------------------------------------------------- #
# BM25 index
# --------------------------------------------------------------------------- #


class BM25Index:
    """Numpy-backed BM25Okapi over a fixed document corpus.

    The corpus is immutable after build: the inverted index, document
    frequencies, and length statistics are all precomputed. For a streaming
    corpus you'd rebuild (cheap for ≤1e5 docs).

    Attributes:
        n_docs:        Number of indexed documents.
        avgdl:         Average document length (in tokens).
        k1, b:         BM25 saturation / length-normalisation params.
        idf:           dict[term] -> idf weight (non-negative form).
        inverted:      dict[term] -> (np.ndarray doc_ids, np.ndarray tf).
        doc_len:       np.ndarray of per-doc token counts.
    """

    def __init__(
        self,
        inverted: dict[str, tuple[np.ndarray, np.ndarray]],
        doc_len: np.ndarray,
        idf: dict[str, float],
        k1: float,
        b: float,
    ) -> None:
        self.inverted = inverted
        self.doc_len = np.asarray(doc_len, dtype=np.float64)
        self.idf = idf
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = int(len(doc_len))
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

    # ---- build / load / save --------------------------------------------- #

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        tokenizer: Callable[[str], list[str]] = tokenize_ru,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Build a BM25 index from an iterable of document texts.

        Args:
            corpus: Iterable of raw text documents (one per chunk).
            tokenizer: Callable text -> list[str]. Defaults to tokenize_ru.
            k1: Term-frequency saturation (1.2–2.0 typical; 1.5 is standard).
            b: Document-length normalisation (0.0 = none, 1.0 = full; 0.75
                is the Okapi default).

        Returns:
            A ready-to-query BM25Index.
        """
        doc_tokens: list[list[str]] = [tokenizer(t) for t in corpus]
        n_docs = len(doc_tokens)
        if n_docs == 0:
            raise ValueError("Cannot build BM25 index over an empty corpus")

        # Per-term postings: term -> {doc_id: tf}. Single pass over all docs.
        postings: dict[str, dict[int, int]] = {}
        doc_len = np.zeros(n_docs, dtype=np.float64)
        for d, tokens in enumerate(doc_tokens):
            doc_len[d] = len(tokens)
            tf_d: dict[str, int] = {}
            for tok in tokens:
                tf_d[tok] = tf_d.get(tok, 0) + 1
            for term, tf in tf_d.items():
                postings.setdefault(term, {})[d] = tf

        # Collapse to numpy arrays for vectorised scoring, and compute IDF.
        inverted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        idf: dict[str, float] = {}
        for term, post in postings.items():
            doc_ids = np.fromiter(post.keys(), dtype=np.int64, count=len(post))
            tfs = np.fromiter(post.values(), dtype=np.float64, count=len(post))
            inverted[term] = (doc_ids, tfs)
            df = len(post)
            # Non-negative IDF (Robertson form): guarantees score ≥ 0.
            idf[term] = float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))

        log.info(
            "BM25 index built: %d docs, %d unique terms, avgdl=%.1f tokens",
            n_docs, len(inverted), float(doc_len.mean()),
        )
        return cls(inverted, doc_len, idf, k1, b)

    def save(self, path: str) -> None:
        """Pickle the index to disk. Parent dirs are created."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {
            "inverted": {
                t: (doc_ids.tolist(), tfs.tolist())
                for t, (doc_ids, tfs) in self.inverted.items()
            },
            "doc_len": self.doc_len.tolist(),
            "idf": self.idf,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("BM25 index saved → %s (%d terms)", path, len(self.inverted))

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a pickled index from disk."""
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inverted = {
            t: (np.asarray(doc_ids, dtype=np.int64), np.asarray(tfs, dtype=np.float64))
            for t, (doc_ids, tfs) in blob["inverted"].items()
        }
        idx = cls(
            inverted,
            np.asarray(blob["doc_len"], dtype=np.float64),
            blob["idf"],
            blob["k1"],
            blob["b"],
        )
        log.info("BM25 index loaded ← %s (%d docs, %d terms)",
                 path, idx.n_docs, len(idx.inverted))
        return idx

    # ---- search ---------------------------------------------------------- #

    def _score_term(self, term: str) -> np.ndarray | None:
        """Return per-doc BM25 contribution for one query term, or None if OOV.

        Vectorised over the term's posting list; documents not containing the
        term contribute 0 (sparse, so we only touch the matching docs).
        """
        post = self.inverted.get(term)
        if post is None:
            return None
        doc_ids, tfs = post
        idf = self.idf[term]
        # Length normalisation factor for each matching doc.
        denom = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_ids] / self.avgdl)
        scores = idf * (tfs * (self.k1 + 1.0)) / (tfs + denom)
        return np.stack([doc_ids, scores], axis=1)  # (n_match, 2): [id, score]

    def search(
        self,
        query: str | list[str],
        top_k: int = 10,
        tokenizer: Callable[[str], list[str]] = tokenize_ru,
    ) -> list[tuple[int, float]]:
        """Score the corpus against ``query`` and return the top_k hits.

        Args:
            query: Raw query string (tokenised internally) OR a pre-tokenised
                list. Passing a list skips re-tokenisation, which lets the
                caller reuse the same tokens across backends.
            top_k: Number of results to return.
            tokenizer: Tokeniser used only when ``query`` is a string.

        Returns:
            List of ``(doc_id, score)`` tuples, descending by score. Scores
            are ≥ 0 (non-negative IDF form). Empty list if no query term is
            in the vocabulary.
        """
        if isinstance(query, str):
            q_tokens = tokenizer(query)
        else:
            q_tokens = query
        if not q_tokens:
            return []

        # Accumulate per-doc scores across query terms (sparse scatter).
        contributions: list[np.ndarray] = []
        seen_terms: set[str] = set()
        for term in q_tokens:
            if term in seen_terms:
                continue  # a query term repeated N times shouldn't score N times
            seen_terms.add(term)
            block = self._score_term(term)
            if block is not None:
                contributions.append(block)
        if not contributions:
            return []

        all_scores = np.concatenate(contributions, axis=0)  # (total_matches, 2)
        doc_ids = all_scores[:, 0].astype(np.int64)
        raw = all_scores[:, 1]

        # Sum scores for docs that match multiple query terms (np.add.at).
        agg = np.zeros(self.n_docs, dtype=np.float64)
        np.add.at(agg, doc_ids, raw)

        k = min(top_k, self.n_docs)
        if k >= self.n_docs:
            order = np.argsort(-agg)
        else:
            part = np.argpartition(-agg, k)[:k]
            order = part[np.argsort(-agg[part])]
        return [(int(i), float(agg[i])) for i in order if agg[i] > 0]


def default_index_path() -> str:
    """Resolve the on-disk BM25 index path from config (with fallback)."""
    return getattr(config, "BM25_INDEX_PATH", os.path.join(config.DATA_FAISS_DIR, "..", "bm25", "bm25_index.pkl"))


if __name__ == "__main__":
    # Quick smoke test on the real corpus.
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    meta_path = os.path.join(config.DATA_FAISS_DIR, "metadata.json")
    meta = json.loads(open(meta_path, encoding="utf-8").read())
    texts = [m["text"] for m in meta]
    idx = BM25Index.build(texts)
    for q in [
        "Какой проходной балл на программную инженерию?",
        "Где находится главный корпус МИРЭА?",
        "Сколько бюджетных мест на очной форме?",
    ]:
        hits = idx.search(q, top_k=3)
        print(f"\n{q}")
        for did, sc in hits:
            print(f"  #{did:3d}  {sc:7.3f}  {meta[did]['text'][:70]}")
