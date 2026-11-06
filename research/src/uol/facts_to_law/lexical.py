"""Full-catalogue lexical scoring: BM25, Spanish-stemmed BM25 and character TF-IDF.

With default parameters ``bm25`` and ``char`` reproduce the October channels
exactly: ``uol.retrieval_benchmarks.c01_statutes.agentic.core.Bm25`` (word
tokens ``(?u)\\b\\w+\\b``, lowercase, accent folding, at most 250,000 terms,
IDF ``log1p((N - df + .5) / (df + .5))``, binary query terms, k1 1.2, b .75)
and ``uol.orchestration.deep_baselines.scoring`` character TF-IDF (char_wb
3-5, min_df 2, at most 250,000 features, accent folding, sublinear TF, L2).
The BM25 document weights come from the same October function
``c01_statutes.experiments._bm25_document_matrix`` with the requested k1/b.
Candidate text is ``instrument_article_body`` (full text, no truncation).
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, strip_accents_unicode

from uol.retrieval_benchmarks.c01_statutes.experiments import _bm25_document_matrix

from .spec import sha256_bytes

KINDS = ("bm25", "bm25_stem", "char")
BM25_DEFAULTS = {"k1": 1.2, "b": 0.75}
BM25_GRID = {"k1": (0.9, 1.2, 1.5), "b": (0.4, 0.75)}
TOKEN_PATTERN = r"(?u)\b\w+\b"
MAX_FEATURES = 250_000
CHAR_SETTINGS: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": MAX_FEATURES,
    "lowercase": True,
    "strip_accents": "unicode",
    "sublinear_tf": True,
    "norm": "l2",
}
CANDIDATE_TEXT_FIELD = "instrument_article_body"
_TOKEN = re.compile(TOKEN_PATTERN)


def spanish_stopwords() -> tuple[frozenset[str], dict[str, Any]]:
    """Accent-folded NLTK Spanish stopwords from the local corpus; never downloads."""
    try:
        import nltk
        import nltk.data

        path = nltk.data.find("corpora/stopwords/spanish")
        raw = Path(str(path)).read_bytes()
    except (ImportError, LookupError, OSError) as error:
        return frozenset(), {"available": False, "error": f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"}
    words = [w.strip() for w in raw.decode("utf-8").splitlines() if w.strip()]
    folded = frozenset(strip_accents_unicode(w.lower()) for w in words)
    return folded, {"available": True, "path": str(path), "sha256": sha256_bytes(raw), "words": len(words),
                    "folded_words": len(folded), "nltk": nltk.__version__}


class SpanishAnalyzer:
    """October word tokens, Spanish stopword removal, Snowball stems, accent folding.

    Tokens are lowercased and matched with ``(?u)\\b\\w+\\b``. A token is dropped
    when its accent-folded form is an NLTK Spanish stopword; otherwise its
    Snowball stem (computed on the accented token) is accent-folded.
    """

    def __init__(self, use_stopwords: bool = True) -> None:
        from nltk.stem.snowball import SnowballStemmer

        self.stemmer = SnowballStemmer("spanish")
        if use_stopwords:
            self.stopwords, stop_receipt = spanish_stopwords()
        else:
            self.stopwords, stop_receipt = frozenset(), {"available": False, "error": "disabled by configuration"}
        import nltk

        self.receipt = {"stemmer": "nltk.stem.snowball.SnowballStemmer('spanish')", "nltk": nltk.__version__,
                        "stopwords": stop_receipt, "stopwords_applied": bool(self.stopwords),
                        "token_pattern": TOKEN_PATTERN, "order": "lowercase, tokenize, stopword check on folded token, stem, fold accents"}
        self._cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        cache, out = self._cache, []
        for token in _TOKEN.findall(text.lower()):
            stem = cache.get(token)
            if stem is None:
                stem = "" if strip_accents_unicode(token) in self.stopwords else strip_accents_unicode(self.stemmer.stem(token))
                cache[token] = stem
            if stem:
                out.append(stem)
        return out


class LexicalIndex:
    """A fitted candidate index scoring query texts against every candidate.

    ``kind`` is ``bm25``, ``bm25_stem`` or ``char``. BM25 kinds accept ``k1`` and
    ``b`` in ``params`` (defaults 1.2 and .75); ``score`` may override them per
    call, reusing the fitted term counts. Scores are float32 ``[n_q, n_c]``,
    computed as sparse products in query batches.
    """

    def __init__(self, candidate_texts: Sequence[str], kind: str = "bm25", params: Mapping[str, Any] | None = None,
                 *, batch_size: int = 64, max_cached_weights: int = 2) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown lexical kind: {kind}")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        texts = list(candidate_texts)
        if not texts:
            raise ValueError("an empty catalogue cannot be indexed")
        self.kind = kind
        self.batch_size = batch_size
        self.n_candidates = len(texts)
        self.analyzer_receipt: dict[str, Any] | None = None
        self._weights: dict[tuple[float, float], sparse.csr_matrix] = {}
        self._max_cached = max(1, max_cached_weights)
        if kind == "char":
            if params:
                raise ValueError("character TF-IDF has no tunable parameters")
            self.params: dict[str, Any] = {}
            self.vectorizer = TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32)
            self._documents_t = self.vectorizer.fit_transform(texts).tocsr().T.tocsr()
            self._counts = None
        else:
            self.params = self._bm25_params(params)
            if kind == "bm25_stem":
                analyzer = SpanishAnalyzer()
                self.analyzer_receipt = analyzer.receipt
                self.vectorizer = CountVectorizer(analyzer=analyzer, min_df=1, max_features=MAX_FEATURES, dtype=np.float32)
            else:
                self.vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=1, max_features=MAX_FEATURES,
                                                  lowercase=True, strip_accents="unicode", dtype=np.float32)
            self._counts = self.vectorizer.fit_transform(texts).tocsr()
            self._documents_t = None

    @staticmethod
    def _bm25_params(params: Mapping[str, Any] | None) -> dict[str, float]:
        values = {**BM25_DEFAULTS, **dict(params or {})}
        if set(values) != set(BM25_DEFAULTS):
            raise ValueError(f"BM25 accepts only k1 and b, got {sorted(values)}")
        k1, b = float(values["k1"]), float(values["b"])
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        return {"k1": k1, "b": b}

    def _document_weights(self, params: Mapping[str, Any] | None) -> sparse.csr_matrix:
        if self.kind == "char":
            if params:
                raise ValueError("character TF-IDF has no tunable parameters")
            return self._documents_t
        values = self._bm25_params({**self.params, **dict(params or {})})
        key = (values["k1"], values["b"])
        if key not in self._weights:
            while len(self._weights) >= self._max_cached:
                self._weights.pop(next(iter(self._weights)))
            self._weights[key] = _bm25_document_matrix(self._counts, k1=key[0], b=key[1]).T.tocsr()
        return self._weights[key]

    def transform_queries(self, query_texts: Sequence[str]) -> sparse.csr_matrix:
        queries = self.vectorizer.transform(list(query_texts))
        if self.kind == "char":
            return queries.tocsr()
        queries = queries.astype(np.float32).tocsr()
        queries.data[:] = 1.0
        return queries

    def score(self, query_texts: Sequence[str], params: Mapping[str, Any] | None = None) -> np.ndarray:
        """Float32 scores ``[len(query_texts), n_candidates]``."""
        documents_t = self._document_weights(params)
        queries = self.transform_queries(query_texts)
        out = np.empty((queries.shape[0], self.n_candidates), dtype=np.float32)
        for begin in range(0, queries.shape[0], self.batch_size):
            out[begin:begin + self.batch_size] = (queries[begin:begin + self.batch_size] @ documents_t).toarray()
        if not np.isfinite(out).all():
            raise ValueError(f"non-finite {self.kind} scores")
        return out

    def receipt(self) -> dict[str, Any]:
        """Implementation, settings and index size for run records."""
        matrix = self._documents_t if self.kind == "char" else self._counts
        settings = ({k: list(v) if isinstance(v, tuple) else v for k, v in CHAR_SETTINGS.items()} if self.kind == "char" else
                    {"token_pattern": TOKEN_PATTERN, "min_df": 1, "max_features": MAX_FEATURES, "lowercase": True,
                     "strip_accents": "unicode" if self.kind == "bm25" else "after stemming",
                     "idf": "log1p((N - df + 0.5) / (df + 0.5))", "query_terms": "binary"})
        return {"kind": self.kind, "params": dict(self.params), "settings": settings, "candidates": self.n_candidates,
                "vocabulary": int(len(self.vectorizer.vocabulary_)), "nnz": int(matrix.nnz),
                "implementation": "uol.facts_to_law.lexical.LexicalIndex",
                "october_reference": ("uol.orchestration.deep_baselines.scoring.lexical_score_matrix('character_tfidf')" if self.kind == "char"
                                      else "uol.retrieval_benchmarks.c01_statutes.agentic.core.Bm25 with experiments._bm25_document_matrix(k1, b)"),
                "analyzer": self.analyzer_receipt}


def bm25_grid() -> list[dict[str, float]]:
    """The declared BM25 grid in deterministic order (k1 major, b minor)."""
    return [{"k1": k1, "b": b} for k1 in BM25_GRID["k1"] for b in BM25_GRID["b"]]
