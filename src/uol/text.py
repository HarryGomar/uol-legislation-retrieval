"""Lexical indexes: BM25 over words or Spanish stems, and character n-gram TF-IDF.

All scores are float32. BM25 queries are binary (a term counts once however often it occurs in the facts), and the
document weights follow Robertson's BM25 with ``idf = log(1 + (N - df + 0.5) / (df + 0.5))``.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, strip_accents_unicode

TOKEN = r"(?u)\b\w+\b"
MAX_FEATURES = 250_000
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=MAX_FEATURES, lowercase=True,
            strip_accents="unicode", sublinear_tf=True, norm="l2")
_TOKEN = re.compile(TOKEN)


@lru_cache(maxsize=1)
def spanish_stopwords() -> frozenset[str]:
    """NLTK's Spanish stopwords, accent-folded. Fails rather than silently scoring without them."""
    import nltk.data
    from pathlib import Path
    path = nltk.data.find("corpora/stopwords/spanish")       # LookupError: run  python -m nltk.downloader stopwords
    words = Path(str(path)).read_text(encoding="utf-8").split()
    return frozenset(strip_accents_unicode(w.lower()) for w in words)


class SpanishStems:
    """Lowercase word tokens; a token whose accent-folded form is a stopword is dropped, the others are replaced by
    their accent-folded Snowball stem (computed on the accented token)."""

    def __init__(self) -> None:
        from nltk.stem.snowball import SnowballStemmer
        self.stem, self.stop, self.cache = SnowballStemmer("spanish").stem, spanish_stopwords(), {}

    def __call__(self, text: str) -> list[str]:
        out = []
        for token in _TOKEN.findall(text.lower()):
            s = self.cache.get(token)
            if s is None:
                s = self.cache[token] = "" if strip_accents_unicode(token) in self.stop else strip_accents_unicode(self.stem(token))
            if s:
                out.append(s)
        return out


def word_vectorizer() -> CountVectorizer:
    return CountVectorizer(token_pattern=TOKEN, min_df=1, max_features=MAX_FEATURES, lowercase=True,
                           strip_accents="unicode", dtype=np.float32)


def bm25_documents(counts: sparse.csr_matrix, k1: float, b: float) -> sparse.csr_matrix:
    """BM25 weights of a document-term count matrix."""
    w = counts.astype(np.float32, copy=True).tocsr()
    df = np.asarray((w > 0).sum(axis=0)).ravel()
    idf = np.log1p((w.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    length = np.asarray(w.sum(axis=1)).ravel().astype(np.float32)
    avg = float(length.mean())
    row = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    tf = w.data
    w.data = idf[w.indices] * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * length[row] / max(avg, 1e-9)))
    return w


class BM25:
    """BM25 over a fixed set of documents; ``stems`` selects Spanish stems instead of words."""

    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75, stems: bool = False) -> None:
        self.vectorizer = (CountVectorizer(analyzer=SpanishStems(), min_df=1, max_features=MAX_FEATURES, dtype=np.float32)
                           if stems else word_vectorizer())
        self.weights_t = bm25_documents(self.vectorizer.fit_transform(list(documents)).tocsr(), k1, b).T.tocsr()

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return (binary_queries(self.vectorizer, texts) @ self.weights_t).toarray().astype(np.float32, copy=False)


class CharTfidf:
    """Cosine similarity of character 3- to 5-gram TF-IDF vectors (sublinear term frequency)."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.vectorizer = TfidfVectorizer(**CHAR, dtype=np.float32)
        self.docs_t = self.vectorizer.fit_transform(list(documents)).tocsr().T.tocsr()

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return (self.vectorizer.transform(list(texts)).tocsr() @ self.docs_t).toarray().astype(np.float32, copy=False)


def binary_queries(vectorizer: CountVectorizer, texts: Sequence[str]) -> sparse.csr_matrix:
    q = vectorizer.transform(list(texts)).astype(np.float32).tocsr()
    q.data[:] = 1.0
    return q


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD) per row, falling back to the standard deviation, then to 1."""
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x, axis=-1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(x - med), axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return ((x - med) / scale).astype(np.float32, copy=False)


def n_words(text: str) -> int:
    return len(_TOKEN.findall(text.lower()))
