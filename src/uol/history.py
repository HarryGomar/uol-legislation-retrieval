"""Votes of the older *tesis*: the articles cited by the published *tesis* most similar to the case.

The memory holds 84,459 *tesis* published before 14 August 2020, outside the benchmark. For a case, the memory entries
published before it are ranked by similarity to its facts (BM25 over their heading and text, or cosine between Qwen3
vectors), and the k most similar vote for the articles they cite, each weighted by its similarity and halved every few
years of age (k and the half-life are chosen per group). Three columns per pooled article: the vote, the number of
neighbours citing it, and the log of one plus the number of articles with a larger vote.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from uol import text as T
from uol.data import Benchmark

KMAX = 100                        # neighbours kept per case; a setting votes with the first k of them


def top_neighbours(sims: np.ndarray, k: int = KMAX) -> tuple[np.ndarray, np.ndarray]:
    """The k most similar entries, by similarity then memory order."""
    part = np.argpartition(-sims, k - 1)[:k]
    order = np.lexsort((part, -sims[part]))
    idx = part[order]
    return idx, sims[idx].astype(np.float32)


class CitationMemory:
    """The older *tesis*: their dates and the articles each cites (mass one split over its articles, and binary)."""

    def __init__(self, bench: Benchmark) -> None:
        h = bench.history
        self.dates = h["date"]
        lens = np.fromiter((len(c) for c in h["cited"]), dtype=np.int64)
        rows = np.repeat(np.arange(len(lens)), lens)
        cols = np.fromiter((a for c in h["cited"] for a in c), dtype=np.int64, count=int(lens.sum()))
        shape = (len(lens), len(bench.article_ids))
        self.P = sparse.csr_matrix((np.repeat(1.0 / lens, lens).astype(np.float32), (rows, cols)), shape=shape)
        self.Pb = sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=shape)

    def columns(self, neighbours: tuple[np.ndarray, np.ndarray], date: np.datetime64, candidates: np.ndarray,
                k: int, half_life: float) -> np.ndarray:
        """The three vote columns from the first ``k`` neighbours, each weighted by its similarity, halved every
        ``half_life`` years of age, and normalised to sum to one."""
        idx, sim = neighbours[0][:k], neighbours[1][:k]
        age = (np.datetime64(date, "D") - self.dates[idx]).astype(np.float64) / 365.25
        w = np.maximum(np.nan_to_num(sim.astype(np.float64), neginf=0.0), 0.0) * np.power(0.5, age / half_life)
        w = w / w.sum() if w.sum() > 0 else np.zeros_like(w)
        n = len(self.dates)
        W = sparse.csr_matrix((w.astype(np.float32), (np.zeros(len(idx), dtype=np.int64), idx)), shape=(1, n))
        B = sparse.csr_matrix(((w > 0).astype(np.float32), (np.zeros(len(idx), dtype=np.int64), idx)), shape=(1, n))
        vote, count = (W @ self.P).toarray().ravel(), (B @ self.Pb).toarray().ravel()
        positive = np.sort(vote[vote > 0])
        larger = len(positive) - np.searchsorted(positive, vote[candidates], side="right")
        return np.stack([vote[candidates], count[candidates], np.log1p(larger.astype(np.float32))], axis=1).astype(np.float32)


class Neighbours:
    """The older *tesis* most similar to a case and published before it. ``key="lex"``: BM25 between the facts and
    their heading and text; ``key="dense"``: cosine between Qwen3 vectors (``uol.dense.Vectors.history``)."""

    def __init__(self, bench: Benchmark, key: str, vectors=None) -> None:
        self.key = key
        self.dates = bench.history["date"]
        if key == "lex":
            self.index = T.BM25(bench.history_texts)
        else:
            self.vectors_t = np.ascontiguousarray(np.asarray(vectors.history, dtype=np.float32).T)

    def __call__(self, query, date: np.datetime64) -> tuple[np.ndarray, np.ndarray]:
        """``query`` is the facts text (lex) or the case's unit vector (dense). Returns memory positions and similarities."""
        if self.key == "lex":
            sims = self.index.score([query])[0].copy()
        else:
            sims = (np.asarray(query, dtype=np.float32)[None, :] @ self.vectors_t)[0]
        sims[self.dates >= date] = -np.inf
        return top_neighbours(sims)
