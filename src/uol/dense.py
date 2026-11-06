"""The three sources built on Qwen3-Embedding-8B vectors.

    dense_qwen8b   cosine between the case vector and each article, taking the article's best window
    knn_qwen8b     the citation votes of ``knn_lex_votes`` with cosine between case vectors
    proto_qwen8b   cosine to each article's prototype: the normalised weighted sum of the vectors of the cases citing it

The vectors are inputs (``uol.encoder`` computes them for new cases): article window vectors with the article each
window belongs to, and one vector per case for each instruction variant used (``c``; fold 4's votes use ``b``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from uol.data import Benchmark
from uol.sources import KNN_DENSE, PROTO_DENSE, case_memory, fill_uncited, prototype_links, spread, votes

PROTO_VARIANT = "c"


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass
class Vectors:
    """Unit Qwen3 vectors: the article windows [windows, 4096] and the article of each; one vector per case for each
    instruction variant [cases, 4096]; and one per older tesis [84,459, 4096], in the order of ``historical/memory.jsonl``."""
    windows: np.ndarray
    owner: np.ndarray
    cases: Mapping[str, np.ndarray]
    history: np.ndarray | None = None

    @classmethod
    def load(cls, folder: str | Path) -> "Vectors":
        folder = Path(folder)
        cases = {v: np.load(folder / f"case_vectors_{v}.npy", mmap_mode="r") for v in ("b", "c")}
        history = folder / "history_vectors.npy"
        return cls(np.load(folder / "article_vectors.npy", mmap_mode="r"), np.load(folder / "article_owner.npy"), cases,
                   np.load(history, mmap_mode="r") if history.exists() else None)


class ArticleIndex:
    """Best-window cosine of a case vector against every article."""

    def __init__(self, windows: np.ndarray, owner: np.ndarray, n_articles: int) -> None:
        order = np.argsort(owner, kind="stable")
        self.windows_t = np.ascontiguousarray(np.asarray(windows, dtype=np.float32)[order].T)
        owner = owner[order]
        self.starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        self.articles, self.n = owner[self.starts], n_articles

    def score(self, vector: np.ndarray) -> np.ndarray:
        sims = np.asarray(vector, dtype=np.float32)[None, :] @ self.windows_t
        out = np.full(self.n, -np.inf, dtype=np.float32)
        out[self.articles] = np.maximum.reduceat(sims, self.starts, axis=1)[0]
        return out


class DenseSources:
    """What the dense sources learn from one group's training cases: the case memory and the prototypes."""

    def __init__(self, bench: Benchmark, group: int, vectors: Vectors, articles: ArticleIndex | None = None) -> None:
        self.n = len(bench.article_ids)
        self.articles = articles or ArticleIndex(vectors.windows, vectors.owner, self.n)
        train, pos = bench.train_rows(group), bench.positives(corrected=False)
        self.variant, self.k = KNN_DENSE[group]
        memory = case_memory(bench, train, pos)
        self.memory_t = np.ascontiguousarray(np.asarray(vectors.cases[self.variant], dtype=np.float32)[list(memory.rows)].T)
        self.memory_cited = spread(memory.cited, self.n)
        mode, by_group = PROTO_DENSE[group]
        link, self.proto_cited = prototype_links(bench, train, pos, specific_weights=mode == "specific", by_group=by_group)
        q = np.asarray(vectors.cases[PROTO_VARIANT], dtype=np.float32)[train]
        self.proto_t = np.ascontiguousarray(unit_rows(np.asarray(link[self.proto_cited] @ q, dtype=np.float32)).T)

    def score(self, case: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """``case`` maps each instruction variant to the case's unit vector."""
        c = np.asarray(case[PROTO_VARIANT], dtype=np.float32)
        knn_query = np.asarray(case[self.variant], dtype=np.float32)
        return {"dense_qwen8b": self.articles.score(c),
                "knn_qwen8b": votes(knn_query @ self.memory_t, self.memory_cited, self.k),
                "proto_qwen8b": fill_uncited(c @ self.proto_t, self.proto_cited, self.n)}
