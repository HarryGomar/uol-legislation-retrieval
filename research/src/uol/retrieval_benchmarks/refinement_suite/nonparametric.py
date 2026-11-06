"""Fitting-only nonparametric supervision and a low-rank linear control.

These functions never read a release or a label file. The runner passes an
explicit component-designated fitting population and candidate positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def positive_matrix(positives: Sequence[Sequence[int]], candidates: int) -> sparse.csr_matrix:
    """Each independent fitting query contributes total mass one."""
    rows, cols, weights = [], [], []
    for i, values in enumerate(positives):
        unique = sorted(set(map(int, values)))
        if not unique or min(unique) < 0 or max(unique) >= candidates:
            raise ValueError("invalid fitting positives")
        rows.extend([i] * len(unique)); cols.extend(unique)
        weights.extend([1.0 / len(unique)] * len(unique))
    return sparse.csr_matrix((weights, (rows, cols)), shape=(len(positives), candidates), dtype=np.float32)


def neighbor_votes(queries: np.ndarray, fitting_queries: np.ndarray,
                   positives: Sequence[Sequence[int]], candidates: int,
                   *, k: int = 8, temperature: float = .05) -> np.ndarray:
    """Softmax-weighted votes from k similar fitting components.

    No candidate is removed. The runner adds this channel to the same
    full-catalogue text baseline, so unseen targets retain their text scores.
    """
    if not 1 <= k <= len(fitting_queries) or temperature <= 0:
        raise ValueError("invalid neighbor vote configuration")
    similarity = normalize(queries) @ normalize(fitting_queries).T
    # Stable fitting row order resolves ties reproducibly.
    indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(similarity, indices, axis=1) / temperature
    weights = np.exp(selected - selected.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    neighbors = sparse.csr_matrix((weights.ravel(),
                    (np.repeat(np.arange(len(queries)), k), indices.ravel())),
                    shape=(len(queries), len(fitting_queries)), dtype=np.float32)
    return (neighbors @ positive_matrix(positives, candidates)).toarray().astype(np.float32)


@dataclass
class ArticlePrototypes:
    positions: np.ndarray
    vectors: np.ndarray
    component_counts: np.ndarray


def fit_prototypes(fitting_queries: np.ndarray, positives: Sequence[Sequence[int]],
                   candidates: int) -> ArticlePrototypes:
    """One query-vector centroid per observed target; no article ID parameter."""
    matrix = positive_matrix(positives, candidates)
    counts = np.asarray((matrix > 0).sum(axis=0)).ravel()
    positions = np.flatnonzero(counts)
    vectors = normalize(np.asarray(matrix[:, positions].T @ normalize(fitting_queries)))
    return ArticlePrototypes(positions, vectors, counts[positions])


def prototype_scores(queries: np.ndarray, articles: np.ndarray,
                     prototypes: ArticlePrototypes, *, weight: float,
                     frozen_scores: np.ndarray) -> np.ndarray:
    if not 0 <= weight <= 1 or frozen_scores.shape != (len(queries), len(articles)):
        raise ValueError("invalid prototype mixture")
    result = np.array(frozen_scores, dtype=np.float32, copy=True)
    positions = prototypes.positions
    vectors = normalize((1 - weight) * articles[positions] + weight * prototypes.vectors)
    result[:, positions] = normalize(queries) @ vectors.T
    # Unseen article scores remain bitwise equal to the supplied frozen scores.
    return result


@dataclass
class LinearQueryMap:
    projection: np.ndarray
    coefficients: np.ndarray
    ridge: float
    delta: float
    seed: int

    def transform(self, queries: np.ndarray) -> np.ndarray:
        raw = normalize(queries)
        residual = (raw @ self.projection) @ self.coefficients
        norm = np.linalg.norm(residual, axis=1, keepdims=True)
        residual *= np.minimum(1, self.delta / np.maximum(norm, 1e-12))
        return normalize(raw + residual)


def fit_linear_query_map(fitting_queries: np.ndarray, articles: np.ndarray,
                         positives: Sequence[Sequence[int]], *, rank: int = 64,
                         ridge: float = 1.0, delta: float = .25,
                         seed: int = 2027, shuffle_targets: bool = False) -> LinearQueryMap:
    """Ridge prediction of target-centroid residual through a fixed projection.

    This is a supervised linear control, not a parameter-matched nonlinear
    adapter: only rank*dimension output coefficients are fitted. The random
    projection is fixed and its storage cost is reported separately.
    """
    if ridge <= 0 or delta <= 0 or not 0 < rank <= articles.shape[1]:
        raise ValueError("invalid linear map configuration")
    raw = normalize(fitting_queries)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((articles.shape[1], rank)).astype(np.float32)
    projection /= np.sqrt(rank)
    target = normalize(positive_matrix(positives, len(articles)) @ articles)
    if shuffle_targets:
        target = target[rng.permutation(len(target))]
    x = raw @ projection
    coefficients = np.linalg.solve(x.T @ x + ridge * np.eye(rank),
                                    x.T @ (target - raw)).astype(np.float32)
    return LinearQueryMap(projection, coefficients, ridge, delta, seed)


def group_prior(frequency: np.ndarray, parents: Sequence[str]) -> np.ndarray:
    """Law count is the sum of fitting article component counts, explicitly."""
    if len(frequency) != len(parents):
        raise ValueError("parent prior length mismatch")
    totals: dict[str, float] = {}
    for count, parent in zip(frequency, parents):
        totals[parent] = totals.get(parent, 0.0) + float(count)
    return np.log1p(np.asarray([totals[p] for p in parents], dtype=np.float32))
