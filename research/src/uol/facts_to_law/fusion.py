"""Row-wise score normalization and fusion for facts-to-law v2 systems.

Rows are queries and columns candidates in catalogue order; a 1-D input is one
row. Every function treats rows independently (any batch of rows gives the
same values), never mutates its inputs and rejects non-finite scores.

- ``robust_z`` is the October operator (``c01_statutes.agentic.core.robust_z``,
  identical to ``refinement.metrics.robust_z_rows``): (x - median) /
  (1.4826 * MAD) in float32. When that scale is at most 1e-8 it falls back to
  the row standard deviation, then to 1. The MAD form is used instead of an
  IQR form so that robust-z fusion is the same operator as in October; the
  fallback keeps zero-inflated rows (sparse BM25, kNN votes, priors) finite
  instead of dividing by a tiny epsilon.
- ``rank_channel`` is the October rank channel
  (``embedding_usefulness.analysis.rank_channel``): (average rank - 1) / (N - 1),
  so the highest score maps to 1.
- ``rrf`` sums w / (k + rank) over exact full-list ranks in float64; the
  default ranks are a total order with ties by candidate index ascending.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .metrics import validate_tie_order

NORMALIZATIONS = ("robust_z", "rank", "none")
TIE_RULES = ("stable", "average", "min")


def _scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim not in (1, 2) or values.shape[-1] == 0:
        raise ValueError("scores must be a non-empty candidate vector or [queries, candidates] matrix")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite score")
    return values


def _weights(weights: Sequence[float], count: int) -> list[float]:
    values = [float(weight) for weight in weights]
    if len(values) != count:
        raise ValueError("one weight per channel is required")
    if not all(math.isfinite(weight) and weight >= 0 for weight in values) or not any(values):
        raise ValueError("weights must be finite, non-negative and not all zero")
    return values


def _channels(matrices: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not matrices:
        raise ValueError("at least one channel is required")
    values = [_scores(matrix) for matrix in matrices]
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("fused channels must share one shape")
    return values


def robust_z(scores: np.ndarray) -> np.ndarray:
    """October robust z per row: (x - median) / (1.4826 * MAD), with std then 1 fallbacks."""
    values = np.asarray(_scores(scores), dtype=np.float32)
    median = np.median(values, axis=-1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=-1, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=-1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return ((values - median) / scale).astype(np.float32, copy=False)


def rank_channel(scores: np.ndarray) -> np.ndarray:
    """October rank channel per row: (average rank - 1) / (N - 1), float32; best score maps to 1."""
    from scipy.stats import rankdata

    values = _scores(scores)
    n = values.shape[-1]
    if n < 2:
        raise ValueError("a rank channel needs at least two candidates")
    return ((rankdata(values, method="average", axis=-1) - 1) / (n - 1)).astype(np.float32)


def normalize_rows(scores: np.ndarray, method: str = "robust_z") -> np.ndarray:
    """Normalize each row with 'robust_z', 'rank' or 'none' (a float32 copy)."""
    if method == "robust_z":
        return robust_z(scores)
    if method == "rank":
        return rank_channel(scores)
    if method == "none":
        return np.array(_scores(scores), dtype=np.float32, copy=True)
    raise ValueError(f"unknown normalization {method!r}; expected one of {NORMALIZATIONS}")


def linear_fusion(matrices: list[np.ndarray], weights: Sequence[float], normalize: str = "robust_z") -> np.ndarray:
    """Weighted sum of normalized channels, sum_i w_i * normalize(M_i), in float32.

    Terms are added in channel order, so ``normalize='rank'`` with weights
    (w, 1 - w) reproduces October ``w * rank(d) + (1 - w) * rank(l)`` bit for
    bit. Weights must be non-negative and not all zero; zero-weight channels
    are validated but not normalized.
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization {normalize!r}; expected one of {NORMALIZATIONS}")
    channels = _channels(matrices)
    fused = None
    for channel, weight in zip(channels, _weights(weights, len(channels))):
        if weight == 0:
            continue
        term = np.float32(weight) * normalize_rows(channel, normalize)
        fused = term if fused is None else np.add(fused, term, out=fused)
    return fused


def add_prior(scores: np.ndarray, prior_vector: np.ndarray, weight: float, normalize: str = "robust_z") -> np.ndarray:
    """normalize(scores) + weight * normalize(prior) in float32, the prior broadcast to every row.

    The prior is one query-independent value per candidate, normalized over the
    catalogue as a single row. The base scores are normalized as one channel,
    so weight 0 preserves their ranking. With ``normalize='none'`` this is
    October ``base + w * p`` (a rank-channel mixture plus the max-scaled prior).
    """
    values = _scores(scores)
    prior = _scores(prior_vector)
    if prior.ndim != 1 or prior.shape[0] != values.shape[-1]:
        raise ValueError("the prior needs one value per candidate")
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("the prior weight must be finite and non-negative")
    return normalize_rows(values, normalize) + np.float32(weight) * normalize_rows(prior, normalize)


def full_ranks(scores: np.ndarray, ties: str = "stable", tie_order: np.ndarray | None = None) -> np.ndarray:
    """Exact 1-based rank of every candidate by score descending, row by row.

    ``ties='stable'`` gives a total order (int64) with ties by candidate index
    ascending, or by ``tie_order`` as in ``metrics``; 'min' gives every tied
    candidate the best rank of its block (int64); 'average' the block's mean
    rank (float64).
    """
    values = _scores(scores)
    rows = values.reshape(1, -1) if values.ndim == 1 else values
    n_rows, n = rows.shape
    if ties == "stable":
        key = validate_tie_order(tie_order, n)
        if key is None:
            order = np.argsort(-rows, axis=1, kind="stable")
        else:
            by_key = np.argsort(key)
            order = by_key[np.argsort(-rows[:, by_key], axis=1, kind="stable")]
        ranks = np.empty((n_rows, n), dtype=np.int64)
        positions = np.broadcast_to(np.arange(1, n + 1, dtype=np.int64), (n_rows, n))
        np.put_along_axis(ranks, order, positions, axis=1)
    elif ties in ("min", "average"):
        if tie_order is not None:
            raise ValueError("tie_order applies only to ties='stable'")
        ascending = np.sort(rows, axis=1)
        ranks = np.empty((n_rows, n), dtype=np.int64 if ties == "min" else np.float64)
        for i in range(n_rows):
            at_most = np.searchsorted(ascending[i], rows[i], side="right")
            if ties == "min":
                ranks[i] = n - at_most + 1
            else:
                below = np.searchsorted(ascending[i], rows[i], side="left")
                ranks[i] = (2 * n + 1 - at_most - below) / 2
    else:
        raise ValueError(f"unknown tie rule {ties!r}; expected one of {TIE_RULES}")
    return ranks.reshape(values.shape)


def rrf_from_ranks(ranks: list[np.ndarray], k: float = 60, weights: Sequence[float] | None = None) -> np.ndarray:
    """Reciprocal-rank fusion sum_i w_i / (k + rank_i) of precomputed full ranks, float64."""
    if not ranks:
        raise ValueError("at least one rank channel is required")
    k = float(k)
    if not math.isfinite(k) or k < 0:
        raise ValueError("the RRF constant must be finite and non-negative")
    values = _weights([1.0] * len(ranks) if weights is None else weights, len(ranks))
    shape = np.shape(ranks[0])
    fused = np.zeros(shape, dtype=np.float64)
    for rank, weight in zip(ranks, values):
        rank = np.asarray(rank, dtype=np.float64)
        if rank.shape != shape or not (rank >= 1).all():
            raise ValueError("rank channels must share one shape and hold ranks >= 1")
        if weight:
            fused += weight / (k + rank)
    return fused


def rrf(matrices: list[np.ndarray], k: int = 60, weights: Sequence[float] | None = None,
        ties: str = "stable", tie_order: np.ndarray | None = None) -> np.ndarray:
    """Reciprocal-rank fusion over exact full-list ranks of each score matrix, float64.

    Ranks follow ``full_ranks`` (default: stable ties by candidate index).
    Zero-weight channels are validated but not ranked.
    """
    channels = _channels(matrices)
    values = _weights([1.0] * len(channels) if weights is None else weights, len(channels))
    used = [(channel, weight) for channel, weight in zip(channels, values) if weight]
    return rrf_from_ranks([full_ranks(channel, ties, tie_order) for channel, _ in used], k,
                          [weight for _, weight in used])
