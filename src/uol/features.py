"""From source scores to the ranker's input: the candidate pool and three columns per source.

The pool is the union of each source's 200 best articles. For a pooled article each source contributes its robust z-score,
the log of one plus the number of articles that score higher, and the z-score of the best article of the same law.
Three columns describe the pair (article length, how often training cases cite the article's law, facts length), and
three come from the past-citation votes (``uol.history``).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

TOP = 200


def sanitize(x: np.ndarray) -> np.ndarray:
    """Non-finite scores become the lowest finite score minus one."""
    x = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    finite = x[~bad]
    out = x.copy()
    out[bad] = np.float32((finite.min() if finite.size else 0.0) - 1.0)
    return out


def top(x: np.ndarray, k: int = TOP) -> np.ndarray:
    """The k best articles; ties at the boundary go to the lower article number."""
    kth = np.partition(x, len(x) - k)[len(x) - k]
    above = np.flatnonzero(x > kth)
    return np.concatenate([above, np.flatnonzero(x == kth)[: k - len(above)]])


def pool(scores: Mapping[str, np.ndarray], sources: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([top(sanitize(scores[s])) for s in sources])).astype(np.int64)


def source_columns(x: np.ndarray, candidates: np.ndarray, law: np.ndarray) -> np.ndarray:
    """[len(candidates), 3]: robust z, log(1 + number scoring higher), robust z of the law's best article."""
    x = sanitize(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    std = x.std()
    scale = np.float32(1.4826) * mad                   # float32 arithmetic, as the stored features
    scale = np.float32(scale if scale > 1e-8 else (std if std > 1e-8 else 1.0))
    raw = x[candidates]
    higher = len(x) - np.searchsorted(np.sort(x), raw, side="right")
    best_of_law = np.full(law.max() + 1, -np.inf, dtype=np.float32)
    np.maximum.at(best_of_law, law, x)
    return np.stack([(raw - med) / scale, np.log((higher + 1).astype(np.float32)),
                     (best_of_law[law[candidates]] - med) / scale], axis=1).astype(np.float32)


def pair_columns(article_lengths: np.ndarray, law_prior: np.ndarray, facts: str, candidates: np.ndarray) -> np.ndarray:
    """log(1 + article length), the law's citation prior, log(1 + facts length); lengths in characters."""
    return np.stack([article_lengths[candidates], law_prior[candidates],
                     np.full(len(candidates), np.log1p(np.float32(len(facts))))], axis=1).astype(np.float32)
