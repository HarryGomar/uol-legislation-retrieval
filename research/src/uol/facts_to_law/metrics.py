"""Exact ranking metrics for facts-to-law v2 (PREREGISTRATION §4).

Conventions, shared with the October helpers
(``retrieval_benchmarks.refinement_suite.evaluation`` and
``orchestration.embedding_usefulness.analysis.component_table``):

- Ranks are exact and 1-based over the whole ranked list: the full catalogue,
  or the condensed list once ``exclude_idx`` candidates are removed.
- The order is score descending, then a tie key ascending. The key is the
  candidate index (the pre-registered rule). ``tie_order`` (each candidate's
  position in candidate-ID order, see ``tie_order_from_ids``) reproduces the
  October candidate-ID rule exactly.
- nDCG@10 is binary with IDCG over min(R, 10) relevant items; MRR is the
  reciprocal of the best exact rank (never truncated); Recall@k is the share
  of relevant items ranked within k.
- Component-macro means average queries within a component, then components.
"""
from __future__ import annotations

import math
import operator
from typing import Any, Mapping, Sequence

import numpy as np

METRICS = ("ndcg_at_10", "mrr", "recall_at_20", "recall_at_100")


def tie_order_from_ids(candidate_ids: Sequence[str]) -> np.ndarray:
    """Each candidate's position in candidate-ID order, for ``tie_order``.

    IDs compare as NumPy unicode strings (code-point order), as in the October
    ``np.lexsort((ids, -scores))`` rule.
    """
    ids = np.asarray([str(value) for value in candidate_ids], dtype=str)
    if np.unique(ids).size != ids.size:
        raise ValueError("candidate IDs must be unique")
    position = np.empty(ids.size, dtype=np.int64)
    position[np.argsort(ids, kind="stable")] = np.arange(ids.size)
    return position


def validate_tie_order(tie_order: np.ndarray | None, n_candidates: int) -> np.ndarray | None:
    """Return ``tie_order`` as int64 after checking it permutes 0..n-1; None stays None."""
    if tie_order is None:
        return None
    key = np.asarray(tie_order)
    if key.shape != (n_candidates,) or not np.issubdtype(key.dtype, np.integer):
        raise ValueError("tie_order must hold one integer position per candidate")
    if not np.array_equal(np.sort(key), np.arange(n_candidates)):
        raise ValueError("tie_order must be a permutation of 0..n_candidates-1")
    return key.astype(np.int64, copy=False)


def _kept(n: int, exclude_idx: np.ndarray | None) -> np.ndarray | None:
    """Ascending catalogue indices of the condensed list; None keeps every candidate."""
    if exclude_idx is None:
        return None
    values = np.asarray(exclude_idx)
    if values.dtype == bool:
        if values.shape != (n,):
            raise ValueError("a boolean exclusion mask needs one entry per candidate")
        excluded = values
    else:
        values = values.ravel()
        if values.size == 0:
            return None
        if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= n:
            raise ValueError("excluded candidate indices must be integers in [0, n_candidates)")
        excluded = np.zeros(n, dtype=bool)
        excluded[values] = True
    keep = np.flatnonzero(~excluded)
    if keep.size == 0:
        raise ValueError("every candidate is excluded")
    return None if keep.size == n else keep


def _positives(positive_idx: Sequence[int], n: int) -> np.ndarray:
    values = np.asarray(positive_idx)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("positive indices must be a flat sequence of integers")
    if values.min() < 0 or values.max() >= n:
        raise ValueError("positive index outside the candidate catalogue")
    if np.unique(values).size != values.size:
        raise ValueError("duplicate positive index")
    return values.astype(np.int64, copy=False)


def _condensed(n: int, exclude_idx: np.ndarray | None, tie_order: np.ndarray | None):
    """(kept indices, catalogue->condensed position map, condensed tie key)."""
    keep = _kept(n, exclude_idx)
    key = validate_tie_order(tie_order, n)
    if keep is None:
        return None, None, key
    position = np.full(n, -1, dtype=np.int64)
    position[keep] = np.arange(keep.size)
    return keep, position, None if key is None else key[keep]


def _row(values: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
    row = np.asarray(values)
    if keep is not None:
        row = row[keep]
    if not np.issubdtype(row.dtype, np.floating):
        row = row.astype(np.float64)
    if not np.isfinite(row).all():
        raise ValueError("non-finite score among ranked candidates")
    return row


def _ranks_in_row(row: np.ndarray, positions: np.ndarray, key: np.ndarray | None) -> np.ndarray:
    ranks = np.empty(positions.size, dtype=np.int64)
    for j, p in enumerate(positions):
        score = row[p]
        if key is None:
            ties = np.count_nonzero(row[:p] == score)
        else:
            ties = np.count_nonzero((row == score) & (key < key[p]))
        ranks[j] = 1 + np.count_nonzero(row > score) + ties
    return ranks


def _topk_in_row(row: np.ndarray, k: int, key: np.ndarray | None) -> np.ndarray:
    n = row.size
    k = min(k, n)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k < n:
        kth = np.partition(row, n - k)[n - k]
        candidates = np.flatnonzero(row >= kth)
        values = row[candidates]
        above, tied = candidates[values > kth], candidates[values == kth]
        if key is not None:
            tied = tied[np.argsort(key[tied], kind="stable")]
        selected = np.concatenate([above, tied[:k - above.size]])
    else:
        selected = np.arange(n)
    tie = selected if key is None else key[selected]
    return selected[np.lexsort((tie, -row[selected]))].astype(np.int64, copy=False)


def rank_of_positives(scores: np.ndarray, positive_idx: Sequence[int], exclude_idx: np.ndarray | None = None,
                      tie_order: np.ndarray | None = None) -> np.ndarray:
    """Exact 1-based ranks of ``positive_idx`` for one query's candidate scores.

    Excluded candidates (indices or a boolean mask) are removed from the list,
    and excluded positives are dropped from the result; the remaining ranks
    follow the input order of ``positive_idx``. Duplicate positives raise.
    """
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("scores must be one query's candidate vector")
    n = values.size
    keep, position, key = _condensed(n, exclude_idx, tie_order)
    positives = _positives(positive_idx, n)
    if position is not None:
        positives = position[positives]
        positives = positives[positives >= 0]
    return _ranks_in_row(_row(values, keep), positives, key)


def query_metrics(ranks: Sequence[int], n_relevant: int) -> dict[str, float]:
    """Binary nDCG@10, exact MRR, Recall@20 and Recall@100 for one query.

    ``ranks`` are exact 1-based ranks of relevant items and ``n_relevant`` the
    number of relevant items after exclusion; relevant items missing from
    ``ranks`` count as unretrieved. A subset of a query's ranks with its own
    ``n_relevant`` gives target-slice metrics against the unchanged list. DCG
    terms are summed in the given order (October sums in candidate-ID order).
    """
    values = [operator.index(rank) for rank in ranks]
    n_relevant = operator.index(n_relevant)
    if n_relevant < 1 or len(values) > n_relevant:
        raise ValueError("metrics need n_relevant >= 1 and at most n_relevant ranks")
    if any(rank < 1 for rank in values) or len(set(values)) != len(values):
        raise ValueError("ranks must be distinct positive integers")
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(n_relevant, 10)))
    gain = sum(1.0 / math.log2(rank + 1) for rank in values if rank <= 10)
    return {
        "ndcg_at_10": gain / ideal,
        "mrr": 1.0 / min(values) if values else 0.0,
        "recall_at_20": sum(rank <= 20 for rank in values) / n_relevant,
        "recall_at_100": sum(rank <= 100 for rank in values) / n_relevant,
    }


def evaluate_matrix(scores: np.ndarray, positives: list[list[int]], exclude_idx: np.ndarray | None = None,
                    top_k: int = 100, tie_order: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Exact per-query metrics, positive ranks and top-k for a score matrix.

    Rows are queries and columns candidates in catalogue order; ``positives[i]``
    holds row i's relevant catalogue indices. Rows are read one at a time, so a
    memmap works; evaluate full-precision scores, because float16 storage
    creates ties. Only ranked (non-excluded) scores must be finite. The top-k
    partitions on the k-th largest score and orders every candidate at or
    above it, so ties crossing the boundary are resolved exactly.

    Each row returns the ``METRICS`` (None when no relevant item remains, such
    queries must be dropped before ``component_macro``), ``n_relevant``,
    ``n_candidates`` (ranked list length), ``positive_idx`` (kept positives in
    input order), ``positive_ranks`` (same order) and ``topk`` (catalogue
    indices, best first).
    """
    matrix = np.asarray(scores)
    if matrix.ndim != 2:
        raise ValueError("scores must be a [queries, candidates] matrix")
    n_queries, n = matrix.shape
    if len(positives) != n_queries:
        raise ValueError("one positive list per query row is required")
    if operator.index(top_k) < 0:
        raise ValueError("top_k must be non-negative")
    keep, position, key = _condensed(n, exclude_idx, tie_order)
    results = []
    for i in range(n_queries):
        row = _row(matrix[i], keep)
        catalogue = _positives(positives[i], n)
        if position is None:
            condensed = catalogue
        else:
            mapped = position[catalogue]
            catalogue, condensed = catalogue[mapped >= 0], mapped[mapped >= 0]
        ranks = _ranks_in_row(row, condensed, key)
        top = _topk_in_row(row, top_k, key)
        if keep is not None:
            top = keep[top]
        metrics = query_metrics(ranks, ranks.size) if ranks.size else dict.fromkeys(METRICS)
        results.append({**metrics, "n_relevant": int(ranks.size), "n_candidates": int(row.size),
                        "positive_idx": catalogue.tolist(), "positive_ranks": ranks.tolist(),
                        "topk": top.tolist()})
    return results


def component_macro(per_query: Sequence[Mapping[str, Any]], component_ids: Sequence[str],
                    keys: Sequence[str] = METRICS) -> tuple[dict[str, float], dict[str, np.ndarray], list[str]]:
    """Component-macro means: queries averaged within components, then components.

    Returns (macro means, per-component mean arrays aligned to the sorted
    component IDs, sorted component IDs). Two systems evaluated on the same
    queries therefore give aligned per-component arrays for paired contrasts.
    """
    if len(per_query) != len(component_ids):
        raise ValueError("one component ID per query row is required")
    if not per_query:
        raise ValueError("component-macro means need at least one query")
    labels = [str(value) for value in component_ids]
    ordered = sorted(set(labels))
    index = {value: i for i, value in enumerate(ordered)}
    codes = np.fromiter((index[value] for value in labels), dtype=np.int64, count=len(labels))
    counts = np.bincount(codes, minlength=len(ordered)).astype(np.float64)
    macro, per_component = {}, {}
    for key in keys:
        raw = [row[key] for row in per_query]
        if any(value is None for value in raw):
            raise ValueError(f"{key} is undefined for queries without relevant items; drop them first")
        values = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {key} value")
        means = np.bincount(codes, weights=values, minlength=len(ordered)) / counts
        per_component[key] = means
        macro[key] = float(means.mean())
    return macro, per_component, ordered
