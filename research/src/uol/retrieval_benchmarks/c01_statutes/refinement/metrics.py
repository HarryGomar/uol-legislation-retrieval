"""Component-macro retrieval metrics and paired uncertainty for refinement."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import contracts as C


METRIC_NAMES = ("ndcg_at_10", "mrr", "recall_at_20", "recall_at_100")
DELTA_TOLERANCE = 1e-12


def exact_ranks(values: np.ndarray, candidate_ids: np.ndarray, positives: set[str]) -> list[int]:
    positions = {value: index for index, value in enumerate(candidate_ids)}
    ranks: list[int] = []
    for positive in positives:
        position = positions.get(positive)
        if position is None:
            continue
        score = float(values[position])
        rank = 1 + int(np.sum(values > score))
        ties = np.sum((values == score) & (candidate_ids < candidate_ids[position]))
        if ties:
            rank += int(ties)
        ranks.append(rank)
    return ranks


def evaluate_scores(
    scores: np.ndarray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    references: Mapping[str, Sequence[str]],
    components: Mapping[str, str],
    *,
    top_k: int = C.TOP_K_RANKINGS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score full-universe matrices with component-macro aggregation."""
    ids = np.asarray(candidate_ids, dtype=str)
    if scores.shape != (len(query_ids), len(ids)):
        raise ValueError("score matrix shape mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite retrieval scores")
    details: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    keep = min(top_k, len(ids))
    for row_index, query_id in enumerate(query_ids):
        values = scores[row_index]
        positives = {str(value) for value in references[query_id]}
        ranks = exact_ranks(values, ids, positives)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(ranks), 10)))
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
        details.append(
            {
                "query_id": query_id,
                "component_id": str(components[query_id]),
                "positive_count": len(ranks),
                "ndcg_at_10": dcg / ideal if ideal else 0.0,
                "mrr": 1.0 / min(ranks) if ranks else 0.0,
                "recall_at_20": sum(rank <= 20 for rank in ranks) / len(ranks) if ranks else 0.0,
                "recall_at_100": sum(rank <= 100 for rank in ranks) / len(ranks) if ranks else 0.0,
            }
        )
        if keep == len(ids):
            order = np.lexsort((ids, -values))
        else:
            pool = np.argpartition(-values, keep - 1)[:keep]
            order = pool[np.lexsort((ids[pool], -values[pool]))]
        rankings.extend(
            {
                "query_id": query_id,
                "candidate_id": str(ids[index]),
                "rank": rank,
                "score": float(values[index]),
            }
            for rank, index in enumerate(order, 1)
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row["component_id"])].append(row)
    component_rows = [
        {
            "component_id": component_id,
            **{name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES},
        }
        for component_id, rows in sorted(grouped.items())
    ]
    metrics = {
        "queries": len(details),
        "components": len(component_rows),
        **{
            f"query_macro_{name}": float(np.mean([row[name] for row in details]))
            for name in METRIC_NAMES
        },
        **{
            f"component_macro_{name}": float(np.mean([row[name] for row in component_rows]))
            for name in METRIC_NAMES
        },
    }
    return metrics, details, rankings


def component_values(details: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    """Per-component mean of ``metric``, averaging repeated query rows.

    Accepts per-query details or rows already merged to one per component.
    An earlier version kept only the last row of each component, so paired
    comparisons fed per-query details silently dropped repeated queries.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for row in details:
        groups[str(row["component_id"])].append(float(row[metric]))
    return {component: float(np.mean(values)) for component, values in groups.items()}


def component_macro(details: Sequence[Mapping[str, Any]], metric: str) -> float:
    """Component-macro mean: query rows averaged within components, then components."""
    values = component_values(details, metric)
    if not values:
        raise ValueError("component-macro mean requires at least one row")
    return float(np.mean(list(values.values())))


def component_rows(
    details: Sequence[Mapping[str, Any]], metrics: Sequence[str] = METRIC_NAMES
) -> list[dict[str, Any]]:
    """One row per component with each metric averaged over its query rows."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in details:
        groups[str(row["component_id"])].append(row)
    return [
        {
            "component_id": component,
            **{name: float(np.mean([float(row[name]) for row in rows])) for name in metrics},
        }
        for component, rows in sorted(groups.items())
    ]


def paired_bootstrap(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    draws: int = C.BOOTSTRAP_DRAWS,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired component bootstrap of ``second - first`` for one metric."""
    left = component_values(first, metric)
    right = component_values(second, metric)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("paired bootstrap requires overlapping components")
    deltas = np.asarray([right[key] - left[key] for key in keys], dtype=np.float64)
    # Averaging identical scores in a different order leaves ~1e-16 residues;
    # they must not register as a direction of effect.
    deltas[np.abs(deltas) <= DELTA_TOLERANCE] = 0.0
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(draws, len(deltas)), replace=True).mean(axis=1)
    mean = float(deltas.mean())
    p_value = float(min(1.0, 2 * min((sampled <= 0).mean(), (sampled >= 0).mean())))
    return {
        "components": len(keys),
        "draws": draws,
        "seed": seed,
        "mean_delta": mean,
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "two_sided_bootstrap_p": max(p_value, 1.0 / draws),
    }


def seed_aware_bootstrap(
    first_by_seed: Sequence[Sequence[Mapping[str, Any]]],
    second_by_seed: Sequence[Sequence[Mapping[str, Any]]],
    metric: str,
    *,
    draws: int = C.BOOTSTRAP_DRAWS,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap of ``second - first`` that resamples components and seeds.

    Each argument holds one detail list per training seed; a frozen arm passes
    a single list.  Every draw resamples components and, independently within
    each arm, seeds, so the interval carries between-seed variation that a
    component-only bootstrap omits.  Seed-level gains pair seeds by position
    when both arms have the same number of seeds, and otherwise compare each
    seed of ``second`` with the seed mean of ``first``.
    """
    if not first_by_seed or not second_by_seed:
        raise ValueError("seed-aware bootstrap requires at least one seed per arm")
    first_values = [component_values(details, metric) for details in first_by_seed]
    second_values = [component_values(details, metric) for details in second_by_seed]
    keys = sorted(set.intersection(*(set(values) for values in first_values + second_values)))
    if not keys:
        raise ValueError("seed-aware bootstrap requires overlapping components")
    first = np.asarray([[values[key] for key in keys] for values in first_values], dtype=np.float64)
    second = np.asarray([[values[key] for key in keys] for values in second_values], dtype=np.float64)
    mean = float(second.mean(axis=0).mean() - first.mean(axis=0).mean())
    if abs(mean) <= DELTA_TOLERANCE:
        mean = 0.0
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        columns = rng.integers(0, len(keys), size=len(keys))
        first_rows = rng.integers(0, first.shape[0], size=first.shape[0])
        second_rows = rng.integers(0, second.shape[0], size=second.shape[0])
        sampled[index] = (
            second[np.ix_(second_rows, columns)].mean() - first[np.ix_(first_rows, columns)].mean()
        )
    sampled[np.abs(sampled) <= DELTA_TOLERANCE] = 0.0
    if first.shape[0] == second.shape[0]:
        seed_gains = second.mean(axis=1) - first.mean(axis=1)
    else:
        seed_gains = second.mean(axis=1) - first.mean()
    seed_gains[np.abs(seed_gains) <= DELTA_TOLERANCE] = 0.0
    p_value = float(min(1.0, 2 * min((sampled <= 0).mean(), (sampled >= 0).mean())))
    return {
        "components": len(keys),
        "seeds_first": int(first.shape[0]),
        "seeds_second": int(second.shape[0]),
        "draws": draws,
        "seed": seed,
        "mean_delta": mean,
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "two_sided_bootstrap_p": max(p_value, 1.0 / draws),
        "seed_gains": [float(value) for value in seed_gains],
        "positive_seeds": int((seed_gains > 0).sum()),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values keyed like the input."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, (total - index) * value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in p_values}


def robust_z_rows(values: np.ndarray) -> np.ndarray:
    """Per-query robust z normalization used by the frozen fusion operator."""
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return ((values - median) / scale).astype(np.float32, copy=False)


def combine_channels(whole: np.ndarray, clause: np.ndarray, weight: float) -> np.ndarray:
    """Robust-z each channel then apply the convex clause weight."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("convex clause weight outside [0, 1]")
    return ((1.0 - weight) * robust_z_rows(whole) + weight * robust_z_rows(clause)).astype(
        np.float32, copy=False
    )
