"""Scoring: nDCG@10 against the cited articles, averaged within each group of cases and then across groups.

Gains are binary. The targets are the cited articles without the generic ones (corrected labels); generic articles and
repeats are removed from a list before its first ten are scored. Two systems are compared by a paired bootstrap over
groups (10,000 resamples). A case left without targets by the corrections is not scored.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from uol.data import Benchmark

DRAWS, SEED, CHUNK = 10_000, 20261004, 256
DISCOUNT = 1.0 / np.log2(np.arange(2, 12))


def ndcg_at_10(ranked: Sequence[int], gold: set[int], generic: set[int]) -> float:
    shown = [c for c in dict.fromkeys(ranked) if c not in generic][:10]
    ideal = DISCOUNT[:min(len(gold), 10)].sum()
    return float(sum(DISCOUNT[k] for k, c in enumerate(shown) if c in gold) / ideal) if ideal > 0 else 0.0


def per_case(bench: Benchmark, lists: Mapping[int, Sequence[int]]) -> dict[int, float]:
    gold, generic = bench.specific(corrected=True), set(bench.generic.tolist())
    return {r: ndcg_at_10(lst, set(gold.get(r, [])), generic) for r, lst in lists.items()}


def by_group(bench: Benchmark, scores: Mapping[int, float]) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for r, v in scores.items():
        sums.setdefault(bench.components[r], []).append(v)
    return {c: float(np.mean(v)) for c, v in sorted(sums.items())}


def scored_rows(bench: Benchmark, population: str = "DEV") -> list[int]:
    gold = bench.specific(corrected=True)
    return [int(r) for r in bench.evaluation_rows(population) if gold.get(int(r))]


def score(bench: Benchmark, lists: Mapping[int, Sequence[int]], population: str = "DEV") -> float:
    """Group-macro nDCG@10 over the population's scored cases (a case without a list scores 0)."""
    cases = per_case(bench, {r: lists.get(r, []) for r in scored_rows(bench, population)})
    return float(np.mean(list(by_group(bench, cases).values())))


def bootstrap_means(values: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for s in range(0, draws, CHUNK):
        n = min(CHUNK, draws - s)
        out[s:s + n] = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return out


def compare(bench: Benchmark, a: Mapping[int, Sequence[int]], b: Mapping[int, Sequence[int]],
            population: str = "DEV") -> dict[str, float]:
    """Mean difference a - b over groups, with its 95% bootstrap interval."""
    rows = scored_rows(bench, population)
    ga = by_group(bench, per_case(bench, {r: a.get(r, []) for r in rows}))
    gb = by_group(bench, per_case(bench, {r: b.get(r, []) for r in rows}))
    diff = np.asarray([ga[c] - gb[c] for c in ga])
    low, high = np.quantile(bootstrap_means(diff), [0.025, 0.975])
    return {"a": float(np.mean(list(ga.values()))), "b": float(np.mean(list(gb.values()))),
            "difference": float(diff.mean()), "low": float(low), "high": float(high), "groups": len(diff)}
