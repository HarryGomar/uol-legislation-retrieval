"""Fitting-fold citation popularity: frequency prior and popularity bands.

A candidate's count is the number of distinct fitting components whose union of
positives contains it, the October definition
(``uol.orchestration.embedding_usefulness.analysis.component_frequency``).
Only the references of the supplied fitting query ids are read.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .spec import FREQUENT, RARE, UNSEEN

RARE_MAX = 4  # 0 unseen, 1-4 rare, 5+ frequent fitting components


def component_counts(references_by_query: Mapping[str, Sequence[str]], component_by_query: Mapping[str, str],
                     fit_query_ids: Iterable[str], n_candidates: int, candidate_index: Mapping[str, int]) -> np.ndarray:
    """Int64 ``[n_c]``: distinct fitting components citing each candidate."""
    groups: dict[str, set[int]] = defaultdict(set)
    for query_id in fit_query_ids:
        component = component_by_query[query_id]
        for candidate_id in references_by_query[query_id]:
            index = candidate_index.get(candidate_id)
            if index is None or not 0 <= index < n_candidates:
                raise ValueError(f"fitting reference outside the candidate catalogue: {candidate_id}")
            groups[component].add(index)
    counts = np.zeros(n_candidates, dtype=np.int64)
    for indices in groups.values():
        counts[np.fromiter(indices, dtype=np.int64, count=len(indices))] += 1
    return counts


def fitting_prior(references_by_query: Mapping[str, Sequence[str]], component_by_query: Mapping[str, str],
                  fit_query_ids: Iterable[str], n_candidates: int, candidate_index: Mapping[str, int],
                  *, scale: str | None = None) -> np.ndarray:
    """Float32 ``[n_c]`` prior ``log1p(count)``.

    ``scale='max'`` divides by ``max(prior.max(), 1)``, the October scaling.
    """
    counts = component_counts(references_by_query, component_by_query, fit_query_ids, n_candidates, candidate_index)
    prior = np.log1p(counts).astype(np.float32)
    if scale == "max":
        prior /= max(float(prior.max()), 1.0)
    elif scale is not None:
        raise ValueError(f"unknown prior scale: {scale}")
    return prior


def band_of(count: int) -> str:
    if count < 0:
        raise ValueError("a component count cannot be negative")
    return UNSEEN if count == 0 else RARE if count <= RARE_MAX else FREQUENT


def bands_from_counts(counts: np.ndarray) -> dict[int, str]:
    """Candidate index -> ``unseen`` / ``rare`` / ``frequent`` for every candidate."""
    values = np.asarray(counts)
    labels = np.where(values == 0, UNSEEN, np.where(values <= RARE_MAX, RARE, FREQUENT))
    if (values < 0).any():
        raise ValueError("a component count cannot be negative")
    return {int(i): str(label) for i, label in enumerate(labels)}


def popularity_bands(references_by_query: Mapping[str, Sequence[str]], component_by_query: Mapping[str, str],
                     fit_query_ids: Iterable[str], n_candidates: int, candidate_index: Mapping[str, int]) -> dict[int, str]:
    """Per outer fold popularity band of every candidate, from fitting components only."""
    return bands_from_counts(component_counts(references_by_query, component_by_query, fit_query_ids, n_candidates, candidate_index))
