"""Supervised citation memory: kNN voting over fitting components.

The memory holds one entry per fitting component: the text (or vector) of its
representative query (lowest query id) and the union of positives over all of
the component's fitting queries. October ``embedding_usefulness.runner
._component_fitting`` used only the representative's own positives; v2 uses the
component union. Votes follow ``refinement_suite.nonparametric.neighbor_votes``:
each memory entry spreads mass one uniformly over its positives, and the top-k
entries by query similarity vote with normalized weights. Evaluation labels are
never an input; evaluation and memory components must be disjoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from uol.retrieval_benchmarks.refinement_suite.nonparametric import normalize, positive_matrix

from .lexical import LexicalIndex

WEIGHTINGS = ("sim", "softmax", "uniform", "rank")


@dataclass(frozen=True)
class Memory:
    """Fitting-component memory in deterministic (component id) order."""

    query_ids: tuple[str, ...]            # representative query per component
    components: tuple[str, ...]
    positives: tuple[tuple[int, ...], ...]  # union of fitting positives, candidate indices


def build_memory(fit_query_ids: Iterable[str], component_by_query: Mapping[str, str],
                 references_by_query: Mapping[str, Sequence[str]], candidate_index: Mapping[str, int]) -> Memory:
    """One entry per fitting component: lowest query id and the component's union of positives."""
    representative: dict[str, str] = {}
    union: dict[str, set[int]] = {}
    for query_id in fit_query_ids:
        component = component_by_query[query_id]
        if component not in representative or query_id < representative[component]:
            representative[component] = query_id
        bucket = union.setdefault(component, set())
        for candidate_id in references_by_query[query_id]:
            if candidate_id not in candidate_index:
                raise ValueError(f"fitting reference outside the candidate catalogue: {candidate_id}")
            bucket.add(candidate_index[candidate_id])
    components = sorted(c for c in union if union[c])
    return Memory(tuple(representative[c] for c in components), tuple(components),
                  tuple(tuple(sorted(union[c])) for c in components))


def query_similarity_lexical(eval_texts: Sequence[str], fit_texts: Sequence[str], *, kind: str = "bm25",
                             params: Mapping[str, Any] | None = None) -> np.ndarray:
    """Float32 ``[n_eval, n_fit]``: lexical scores of each evaluation text against fitting texts as documents."""
    return LexicalIndex(fit_texts, kind=kind, params=params).score(eval_texts)


def query_similarity_dense(eval_vecs: np.ndarray, fit_vecs: np.ndarray) -> np.ndarray:
    """Float32 ``[n_eval, n_fit]`` cosine similarity."""
    eval_vecs, fit_vecs = np.asarray(eval_vecs), np.asarray(fit_vecs)
    if eval_vecs.ndim != 2 or fit_vecs.ndim != 2 or eval_vecs.shape[1] != fit_vecs.shape[1]:
        raise ValueError("query vectors must be 2-D with equal width")
    return (normalize(eval_vecs) @ normalize(fit_vecs).T).astype(np.float32, copy=False)


def _weights(top: np.ndarray, weighting: str, temperature: float) -> np.ndarray:
    k = top.shape[1]
    if weighting == "softmax":
        if temperature <= 0:
            raise ValueError("softmax voting needs a positive temperature")
        z = top / temperature
        w = np.exp(z - z.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)
    if weighting == "sim":
        w = np.maximum(top, 0.0)
        total = w.sum(axis=1, keepdims=True)
        return np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    if weighting == "uniform":
        return np.full(top.shape, 1.0 / k)
    if weighting == "rank":
        w = 1.0 / np.arange(1, k + 1, dtype=np.float64)
        return np.broadcast_to(w / w.sum(), top.shape)
    raise ValueError(f"unknown kNN weighting: {weighting}")


def knn_votes(sim: np.ndarray, fit_positive_idx: Sequence[Sequence[int]], k: int, n_candidates: int,
              weighting: str = "sim", *, temperature: float = 0.05, eval_components: Sequence[str] | None = None,
              fit_components: Sequence[str] | None = None, dense: bool = True) -> np.ndarray | sparse.csr_matrix:
    """Citation votes ``[n_eval, n_c]`` from the k most similar memory entries.

    Ties in similarity keep memory order (stable sort). ``sim`` weights are the
    clipped non-negative similarities normalized per query (a query with no
    positive similarity casts no votes); ``softmax`` is the October rule
    ``softmax(sim / temperature)``; ``uniform`` and ``rank`` (1/rank) ignore the
    similarity scale. Each query's votes sum to one unless it casts none.
    """
    sim = np.asarray(sim)
    if sim.ndim != 2 or sim.shape[1] != len(fit_positive_idx) or not np.isfinite(sim).all():
        raise ValueError("invalid query similarity or memory labels")
    if k < 1 or sim.shape[1] < 1:
        raise ValueError("kNN voting needs k >= 1 and a non-empty memory")
    if (eval_components is None) != (fit_components is None):
        raise ValueError("pass evaluation and memory components together")
    if eval_components is not None:
        if len(eval_components) != sim.shape[0] or len(fit_components) != sim.shape[1]:
            raise ValueError("component lists do not match the similarity shape")
        shared = set(eval_components) & set(fit_components)
        if shared:
            raise AssertionError(f"evaluation component in the kNN memory: {sorted(shared)[:3]}")
    k = min(int(k), sim.shape[1])
    index = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(sim, index, axis=1).astype(np.float64)
    weights = _weights(top, weighting, temperature)
    rows = np.repeat(np.arange(sim.shape[0]), k)
    neighbours = sparse.csr_matrix((np.asarray(weights, dtype=np.float32).ravel(), (rows, index.ravel())),
                                   shape=sim.shape, dtype=np.float32)
    votes = (neighbours @ positive_matrix(fit_positive_idx, n_candidates)).tocsr()
    return votes.toarray().astype(np.float32, copy=False) if dense else votes
