"""The learned ranker: LightGBM LambdaMART over the pooled candidates of each case.

Each training case contributes its pooled cited articles and at most 300 pooled articles it does not cite, drawn at
random. The targets are the cited articles without the generic ones, after the label corrections.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRanker

FIXED = dict(objective="lambdarank", metric="ndcg", n_estimators=300, min_child_samples=40, colsample_bytree=0.5,
             reg_lambda=1.0, max_bin=127, deterministic=True, force_row_wise=True, verbose=-1)
GRID = [dict(learning_rate=lr, num_leaves=nl) for lr in (0.05, 0.1) for nl in (15, 63)]
NEGATIVES = 300
SEED = 20261004                   # + 1000 x group


def seed_of(group: int) -> int:
    return SEED + 1000 * group


def training_lists(case_of_pair: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                   negatives: int = NEGATIVES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs kept for training (cases with a pooled positive; all positives, ``negatives`` random negatives per case),
    their targets and the number of pairs per case. Pairs come ordered by case."""
    has_pos = np.zeros(case_of_pair.max() + 1, dtype=bool)
    has_pos[case_of_pair[y > 0]] = True
    idx = np.flatnonzero(has_pos[case_of_pair])
    cases, yy = case_of_pair[idx], y[idx]
    neg = np.flatnonzero(yy == 0)
    order = neg[np.lexsort((rng.random(len(neg)), cases[neg]))]
    first = np.r_[True, cases[order][1:] != cases[order][:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    kept = order[np.arange(len(order)) - start < negatives]
    keep = np.sort(np.concatenate([np.flatnonzero(yy > 0), kept]))
    _, counts = np.unique(cases[keep], return_counts=True)
    return idx[keep], yy[keep].astype(np.int32), counts


def fit(X: np.ndarray, y: np.ndarray, counts: np.ndarray, setting: dict, seed: int, threads: int = 4) -> LGBMRanker:
    model = LGBMRanker(**FIXED, n_jobs=threads, **setting, random_state=seed)
    model.fit(X, y, group=counts)
    return model


def order(model: LGBMRanker, X: np.ndarray, candidates: np.ndarray, generic: np.ndarray, k: int = 200) -> list[int]:
    """The ranked list: candidates by score (ties to the lower article number), generic articles removed."""
    ranked = candidates[np.argsort(-model.predict(X), kind="stable")]
    return [int(c) for c in ranked[~np.isin(ranked, generic)][:k]]
