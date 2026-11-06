"""Scorer protocol shared by every adapter, and the inner-fold selection helper.

A scorer is built from one ``systems.json`` entry. The harness calls, per population:

    prepare(pop)          once: load artefacts, verify hashes, build label-free indexes
    fit(fold) -> dict     per outer fold: tune on fold.tune_rows / fold.inner, fit on fold.train_rows only
    score(rows) -> [len(rows), N] float32 scores for rows of that fold (full catalogue columns)

Rows are release rows of the active profile and pass ``guard.dev_rows_only`` before any score or label is read.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from uol.final import guard


class SystemPending(RuntimeError):
    """The system's artefact (e.g. GPU vectors) does not exist yet; the harness records it as pending."""


class SystemUnavailable(RuntimeError):
    """The system cannot be scored on this population or profile (e.g. LLM stage outside DEV)."""


class Scorer:
    needs_full_rows = True

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.sid = spec["id"]
        self.params: dict[str, Any] = {}
        self.fold = None

    # ------------------------------------------------------------------ protocol
    def prepare(self, pop) -> None:
        self.pop = pop

    def fit(self, fold) -> dict[str, Any]:
        self.fold = fold
        return {}

    def score(self, rows: Sequence[int]) -> np.ndarray:            # pragma: no cover - abstract
        raise NotImplementedError

    def inputs(self) -> dict[str, Any]:
        """Input artefacts and their hashes for the receipt."""
        return {}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def rows(rows: Sequence[int], what: str = "scored rows") -> np.ndarray:
        return guard.dev_rows_only(rows, what)

    def score_blocks(self, rows: Sequence[int], block: int = 256) -> np.ndarray:
        rows = self.rows(rows)
        from uol.embedding_push import common as ep
        out = np.empty((len(rows), len(ep.release().candidate_ids)), dtype=np.float32)
        for s in range(0, len(rows), block):
            out[s:s + block] = self.score(rows[s:s + block])
        return out


def macro_ndcg(scores: np.ndarray, rows: Sequence[int], population: str) -> float:
    """Component-macro nDCG@10 of ``scores`` on ``rows`` under the project evaluator (``ep.evaluate``)."""
    from uol.embedding_push import common as ep
    rows = guard.dev_rows_only(rows, "tuning rows")
    return float(ep.evaluate(np.asarray(scores), rows, population, top_k=0)["macro"]["ndcg_at_10"])


def select(grid: Sequence[dict[str, Any]], score_fn: Callable[[dict[str, Any], np.ndarray], np.ndarray], fold,
           population: str, *, block: int = 512) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick a configuration on the inner grouped folds of ``fold`` (mean over inner folds of the component-macro
    nDCG@10 on each inner evaluation part; ties go to the earlier grid entry, tolerance 1e-12).

    ``score_fn(params, rows)`` returns [len(rows), N] scores. For label-free configurations (an LSA dimension, an
    encoder's query variant) the inner training part is unused; fitted ones receive it through ``fold.inner``.
    A single-entry grid is returned without scoring."""
    if len(grid) == 1:
        return dict(grid[0]), [{"params": dict(grid[0]), "objective": None, "note": "single configuration"}]
    from uol.embedding_push import common as ep
    from uol.facts_to_law import metrics
    history = []
    for params in grid:
        vals = []
        for _, ev_rows in fold.inner:
            ev_rows = guard.dev_rows_only(ev_rows, "inner evaluation rows")
            per_query, comps = [], []
            for s in range(0, len(ev_rows), block):
                r = ev_rows[s:s + block]
                res = ep.evaluate(score_fn(params, r), r, population, top_k=0)
                per_query += res["per_query"]
                comps += [ep.release().queries[int(x)].component_id for x in res["rows"]]
            macro, _, _ = metrics.component_macro(per_query, comps)
            vals.append(float(macro["ndcg_at_10"]))
        history.append({"params": dict(params), "inner": vals, "objective": float(np.mean(vals)) if vals else float("nan")})
    objs = [h["objective"] for h in history]
    best = max(objs)
    pick = min(i for i, o in enumerate(objs) if o >= best - 1e-12)
    return dict(grid[pick]), history
