"""Untrained fusions (Block B, PLAN_DRAFT §4): reciprocal-rank fusion (k = 60) and a linear robust-z mix.

Both combine registered member systems with the project operators ``facts_to_law.fusion.rrf`` and
``facts_to_law.fusion.linear_fusion`` (robust z per query, weights ``embedding_push.base.pair(w)``). RRF has no
parameter; the linear weight of the second member comes from the inner grouped folds (grid 0, .1, ..., 1).
Each member is fitted on the same outer fold first (its own inner-fold choices), so nothing leaks across folds.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from uol.final.adapters.base import Scorer, select

W_GRID = tuple(round(0.1 * i, 1) for i in range(11))


class Fusion(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        from uol.final import registry
        a = self.spec["adapter_args"]
        self.method = a["method"]
        self.members = [registry.make(m) for m in a["members"]]
        for m in self.members:
            m.prepare(pop)
        self.k = int(a.get("k", 60))
        self.grid = [{"w": w} for w in a.get("grid", W_GRID)] if self.method == "linear" else [{}]

    def _member_scores(self, rows: Sequence[int]) -> list[np.ndarray]:
        return [m.score(rows) for m in self.members]

    def _combine(self, mats: list[np.ndarray], params: dict) -> np.ndarray:
        from uol.facts_to_law import fusion
        if self.method == "rrf":
            from uol.embedding_push import common as ep
            return np.asarray(fusion.rrf(mats, k=self.k, tie_order=ep.tie_order()), dtype=np.float32)
        from uol.embedding_push import base
        return np.asarray(fusion.linear_fusion(mats, base.pair(float(params["w"])), normalize="robust_z"), dtype=np.float32)

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        members = {m.sid: m.fit(fold) for m in self.members}
        cache: dict[tuple, list[np.ndarray]] = {}

        def score_fn(params: dict, rows: np.ndarray) -> np.ndarray:
            key = (int(rows[0]), int(rows[-1]), len(rows))
            if key not in cache:
                cache.clear()
                cache[key] = self._member_scores(rows)
            return self._combine(cache[key], params)

        self.params, hist = select(self.grid, score_fn, fold, self.pop.eval_population)
        return {"params": self.params, "selection": hist, "members": members}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        return self._combine(self._member_scores(rows), self.params)

    def inputs(self) -> dict[str, Any]:
        return {m.sid: m.inputs() for m in self.members}
