"""Dense models adapted to the task (RUN_PLAN §3.5): ``qwen8b_adapter``, the E2 low-rank adapter on the frozen
Qwen3-Embedding-8B vectors (``round3.e2_adapter``), one per outer fold.

For outer fold k the adapter excluded set is E = {k} and its configuration the one E2 selected for k
(``e2_adapter.e2_selected``; form, rank, tau; seed 1). ``e2_adapter.ensure_adapter`` trains it on the dev rows of the
other folds (hard negatives from the nested out-of-fold ranker refits) if it is not cached, and
``e2_adapter.score_with_weights`` gives the full-catalogue cosine rows. On the thesis PC the five outer adapters of
Round 3 are cached and are loaded; on the run computer they are trained (CPU, torch) in the work folder.
Used alone (Block A). As a ranker column it would need the nested adapters of the a_p12 rule (``E = {j, k}``); that
route is ``e2_adapter.columns_for_arm`` and is not wired here.

Pair table (``adapter_args.table``, as the rankers): ``frozen`` is the embedding_push table (thesis PC only; it holds
test-set pairs), ``dev`` the development-only table the bundle builds (``ltr.dev_feature_table``). With ``dev`` the
adapter's hard negatives and its served columns index that table (``e2_adapter.pair_table`` is pointed at it), and the
nested refits ``features.nested_ltr(k, j)`` behind the hard negatives are fitted on it first (21 Oct 2026: the bundle
has no ``ep_run/fusion/features/pq.npy``).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from uol.final import context
from uol.final.adapters.base import Scorer, SystemUnavailable


class QwenAdapter(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        from uol.final.adapters import ltr
        from uol.round3 import e2_adapter as e2
        prof = context.active()
        e2.QWEN = prof.path("qwen8b")                                 # vectors of this profile (read only)
        prof.patches["uol.round3.e2_adapter.QWEN"] = str(e2.QWEN)
        self.e2 = e2
        a = self.spec.get("adapter_args", {})
        self.seed = int(a.get("seed", 1))
        self.table = ltr.use_table(a.get("table", "frozen"))
        self.F = ltr.dev_feature_table()[1] if self.table == "dev" else None

    def _tables(self):
        from uol.final.adapters import ltr
        return ltr._pair_table(self.F, self.table)

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        if not fold.role.startswith("dev_fold"):
            raise SystemUnavailable("the E2 adapter is defined for the five outer folds")
        cfg = self.e2.e2_selected()[fold.k]
        self.cfg = cfg
        E = frozenset({fold.k})
        with self._tables():
            if self.table == "dev" and not self.e2.adapter_dir("E2").joinpath(
                    f"{self.e2.adapter_key(cfg, 'raw', E, self.seed)}.npz").exists():
                from uol.round3 import features as FT
                for j in self.e2.FOLDS:                               # hard-negative sources of the training folds
                    if j != fold.k:
                        FT.nested_ltr(fold.k, j, "emb")
            self.path = self.e2.ensure_adapter("E2", cfg, "raw", E, self.seed)
        return {"params": cfg, "adapter": self.path.name, "seed": self.seed, "table": self.table}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        with self._tables():
            return np.asarray(self.e2.score_with_weights(self.path, self.cfg, "raw", rows), dtype=np.float32)
