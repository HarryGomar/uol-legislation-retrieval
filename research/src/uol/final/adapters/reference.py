"""Reference systems (PLAN_DRAFT §3): random ranking and popularity of past citations.

* ``random``: uniform scores from ``numpy.random.default_rng(20261004 + release-row-independent query hash)``, so
  a case gets the same random ranking in every population and profile.
* ``popularity``: ``facts_to_law.prior.fitting_prior`` (log1p of the number of distinct training components citing
  each article) from the outer fold's training rows only; the same vector for every scored case.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from uol.final.adapters.base import Scorer

SEED = 20261004


class RandomScorer(Scorer):
    def score(self, rows: Sequence[int]) -> np.ndarray:
        from uol.embedding_push import common as ep
        d = ep.release()
        rows = self.rows(rows)
        out = np.empty((len(rows), len(d.candidate_ids)), dtype=np.float32)
        for i, r in enumerate(rows):
            h = int(hashlib.sha256(d.queries[int(r)].query_id.encode()).hexdigest()[:12], 16)
            out[i] = np.random.default_rng(SEED + h).random(len(d.candidate_ids), dtype=np.float32)
        return out


class Popularity(Scorer):
    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        from uol.embedding_push import common as ep
        from uol.facts_to_law import prior as prior_mod
        d = ep.release()
        train = self.rows(fold.train_rows, "popularity training rows")
        ids = [d.queries[int(r)].query_id for r in train]
        self.vec = prior_mod.fitting_prior(d.references, d.component_by_query, ids, len(d.candidate_ids),
                                           d.candidate_index).astype(np.float32)
        return {"train_rows": int(len(train)), "nonzero": int((self.vec > 0).sum())}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        return np.broadcast_to(self.vec, (len(rows), len(self.vec))).copy()
