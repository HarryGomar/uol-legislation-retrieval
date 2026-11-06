"""Per-case measures (PLAN_DRAFT §6) on top of the project evaluator ``embedding_push.common.evaluate``.

``ep.evaluate`` gives the frozen semantics: population admission, condensed specific lists (generic positives
dropped, generic candidates excluded) or full references, candidate-ID tie order, nDCG@10 / MRR / Recall@20 /
Recall@100, the positive ranks and the top-k ranking actually scored. This module adds, per case:

* ``success_at_1``, ``success_at_10``: a cited article at rank 1 / within the top 10;
* ``unseen_n``, ``unseen_hits_at_20``, ``unseen_recall_at_20``: positives never cited by a training component
  (``facts_to_law.prior.component_counts`` over the rows the system was trained with), and their Recall@20;
* ``law_at_1``, ``law_at_10``: an article of a cited law (instrument) at rank 1 / within the top 10;
* ``success_at_1_in_law``: restricted to the articles of the cited laws (generic candidates still excluded in the
  specific view), the best-scored article is cited.

Group-level values are component-macro means (cases averaged within their component, then components), NaN cases
dropped per measure, exactly the ``metrics.component_macro`` convention.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

BASE = ("ndcg_at_10", "mrr", "recall_at_20", "recall_at_100")
EXTRA = ("success_at_1", "success_at_10", "unseen_recall_at_20", "law_at_1", "law_at_10", "success_at_1_in_law")
MEASURES = BASE + EXTRA
PRIMARY = "ndcg_at_10"
TOP_K = 200


@lru_cache(maxsize=4)
def _instr_codes(ids_key: int) -> np.ndarray:
    from uol.embedding_push import common as ep
    ids = ep.release().candidate_ids
    names = [c.split("#", 1)[0] for c in ids]
    code = {v: i for i, v in enumerate(sorted(set(names)))}
    return np.asarray([code[v] for v in names], dtype=np.int32)


def instrument_codes() -> np.ndarray:
    """Instrument (law) code per catalogue candidate, from the candidate id (``<law pdf>#articulo=<n>``)."""
    from uol.embedding_push import common as ep
    return _instr_codes(id(ep.release()))


def unseen_mask(train_rows: Sequence[int]) -> np.ndarray:
    """Per candidate: True when no training component cites it (fold-local 'never cited in training')."""
    from uol.embedding_push import common as ep
    from uol.facts_to_law import prior as prior_mod
    from uol.final import guard
    d = ep.release()
    rows = guard.dev_rows_only(train_rows, "training rows")
    ids = [d.queries[int(r)].query_id for r in rows]
    counts = prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)
    return np.asarray(counts) == 0


def per_case(per_query: list[dict[str, Any]], rows: Sequence[int], scores: np.ndarray, *, specific: bool,
             unseen: np.ndarray | None) -> list[dict[str, Any]]:
    """Measures per evaluated case. ``per_query`` and ``rows`` are the ``ep.evaluate`` outputs for ``scores``
    (rows aligned to the admitted rows, i.e. ``res['rows']``); ``scores`` is the matching [len(rows), N] block."""
    from uol.embedding_push import common as ep
    d = ep.release()
    instr = instrument_codes()
    generic = np.zeros(len(instr), dtype=bool)
    if specific and len(d.generic_idx):
        generic[d.generic_idx] = True
    out = []
    for i, (q, r) in enumerate(zip(per_query, rows)):
        ranks = np.asarray(q["positive_ranks"], dtype=np.int64)
        pos = np.asarray(q["positive_idx"], dtype=np.int64)
        top = np.asarray(q.get("topk") or [], dtype=np.int64)
        rec = {m: (float(q[m]) if q.get(m) is not None else np.nan) for m in BASE}
        best = int(ranks.min()) if ranks.size else 10 ** 9
        rec["success_at_1"] = float(best <= 1)
        rec["success_at_10"] = float(best <= 10)
        if unseen is not None and pos.size:
            u = unseen[pos]
            rec["unseen_n"] = int(u.sum())
            rec["unseen_hits_at_20"] = int((ranks[u] <= 20).sum())
            rec["unseen_recall_at_20"] = rec["unseen_hits_at_20"] / rec["unseen_n"] if rec["unseen_n"] else np.nan
        else:
            rec["unseen_n"], rec["unseen_hits_at_20"], rec["unseen_recall_at_20"] = 0, 0, np.nan
        laws = set(instr[pos].tolist())
        rec["law_at_1"] = float(top.size > 0 and int(instr[top[0]]) in laws)
        rec["law_at_10"] = float(bool(laws & set(instr[top[:10]].tolist())))
        row = np.asarray(scores[i], dtype=np.float64)
        mask = np.isin(instr, list(laws)) & ~generic
        if mask.any() and pos.size:
            cand = np.flatnonzero(mask)
            pick = cand[int(np.argmax(row[cand]))]            # first maximum = lowest index = candidate-ID tie order
            rec["success_at_1_in_law"] = float(pick in set(pos.tolist()))
        else:
            rec["success_at_1_in_law"] = np.nan
        rec["n_relevant"] = int(q.get("n_relevant", ranks.size))
        rec["best_rank"] = best if ranks.size else -1
        rec["positive_idx"] = ",".join(str(int(p)) for p in pos)          # worked examples (thesis pack)
        rec["positive_ranks"] = ",".join(str(int(x)) for x in ranks)
        out.append(rec)
    return out


def macro(values: Sequence[float], components: Sequence[str]) -> float:
    """Component-macro mean with NaN cases dropped (NaN when no component has a value)."""
    v = np.asarray(values, dtype=np.float64)
    comps = np.asarray(components)
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan")
    ids, inv = np.unique(comps[ok], return_inverse=True)
    sums = np.bincount(inv, weights=v[ok], minlength=len(ids))
    cnt = np.bincount(inv, minlength=len(ids))
    return float((sums / cnt).mean())


def per_component(values: Sequence[float], components: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """(sorted component ids, per-component means) over finite cases (the paired-bootstrap input)."""
    v = np.asarray(values, dtype=np.float64)
    comps = np.asarray(components)
    ok = np.isfinite(v)
    ids, inv = np.unique(comps[ok], return_inverse=True)
    sums = np.bincount(inv, weights=v[ok], minlength=len(ids))
    cnt = np.bincount(inv, minlength=len(ids))
    return ids.tolist(), sums / cnt


def summary(frame) -> dict[str, Any]:
    """Component-macro means of every measure in a per-case DataFrame, plus the pooled pair-level unseen recall
    (the round-2 A definition used by the frozen numbers)."""
    comps = frame["component_id"].to_numpy()
    out = {m: macro(frame[m].to_numpy(), comps) for m in MEASURES if m in frame}
    n_u = int(frame["unseen_n"].sum()) if "unseen_n" in frame else 0
    out["unseen_pairs"] = n_u
    out["unseen_recall_at_20_pooled"] = (float(frame["unseen_hits_at_20"].sum()) / n_u) if n_u else float("nan")
    out["n_cases"] = int(len(frame))
    out["n_groups"] = int(len(set(comps.tolist())))
    return out
