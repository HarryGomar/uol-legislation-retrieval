"""F4 provenance share (ANALYSIS_PLAN_C §13b, D109; 6.7; requirement F4 of D92).

For each DEV case and each of the integrated method's (S_full_ext) first ten articles, the article is supported when at
least one past tesis among the case's nearest neighbours cited it, in either memory the integrated method reads:

* the development memory: a positive vote of ``src_knn_qwen8b`` (Qwen3-8B key, k 10, the training folds of the case's
  outer fold), read from its saved top 200 (a case has at most a few dozen positive votes, all inside its top 200);
* the historical memory M1 (``uol.round3.m1_memory``, the per-fold selected configuration that S_full_ext reads):
  a neighbour with positive weight cited the article (its ``count`` column > 0). Recomputed from the cached
  neighbours on the thesis PC; when that is not possible the development memory alone is reported as a lower bound.

The share is the mean over DEV cases of supported articles among the first ten (cases with fewer than ten listed
articles use the ones they have); F4 passes at .5. Interval: the group bootstrap of §1 on the case mean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

SYSTEM = "S_full_ext"
DEV_MEMORY = "src_knn_qwen8b"
PASS = 0.5


def m1_counts(query_ids: np.ndarray, folds: np.ndarray, top: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """[n, 10] bool: the historical memory's neighbours of the case (its fold's selected configuration) cite the
    article. ``query_ids``: each case's id; ``top``: [n, 10] candidate index (-1 empty)."""
    from uol.round3 import common as R3, m1_memory as M
    sel = json.loads((R3.ARMS / "M1" / "RESULT.json").read_text(encoding="utf-8"))["selected_config"]
    rel = R3.release()
    pos = {rel.queries[int(r)].query_id: i for i, r in enumerate(R3.dev_rows())}       # M1's rows follow dev_rows()
    out = np.zeros(top.shape, dtype=bool)
    for k in sorted(set(int(f) for f in folds)):
        _, Cn = M.config_votes(sel[str(k)])
        for i in np.flatnonzero(folds == k):
            c = Cn.getrow(pos[str(query_ids[i])]).toarray().ravel()
            t = top[i]
            ok = t >= 0
            out[i, ok] = c[t[ok]] > 0
    return out, {"configs": sel, "n_candidates": int(Cn.shape[1])}


def share(supported: np.ndarray, listed: np.ndarray) -> np.ndarray:
    n = listed.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (supported & listed).sum(axis=1) / np.maximum(n, 1), np.nan)


def run(runs: Path, out: Path, data: Path | None = None, population: str = "DEV", draws: int = C.DRAWS,
        seed: int = C.SEED, with_m1: bool = True) -> dict[str, Any]:
    store = C.RunStore(runs, population)
    lab = store.labels
    top, _ = store.top(SYSTEM, 10)
    listed = top >= 0
    midx, msc = store.top(DEV_MEMORY, C.TOP_K)
    dev_sup = np.zeros(top.shape, dtype=bool)
    full_votes = 0
    for i in range(lab.n):
        votes = set(midx[i][(midx[i] >= 0) & (msc[i] > 0)].tolist())
        full_votes += int(np.sum(msc[i] > 0) == C.TOP_K)               # a case whose votes may run past rank 200
        dev_sup[i] = [int(a) in votes for a in top[i]]
    m1, m1_info, notes = None, None, []
    if with_m1:
        try:
            m1, m1_info = m1_counts(lab.query_id, lab.fold, top)
        except Exception as exc:                                          # not on this machine: lower bound only
            notes.append(f"M1 not recomputed ({type(exc).__name__}: {str(exc)[:160]}); development memory alone = lower bound")
    boot = C.Boot(lab.component, draws, seed)
    res = {}
    for name, sup in (("development memory", dev_sup), ("historical memory M1", m1),
                      ("either memory", None if m1 is None else (dev_sup | m1))):
        if sup is None:
            continue
        s = share(sup, listed)
        ok = np.isfinite(s)
        r = boot.ratio(np.where(ok, s, 0.0), ok.astype(np.float64))
        res[name] = {"share": r["mean"], "lo": r["lo"], "hi": r["hi"], "cases": int(ok.sum()),
                     "component_macro": boot.macro(s)["mean"], "cases_at_least_half": float(np.mean(s[ok] >= 0.5)),
                     "passes_F4": bool(r["mean"] >= PASS)}
    headline = "either memory" if "either memory" in res else "development memory"
    rec = {"item": "f4", "scope": C.SCOPE, "population": population, "system": SYSTEM, "dev_memory_source": DEV_MEMORY,
           "pass_line": PASS, "headline": headline, "lower_bound": headline == "development memory",
           "results": res, "m1": m1_info, "cases_with_200_positive_votes": full_votes, "notes": notes}
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "f4.json", rec)
    C.write_csv(out / "f4.csv", [{"memory": k, **v} for k, v in res.items()])
    per_case = pd.DataFrame({"query_id": lab.query_id, "dev_memory": share(dev_sup, listed)})
    if m1 is not None:
        per_case["m1"] = share(m1, listed)
        per_case["either"] = share(dev_sup | m1, listed)
    per_case.to_csv(out / "f4_per_case.csv", index=False)
    return rec
