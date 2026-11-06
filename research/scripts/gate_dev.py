"""The ranker as a gate on the second reading, on the development cases only (v27, 3 Nov 2026; future work in 7.4).

The system's second reading is skipped where the ranker and the model's list put the same article first; there the
answer is the model's list with the ranker's list after it. Read from the stored lists of the final runs, with no model
call and no test case. Also the ranker and the system on the tenth of cases where the ranker is most confident (its
highest top score), the router that 7.4 no longer proposes. Writes
``research/programs/final_results_20261102/gate.json``, which ``scripts/final_results.py`` reads.

    py -3.12 scripts/gate_dev.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uol.round3 import stats  # noqa: E402

RUNS = ROOT / "experiments/runs/footing_20261101/dev/store/runs"
OUT = ROOT / "research/programs/final_results_20261102"
RANKER, ALONE = "rt_lm_tesis", "gpt61sol_alone"


def lists(sid: str) -> dict[str, pd.DataFrame]:
    d = pd.read_parquet(RUNS / sid / "DEV" / "top200.parquet")
    return {q: g.sort_values("rank") for q, g in d.groupby("query_id")}


def ndcg(lst: list[int], y: set[int]) -> float:
    dcg = sum(1 / math.log2(1 + r) for r, a in enumerate(lst[:10], 1) if a in y)
    z = sum(1 / math.log2(1 + r) for r in range(1, min(len(y), 10) + 1))
    return dcg / z if z else 0.0


def main() -> int:
    sys_id = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["system"]["dev_system"]
    pc = pd.read_parquet(RUNS / RANKER / "DEV" / "per_case.parquet")
    Q = list(pc.query_id)
    rel = {r.query_id: {int(x) for x in str(r.positive_idx).split(",") if x} for r in pc.itertuples()}
    comp = dict(zip(pc.query_id, pc.component_id))
    rk, al, sy = lists(RANKER), lists(ALONE), lists(sys_id)
    L = {q: list(rk[q].candidate_idx) for q in Q}
    A = {q: list(al[q][al[q].score > -1e8].candidate_idx) for q in Q}      # the model's own list, without the padding
    S = {q: list(sy[q].candidate_idx) for q in Q}
    top = {q: float(rk[q].score.iloc[0]) for q in Q}
    agree = {q: bool(A[q]) and A[q][0] == L[q][0] for q in Q}
    appended = {q: A[q] + [x for x in L[q] if x not in set(A[q])] for q in Q}
    n_sys = {q: ndcg(S[q], rel[q]) for q in Q}
    n_gate = {q: ndcg(appended[q], rel[q]) if agree[q] else n_sys[q] for q in Q}
    n_rank = {q: ndcg(L[q], rel[q]) for q in Q}

    def per_comp(v: dict, qs: list[str]) -> tuple[list[str], np.ndarray]:
        by = defaultdict(list)
        for q in qs:
            by[comp[q]].append(v[q])
        keys = sorted(by)
        return keys, np.array([np.mean(by[k]) for k in keys])

    keys, g = per_comp(n_gate, Q)
    _, s = per_comp(n_sys, Q)
    con = stats.contrast(g, s, family_size=1)
    thr = float(np.quantile(list(top.values()), 0.9))
    conf = [q for q in Q if top[q] >= thr]
    out = {
        "system_id": sys_id, "rule": "skip the second reading where the model's first article is the ranker's first",
        "cases": len(Q), "share_skipped": float(np.mean([agree[q] for q in Q])),
        "gate_ndcg": float(g.mean()), "system_ndcg": float(s.mean()),
        "gate_minus_system": {k: con[k] for k in ("delta", "ci95", "ci90", "ci_adjusted", "adjusted_lower", "class", "family_size", "n")},
        "confident_tenth": {"cases": len(conf), "threshold_top_score": thr,
                            "ranker": float(per_comp(n_rank, conf)[1].mean()), "system": float(per_comp(n_sys, conf)[1].mean())},
        "note": "development cases only; the test set is not read",
    }
    (OUT / "gate.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
