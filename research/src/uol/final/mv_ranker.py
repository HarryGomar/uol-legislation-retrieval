"""E7b, next step (overnight_20261105/PLAN.md): the stored multi-vector scores as columns of the chosen ranker. No call, no card.

    py -3.12 -m uol.final.mv_ranker --go
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict, deque

import numpy as np

from uol.final import rank_rerun as R
from uol.final import audit_d181_common as A, footing as FT
from uol.final import multivector as MV

PROG = A.ROOT / "research" / "programs" / "overnight_20261105"
RUNS = {"rk_mv": "rt_lm_tesis", "rk_noenc_mv": "rk_noenc"}


def own_systems() -> None:
    src = json.loads((A.RUN / "systems_audit.json").read_text(encoding="utf-8"))
    for sid, base in RUNS.items():
        s = copy.deepcopy(next(x for x in src["systems"] if x["id"] == base))
        s["id"], s["notes"] = sid, f"{base} plus the trained multi-vector model's score and rank"
        ext = s["adapter_args"].get("extension")
        s["adapter_args"].update({"extension": f"{ext}+columns" if ext and "columns" not in ext else "columns" if not ext else ext, "arm": "mv"})
        s.pop("answer", None)
        src["systems"] = [x for x in src["systems"] if x["id"] != sid] + [s]
    path = PROG / "systems_mv.json"
    path.write_text(json.dumps(src, indent=1), encoding="utf-8")
    os.environ["UOL_FINAL_SYSTEMS"] = str(path)


def scores_by_row(d) -> tuple[dict[int, tuple[int, int]], dict[int, np.ndarray]]:
    norm = lambda t: re.sub(r"\s+", " ", t).strip()
    free = defaultdict(deque)
    for i, q in enumerate(d.queries):
        free[norm(q.text)].append(i)
    where, S = {}, {}
    for k in range(MV.FOLDS):
        q = [json.loads(x) for x in (MV.exchange() / f"f{k}" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
        S[k] = np.load(MV.exchange() / "outputs" / f"f{k}" / "scores.npy", mmap_mode="r")
        for i, x in enumerate(q):
            where[free[norm(x["text"])].popleft()] = (k, i)
    if len(where) != len(d.queries):
        raise RuntimeError(f"{len(where)} of {len(d.queries)} joined")
    return where, S


def install(where, S) -> None:
    def arm_columns(arm: str, k: int, h) -> tuple:
        from uol.embedding_push import common as ep
        if arm != "mv":
            raise ValueError(arm)
        cols = h.empty_columns(2)
        for r in ep.dev_rows():
            r = int(r)
            sl, pc = h.pair_slice(r)
            fold, i = where[r]
            row = np.asarray(S[fold][i], dtype=np.float32)
            rank = np.empty(len(row), dtype=np.float32)
            rank[np.argsort(-row, kind="stable")] = np.arange(1, len(row) + 1, dtype=np.float32)
            cols[sl] = np.column_stack([row[pc], rank[pc]])
        return cols, {"rows": len(where)}
    FT.arm_columns = arm_columns


def run() -> int:
    from uol.embedding_push import common as ep
    from uol.final import harness, lean_rerun as LR
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0")
    own_systems()
    R._agents()
    R.D.activate(R.STORE)
    LR._overlay_stat_glob()
    d = ep.release()
    where, S = scores_by_row(d)
    install(where, S)
    base = A.per_case(R.STORE, R.BASE)
    res = {"base_ndcg": float(json.loads((A.RUN / "dev" / "store" / "runs" / R.BASE / "DEV" / "summary.json").read_text(encoding="utf-8"))["ndcg_at_10"]), "runs": {}}
    for sid in RUNS:
        s = harness.run(sid, "DEV", store=R.STORE)
        if s.get("status") != "done":
            raise RuntimeError(f"{sid}: {s.get('status')} {s.get('reason')}")
        fam = A.dev_family(f"mv_{sid}", [(sid, A.per_case(R.STORE, sid), R.BASE, base)], population="DEV", size=len(RUNS))
        c = next(iter(fam["contrasts"].values()))
        res["runs"][sid] = {"ndcg_at_10": float(s["ndcg_at_10"]), **{x: c[x] for x in ("delta", "ci95", "adjusted_lower", "class")}}
        print(sid, json.dumps(res["runs"][sid], default=float), flush=True)
    (PROG / "mv_ranker.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    return run() if a.go else 0


if __name__ == "__main__":
    sys.exit(main())
