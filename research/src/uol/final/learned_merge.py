"""A learned merge, first read: the chosen ranker's inputs plus the model's stored list (research/programs/learned_merge_20261105/PLAN.md).

No model call, development cases only, stops after this read.

    py -3.12 -m uol.final.learned_merge --go
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from uol.final import rank_rerun as R
from uol.final import audit_d181_common as A, footing as FT, lean_rerun as LR

SID = "lm_merge"
PROG = A.ROOT / "research" / "programs" / "learned_merge_20261105"
CUTS = (10, 30, 100, 200)
NOT_NAMED = 21.0
POOL: dict[int, set[int]] = {}      # each case's scored pool, filled when the columns are built


def own_systems() -> None:
    """A copy of the registry with this one run added: the shared registry stays as it is."""
    src = json.loads((A.RUN / "systems_audit.json").read_text(encoding="utf-8"))
    base = next(s for s in src["systems"] if s["id"] == "rt_lm_tesis")
    s = copy.deepcopy(base)
    s["id"], s["notes"] = SID, "a learned merge: the chosen ranker's inputs plus the model's stored list"
    s["adapter_args"].update({"extension": "m1+columns", "arm": "model_list"})
    s.pop("answer", None)
    src["systems"] = [x for x in src["systems"] if x["id"] != SID] + [s]
    PROG.mkdir(parents=True, exist_ok=True)
    path = PROG / "systems.json"
    path.write_text(json.dumps(src, indent=1), encoding="utf-8")
    os.environ["UOL_FINAL_SYSTEMS"] = str(path)


def model_lists(G, rows, generic) -> dict[int, list[int]]:
    answers = json.loads((A.WORK / "agents" / "alone_full" / "answers.json").read_text(encoding="utf-8"))
    return {r: G.resolved(answers[str(r)]["articulos"], generic) for r in rows}


def install_columns(lists: dict[int, list[int]]) -> None:
    def arm_columns(arm: str, k: int, h) -> tuple:
        from uol.embedding_push import common as ep
        if arm != "model_list":
            raise ValueError(arm)
        cols = h.empty_columns(2)
        with_list = named_in_pool = 0
        for r in ep.dev_rows():
            r = int(r)
            sl, pc = h.pair_slice(r)
            out = np.full((len(pc), 2), np.nan, dtype=np.float32)
            if r in lists:
                POOL[r] = {int(c) for c in h.F.pc[h.v.idx[h.v.ptr[r]:h.v.ptr[r + 1]]]}
                rank = {int(c): i for i, c in enumerate(lists[r], 1)}
                out[:, 0] = [1.0 if int(c) in rank else 0.0 for c in pc]
                out[:, 1] = [float(rank.get(int(c), NOT_NAMED)) for c in pc]
                with_list += 1
                named_in_pool += int(out[:, 0].sum())
            cols[sl] = out
        return cols, {"cases_with_a_list": with_list, "named_in_pool": named_in_pool}
    FT.arm_columns = arm_columns


def recalls(rows, ranked, d, generic) -> pd.DataFrame:
    out = []
    for r in rows:
        gold = {int(x) for x in d.specific_idx[r]}
        kept = [int(c) for c in ranked[r] if int(c) not in generic]
        out.append({"query_id": d.queries[r].query_id, "cited_in_first_30": len(gold & set(kept[:30])), "cited": len(gold),
                    **{f"recall_at_{k}": len(gold & set(kept[:k])) / len(gold) if gold else 0.0 for k in CUTS}})
    return pd.DataFrame(out)


def run() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0: the doc2vec columns depend on it")
    from uol.embedding_push import common as ep
    from uol.final import guard, harness
    own_systems()
    G = R._agents()
    R.D.activate(R.STORE)
    LR._overlay_stat_glob()
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = G.all_rows()
    guard.dev_rows_only(rows, "development cases")
    alone_frame = A.per_case(R.STORE, "gpt61sol_alone")
    if any(d.queries[int(r)].query_id != q for r, q in zip(alone_frame["row"], alone_frame["query_id"])) or len(rows) != 1217:
        raise RuntimeError("stored ids and the active release disagree")
    lists = model_lists(G, rows, generic)
    install_columns(lists)
    t0 = time.perf_counter()
    s = harness.run(SID, "DEV", store=R.STORE)
    if s.get("status") != "done":
        raise RuntimeError(f"{SID}: {s.get('status')} {s.get('reason')}")
    seconds = round(time.perf_counter() - t0)
    learned_pool = G.ranker_lists(rows, SID)
    rk = G.ranker_lists(rows, R.BASE)
    merged = {r: G.rrf([lists[r], [c for c in rk[r] if c not in generic]]) for r in rows}

    # the articles the model names outside the pool, at twice their rank in its list
    pool = POOL
    learned, outside = {}, []
    for r in rows:
        extra = [(i, c) for i, c in enumerate(lists[r], 1) if int(c) not in pool[r]]
        outside.append(len(extra))
        order = [c for c in learned_pool[r] if c not in generic and c not in {x for _, x in extra}]
        for i, c in extra:
            order.insert(min(2 * i - 1, len(order)), c)
        learned[r] = order
    frames = {}
    for name, src in (("learned", learned), ("merged", merged), ("alone", lists), ("ranker", rk)):
        frames[name] = G.score_ranked(rows, src).merge(recalls(rows, src, d, generic), on="query_id")
    PROG_CACHE = A.WORK / "learned_merge"
    PROG_CACHE.mkdir(parents=True, exist_ok=True)
    frames["learned"].to_parquet(PROG_CACHE / "per_case.parquet")
    (PROG_CACHE / "ranked.json").write_text(json.dumps({str(r): v for r, v in learned.items()}), encoding="utf-8")
    stored_merged = A.per_case(R.STORE, f"{R.SYSTEM}_merged")
    stored_alone = alone_frame
    measures = ["ndcg_at_10", *[f"recall_at_{k}" for k in CUTS]]
    pairs = {}
    for other in ("merged", "alone"):
        for m in measures:
            fam = A.dev_family(f"{SID}_{other}_{m}", [("learned", frames["learned"], other, frames[other])], population="DEV", measure=m, size=2 * len(measures))
            pairs[f"{other}:{m}"] = next(iter(fam["contrasts"].values()))
    by_group = {n: {m: float(f.groupby("component_id")[m].mean().mean()) for m in measures} for n, f in frames.items()}
    res = {"cases": len(rows), "means_by_group": by_group,
           "contrasts_learned_minus": pairs,
           "cited_in_first_30": {n: {"count": int(f["cited_in_first_30"].sum()), "of": int(f["cited"].sum()), "share": float(f["cited_in_first_30"].sum() / f["cited"].sum())} for n, f in frames.items()},
           "pool_median": float(np.median([len(POOL[r]) for r in rows])),
           "named_outside_pool": {"articles": int(sum(outside)), "cases": int(sum(1 for x in outside if x)), "named": int(sum(len(v) for v in lists.values()))},
           "checks": {"merged_ndcg_stored": float(stored_merged["ndcg_at_10"].mean()), "merged_ndcg_here": float(frames["merged"]["ndcg_at_10"].mean()),
                      "alone_ndcg_stored": float(stored_alone["ndcg_at_10"].mean()), "alone_ndcg_here": float(frames["alone"]["ndcg_at_10"].mean()),
                      "ranker_ndcg_stored": float(A.per_case(R.STORE, R.BASE)["ndcg_at_10"].mean()), "ranker_ndcg_here": float(frames["ranker"]["ndcg_at_10"].mean())},
           "learned_pool_only_ndcg": float(s.get("ndcg_at_10")), "seconds": seconds}
    gate = {"recall30_lower_above_zero": pairs["merged:recall_at_30"]["adjusted_lower"] > 0,
            "ndcg_upper_not_below_zero": pairs["merged:ndcg_at_10"]["ci_adjusted"][1] >= 0,
            "recall100_upper_not_below_zero": pairs["merged:recall_at_100"]["ci_adjusted"][1] >= 0,
            "recall200_upper_not_below_zero": pairs["merged:recall_at_200"]["ci_adjusted"][1] >= 0}
    res["gate"] = {**gate, "met": all(gate.values())}
    (PROG / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("means_by_group", "cited_in_first_30", "pool_median", "named_outside_pool", "checks", "gate", "seconds")}, indent=1, default=float), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print("[dry] would fit the learned merge on the development cases")
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main())
