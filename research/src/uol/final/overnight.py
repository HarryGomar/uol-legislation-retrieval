"""Overnight experiments of 5 Nov 2026 (research/programs/overnight_20261105/PLAN.md). Development cases only.

    py -3.12 -m uol.final.overnight lists --go       the list of 100 on the 917 cases outside the sample
    py -3.12 -m uol.final.overnight rerank --which list100_rerank --go
    py -3.12 -m uol.final.overnight rerank --which list100_merged_rerank --go
    py -3.12 -m uol.final.overnight two_models       no call: GPT-6.1 Sol's and Opus's lists on the sample
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from uol.final import rank_rerun as R
from uol.final import audit_d181_common as A, footing as FT, lean_rerun as LR
from uol.final.learned_merge import recalls, CUTS

PROG = A.ROOT / "research" / "programs" / "overnight_20261105"
OUT = A.WORK / "overnight"
LISTS = OUT / "list100_full" / "answers.json"
MEASURES = ["ndcg_at_10", *[f"recall_at_{k}" for k in CUTS]]
CAP_USD = 30.0                      # each reranking; the expected cost is USD 18


def setup():
    from uol.embedding_push import common as ep
    from uol.final import guard
    G = R._agents()
    R.D.activate(R.STORE)
    d = ep.release()
    rows = G.all_rows()
    guard.dev_rows_only(rows, "development cases")
    stored = A.per_case(R.STORE, "gpt61sol_alone")
    if len(rows) != 1217 or any(d.queries[int(r)].query_id != q for r, q in zip(stored["row"], stored["query_id"])):
        raise RuntimeError("stored ids and the active release disagree")
    return G, d, {int(i) for i in d.generic_idx}, rows


def lists() -> int:
    from uol.final import llm_agents as L, prompt_check as P
    G, d, generic, rows = setup()
    luna = L.pin_model()
    have = {int(k): v for k, v in json.loads((A.WORK / "agents" / "check_list100" / "answers.json").read_text(encoding="utf-8")).items()}
    rest = [r for r in rows if r not in have]
    if len(have) != 300 or len(rest) != 917:
        raise RuntimeError(f"{len(have)} stored, {len(rest)} to ask")
    schema = P.item_schema(100)
    batches = [rest[i:i + 10] for i in range(0, len(rest), 10)]
    built = [P.build(b, 100) for b in batches]
    t0 = time.perf_counter()
    res = luna.run_many([(p, schema) for p, _ in built], effort="medium", tag="overnight_list100", workers=3)
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    answers = dict(have)
    for (prompt, ids), part, r in zip(built, batches, res):
        got = luna.checked(r, prompt, schema, ids, effort="medium", tag="overnight_list100")
        for i, row in zip(ids, part):
            answers[row] = got[i]["articulos"]
        usage["calls"] += 1
        for k in ("input_tokens", "output_tokens"):
            usage[k] += int((r.get("usage") or {}).get(k) or 0)
    LISTS.parent.mkdir(parents=True, exist_ok=True)
    LISTS.write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    usd = (usage["input_tokens"] * 2 + usage["output_tokens"] * 10) / 1e6
    run = {"cases_asked": len(rest), "cases": len(answers), "usage": usage, "usd_list_no_cache": usd, "seconds": round(time.perf_counter() - t0, 1)}
    (LISTS.parent / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run), flush=True)
    return 0


def rerank(which: str) -> int:
    from uol.final import external as X
    G, d, generic, rows = setup()
    if which == "learned_rerank":
        got = json.loads((A.WORK / "learned_merge" / "ranked.json").read_text(encoding="utf-8"))
        inp = {r: [int(c) for c in got[str(r)]] for r in rows}
        mdl, family = inp, len(MEASURES)
    else:
        answers = json.loads(LISTS.read_text(encoding="utf-8"))
        mdl, family = {r: G.resolved(answers[str(r)], generic) for r in rows}, 2 * len(MEASURES)
        if which == "list100_rerank":
            inp = mdl
        elif which == "list100_merged_rerank":
            rk = G.ranker_lists(rows, R.BASE)
            inp = {r: G.rrf([mdl[r], [c for c in rk[r] if c not in generic]]) for r in rows}
        else:
            raise ValueError(which)
    w = float(json.loads((A.WORK / "dev" / "llm" / f"{G.KIND}_rerank_S0+qwen3_8b" / "meta.json").read_text(encoding="utf-8"))["w_all"])
    shown = {r: inp[r][:X.TOP] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[G.KIND]}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    inner = G.ask_some(lambda items: X.ask_claude(items, X.codex_api(cfg, f"overnight_{which}", usage), stats, cfg))

    def ask(items):
        got = inner(items)
        spent = X.codex_usage(usage, cfg)[1]
        print(json.dumps({"items": len(items), "usd_so_far": round(spent, 3)}), flush=True)
        if spent > CAP_USD:
            raise RuntimeError(f"USD {spent:.2f} passed the cap of {CAP_USD}")
        return got

    t0 = time.perf_counter()
    o = FT.rerank_depth(rows, shown, ask, X.TOP, passes=("fwd",))
    tot, usd = X.codex_usage(usage, cfg)
    seconds = round(time.perf_counter() - t0, 1)
    out = OUT / which
    out.mkdir(parents=True, exist_ok=True)
    (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    ranked = {r: G.fuse_order(shown[r], o["orders"], ["fwd"], r, w) + inp[r][X.TOP:] for r in rows}
    (out / "ranked.json").write_text(json.dumps({str(r): v for r, v in ranked.items()}), encoding="utf-8")
    sys_lists = G.ranker_lists(rows, R.SYSTEM)
    frames = {n: G.score_ranked(rows, src).merge(recalls(rows, src, d, generic), on="query_id") for n, src in
              (("run", ranked), ("input", inp), ("system", sys_lists))}
    frames["run"].to_parquet(out / "per_case.parquet")
    stored = A.per_case(R.STORE, R.SYSTEM)
    contrasts = {}
    for m in MEASURES:
        fam = A.dev_family(f"{which}_{m}", [(which, frames["run"], "system", frames["system"])], population="DEV", measure=m, size=family)
        contrasts[m] = next(iter(fam["contrasts"].values()))
    res = {"which": which, "cases": len(rows), "kept_median": float(np.median([len(v) for v in mdl.values()])), "read_per_case": X.TOP, "w_all": w,
           "means_by_group": {n: {m: float(f.groupby("component_id")[m].mean().mean()) for m in MEASURES} for n, f in frames.items()},
           "cited_in_first_30": {n: float(f["cited_in_first_30"].sum() / f["cited"].sum()) for n, f in frames.items()},
           "contrasts_run_minus_system": contrasts,
           "checks": {"system_ndcg_stored": float(stored["ndcg_at_10"].mean()), "system_ndcg_here": float(frames["system"]["ndcg_at_10"].mean())},
           "usage": tot, "usd_list_equivalent": usd, "seconds": seconds, "stats": stats}
    PROG.mkdir(parents=True, exist_ok=True)
    (PROG / f"{which}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("means_by_group", "checks", "usd_list_equivalent", "seconds")}, indent=1, default=float), flush=True)
    return 0


def two_models() -> int:
    from uol.embedding_push import common as ep
    G, d, generic, rows = setup()
    keep = sorted(LR.sample_rows())
    sol = json.loads((A.WORK / "agents" / "alone" / "answers.json").read_text(encoding="utf-8"))
    opus = json.loads((A.WORK / "why_final_sample" / "alone_opus_medium" / "answers.json").read_text(encoding="utf-8"))
    a = {r: G.resolved(sol[str(r)]["articulos"], generic) for r in keep}
    b = {r: G.resolved(opus[str(r)], generic) for r in keep}
    both = {r: G.rrf([a[r], b[r]]) for r in keep}
    rec = lambda src: G.score_ranked(keep, src).merge(pd.DataFrame([{"query_id": d.queries[r].query_id,
              "recall_at_20": (len({int(x) for x in d.specific_idx[r]} & set(src[r][:20])) / len(d.specific_idx[r])) if len(d.specific_idx[r]) else 0.0} for r in keep]), on="query_id")
    frames = {"sol": rec(a), "opus": rec(b), "both": rec(both)}
    measures = ["ndcg_at_10", "recall_at_20"]
    contrasts = {}
    for m in measures:
        fam = A.dev_family(f"two_models_{m}", [("both", frames["both"], "sol", frames["sol"])], population="DEV-A300", measure=m, size=len(measures))
        contrasts[m] = next(iter(fam["contrasts"].values()))
    res = {"cases": len(keep), "means_by_group": {n: {m: float(f.groupby("component_id")[m].mean().mean()) for m in measures} for n, f in frames.items()},
           "contrasts_both_minus_sol": contrasts, "named_median": {"sol": float(np.median([len(v) for v in a.values()])), "both": float(np.median([len(v) for v in both.values()]))}}
    PROG.mkdir(parents=True, exist_ok=True)
    (PROG / "two_models.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["means_by_group"], indent=1), flush=True)
    for m, c in contrasts.items():
        print(m, c["delta"], c["ci95"], c["adjusted_lower"], c["class"], flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lists", "rerank", "two_models"])
    ap.add_argument("--which")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "two_models":
        return two_models()
    if not a.go:
        print(f"[dry] would run {a.step}")
        return 0
    return lists() if a.step == "lists" else rerank(a.which)


if __name__ == "__main__":
    sys.exit(main())
