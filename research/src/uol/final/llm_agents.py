"""The chosen model without the reranker's fixed list (common_footing PLAN addendum 13, DEV-A300, dev only).

Three runs with one model, through the Codex CLI route with no tools of its own:
* ``alone``: the facts only; twenty (law, article) pairs, the first ten countable ones scored (``eval_lenses.llm_alone``);
* ``agent``: the lockstep agent of ``eval_lenses.e6_agent``, its search tool ``words`` or ``embedding``;
* ``list_agent``: the same agent, whose history opens with the ranker's first 30 articles, excerpted as the reranker
  sees them, so it may read, search further and answer; an answer of fewer than ten is filled from the ranker's order.

Every case's nDCG@10 is computed here and checked against the stored score of the system on the same cases.

Addendum 14: ``later`` asks the model alone on the later cases (after ``footing new-cases --kind gpt61sol``); ``combo``
gives the same reranker the first 30 of the model's and the ranker's lists merged, and fuses as the system does.

    py -3.12 -m uol.final.llm_agents alone --smoke 10 --go
    py -3.12 -m uol.final.llm_agents list_agent --tool words --smoke 5 --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A, audit_d181_dev as D

MODEL, KIND = "gpt-6.1-sol", "gpt61sol"
SYSTEM = f"{KIND}_rerank_{FT.RANKER}"
OUT = A.WORK / "agents"
LIST_PROMPT = Path(__file__).resolve().parents[1] / "eval_lenses" / "prompts" / "E6_list_agent.txt"
LIST_PER_CALL = 5
LIST_DEPTH = 30
RRF_K = 60
ALONE_SYSTEM, COMBO_SYSTEM = f"{KIND}_alone", f"{KIND}_combo"
OTHER_MODELS = ("gpt61terra", "gpt6solnw", "gpt56sol", "gpt6luna", "gpt56terra", "luna")
REPEATS = (11, 12)                  # the samples' own cache keys, apart from the resamples 1 to 3 of ``luna.checked``
EXAMPLES, EXAMPLE_CHARS = 5, 500
EXAMPLES_PROMPT = Path(__file__).resolve().parents[1] / "round3" / "prompts" / "closedbook_examples.txt"


def pin_model(model: str = MODEL):
    """Every call of the inherited runners goes to the chosen model; the answers of models never mix (key has the model)."""
    from uol.round3 import luna
    luna.MODEL = model
    from uol.final import codex
    codex.install()
    for name in ("checked", "checked_or_split"):
        f = getattr(luna, name)
        setattr(luna, name, lambda *a, _f=f, **k: _f(*a, **{"model": model, **k}))
    return luna


def sample_rows(smoke: int | None) -> list[int]:
    """DEV-A300 as the stored sample runs have it (the model alone's rows). Drawing it again from the current development
    rows gives 20 other cases (3 Nov 2026), so once that run exists its rows are read, never redrawn."""
    stored = OUT / "alone" / "per_case.parquet"
    if stored.exists():
        rows = sorted(int(r) for r in pd.read_parquet(stored, columns=["row"])["row"])
    else:
        from uol.embedding_push import common as ep
        from uol.final import populations
        rows = [int(r) for r in populations.agent_sample(ep.dev_rows())]
    return rows[:smoke] if smoke else rows


def score_ranked(rows: list[int], ranked: dict[int, list[int]]) -> pd.DataFrame:
    """nDCG@10 per case on the labels of the active profile: generic articles out, a repeated article once."""
    from uol.embedding_push import common as ep
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    out = []
    for r in rows:
        gold = {int(x) for x in d.specific_idx[r]}
        keep = list(dict.fromkeys(int(i) for i in ranked.get(r, []) if int(i) not in generic))[:10]
        gains = sum(1 / np.log2(k + 2) for k, i in enumerate(keep) if i in gold)
        idcg = sum(1 / np.log2(k + 2) for k in range(min(len(gold), 10)))
        out.append({"row": r, "query_id": d.queries[r].query_id, "component_id": d.queries[r].component_id,
                    "ndcg_at_10": gains / idcg if idcg else 0.0, "success_at_10": float(bool(gold & set(keep))), "returned": len(keep)})
    return pd.DataFrame(out)


def system_frames(rows: list[int]) -> dict[str, pd.DataFrame]:
    keep = set(rows)
    return {n: (lambda f: f[f["row"].isin(keep)])(A.per_case(s, n)) for s, n in ((FT.STORE, SYSTEM), (FT.AUDIT, FT.RANKER))}


def check_scorer(rows: list[int]) -> float:
    """The scorer here against the stored per-case scores of the ranker's top 10: they must agree."""
    from uol.embedding_push import common as ep
    d = ep.release()
    want = {d.queries[r].query_id: r for r in rows}
    top = pd.read_parquet(FT.base_top200(FT.RANKER))
    ranked = {want[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in top[top["query_id"].isin(want)].groupby("query_id")}
    mine = score_ranked(rows, ranked).set_index("row")["ndcg_at_10"]
    stored = A.per_case(FT.AUDIT, FT.RANKER).set_index("row")["ndcg_at_10"].reindex(mine.index)
    return float((mine - stored).abs().max())


def finish(name: str, rows: list[int], ranked: dict[int, list[int]], extra: dict, others: dict[str, pd.DataFrame] | None = None) -> int:
    """Per-case scores and the contrasts against the system and the ranker (and ``others``) on the same cases."""
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    frame = score_ranked(rows, ranked)
    frame.to_parquet(out / "per_case.parquet")
    frames = {**system_frames(rows), **(others or {})}
    fam = A.dev_family(name, [(name, frame, n, f) for n, f in frames.items()], population="DEV-A300", size=max(3, len(frames)))
    A.write_json(out / "contrasts.json", fam)
    res = {"name": name, "model": MODEL, "cases": len(rows), "mean_per_case": float(frame["ndcg_at_10"].mean()),
           "success_at_10": float(frame["success_at_10"].mean()), "returned_mean": float(frame["returned"].mean()),
           "others_same_cases": {n: float(f["ndcg_at_10"].mean()) for n, f in frames.items()},
           "contrasts": {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in fam["contrasts"].items()}, **extra}
    A.receipt(f"agents_{name}", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return 0


def alone(smoke: int | None) -> int:
    from uol.eval_lenses import llm_alone as LA
    from uol.round3 import catalog_map
    pin_model()
    rows = sample_rows(smoke)
    print(f"scorer against the stored ranker, largest difference {check_scorer(rows):.6f}", flush=True)
    t0 = time.perf_counter()
    answers = LA.run(rows)
    ranked = {}
    for r in rows:
        ranked[r] = [i for i in (catalog_map.resolve(x["ley"], x["articulo"])[0] for x in answers[r]["articulos"]) if i is not None]
    (OUT / "alone").mkdir(parents=True, exist_ok=True)
    (OUT / "alone" / "answers.json").write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    return finish("alone" + (f"_smoke{smoke}" if smoke else ""), rows, ranked, {"seconds": round(time.perf_counter() - t0, 1)})


def agent(tool: str, with_list: bool, smoke: int | None) -> int:
    from uol.eval_lenses import e6_agent as E
    from uol.final import external as X, llm_jobs
    pin_model()
    name = f"{'list_agent' if with_list else 'agent'}_{tool}" + (f"_smoke{smoke}" if smoke else "")
    E.OUT = OUT / name
    E.OUT.mkdir(parents=True, exist_ok=True)
    llm_jobs.agent_baseline_cache()
    if tool == "embedding":
        from uol.final import query_server
        query_server.patch_endpoint()
    rows = sample_rows(smoke)
    start = None
    if with_list:
        E.PROMPT, E.PER_CALL = LIST_PROMPT, LIST_PER_CALL
        top = X.top30(FT.base_top200(FT.RANKER), rows)
        start = {r: [int(c) for c in top[r]][:LIST_DEPTH] for r in rows}
    t0 = time.perf_counter()
    res = E.run(rows, tool, f"agents_{name}", start=start)
    (E.OUT / "result.json").write_text(json.dumps(res, default=str, ensure_ascii=False), encoding="utf-8")
    ranked = {}
    for c in res["cases"]:
        r = int(c["row"])
        ids = [E.parse_aid(x) for x in c["returned"]]
        ranked[r] = [i for i in ids if i is not None] + (start[r] if start else [])
    steps = [c["steps"] for c in res["cases"]]
    searches = [c["searches"] for c in res["cases"]]
    return finish(name, rows, ranked, {"usage": res["usage"], "splits": res["splits"], "seconds": round(time.perf_counter() - t0, 1),
                                       "steps_mean": float(np.mean(steps)), "searches_mean": float(np.mean(searches))})


def resolved(answer: list[dict], generic: set[int]) -> list[int]:
    """The model's articles in its order, as catalogue rows: unresolved and generic ones out, a repeat once."""
    from uol.round3 import catalog_map
    ids = [catalog_map.resolve(x["ley"], x["articulo"])[0] for x in answer]
    return list(dict.fromkeys(int(i) for i in ids if i is not None and int(i) not in generic))


def rrf(lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion; a tie goes to the earlier list's order."""
    score: dict[int, float] = {}
    for lst in lists:
        for rank, c in enumerate(lst, 1):
            score[c] = score.get(c, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda c: -score[c])


def fuse_order(shown: list[int], order: dict, passes: list[str], row: int, w: float) -> list[int]:
    """z(position in the list given to the reranker) + w z(1 / mean position over the passes), as ``features.fuse_rows``."""
    from uol.round3 import features
    pos: dict[int, list[int]] = {}
    for p in passes:
        for k, c in enumerate(order[p][str(row)], 1):
            pos.setdefault(int(c), []).append(k)
    base = -np.arange(len(shown), dtype=np.float64)
    arm = np.asarray([1.0 / np.mean(pos[c]) if c in pos else 0.0 for c in shown], dtype=np.float64)
    comb = features._z(base) + w * features._z(arm)
    return [shown[i] for i in np.argsort(-comb, kind="stable")]


def ranker_lists(rows: list[int], base: str = FT.RANKER) -> dict[int, list[int]]:
    from uol.embedding_push import common as ep
    d = ep.release()
    want = {d.queries[r].query_id: r for r in rows}
    top = pd.read_parquet(FT.base_top200(base))
    return {want[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in top[top["query_id"].isin(want)].groupby("query_id")}


def combo(smoke: int | None) -> int:
    """PLAN addendum 14: the model's list and the ranker's merged, the first 30 reranked; nothing tuned."""
    from uol.embedding_push import common as ep
    from uol.final import external as X
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    answers = json.loads((OUT / "alone" / "answers.json").read_text(encoding="utf-8"))
    rows = sample_rows(smoke)
    rk = ranker_lists(rows)
    model = {r: resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    # control: this fusion on the system's own orders must give the stored system
    so = json.loads((A.WORK / "effort" / KIND / "full" / f"{KIND}_orders_s0.json").read_text(encoding="utf-8"))
    ctrl = {r: fuse_order(rk[r][:X.TOP], so["orders"], so["passes"], r, w) + rk[r][X.TOP:] for r in rows}
    mine = score_ranked(rows, ctrl)["ndcg_at_10"].mean()
    stored = system_frames(rows)[SYSTEM]["ndcg_at_10"].mean()
    print(f"control: this fusion on the system's orders {mine:.4f}, stored {stored:.4f}", flush=True)
    if abs(mine - stored) > .005:
        raise RuntimeError("the fusion does not reproduce the stored system")
    frames = system_frames(rows)
    keep = {"alone": pd.read_parquet(OUT / "alone" / "per_case.parquet")}
    keep = {"alone": keep["alone"][keep["alone"]["row"].isin(rows)]}
    sys_top = pd.read_parquet(FT.STORE / "runs" / SYSTEM / "DEV" / "top200.parquet")
    want = {d.queries[r].query_id: r for r in rows}
    sys_rank = {want[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in sys_top[sys_top["query_id"].isin(want)].groupby("query_id")}
    tag = "" if not smoke else f"_smoke{smoke}"
    finish(f"sum_model_ranker{tag}", rows, {r: rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}, {"calls": 0}, keep)
    finish(f"sum_model_system{tag}", rows, {r: rrf([model[r], [c for c in sys_rank[r] if c not in generic]]) for r in rows}, {"calls": 0}, keep)
    merged = {r: rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    reached = float(np.mean([bool({int(g) for g in d.specific_idx[r]} & set(shown[r])) for r in rows]))
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[KIND]}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    o = X.rerank_orders(rows, shown, lambda items: X.ask_claude(items, X.codex_api(cfg, f"agents_combo{tag}", usage), stats, cfg))
    tot, usd = X.codex_usage(usage, cfg)
    out = OUT / f"combo{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    ranked = {r: fuse_order(shown[r], o["orders"], o["passes"], r, w) + merged[r][X.TOP:] for r in rows}
    return finish(f"combo{tag}", rows, ranked, {"usage": tot, "usd_list_equivalent": usd, "stats": stats, "w_all": w, "reached_in_30": reached,
                                                "seconds": round(time.perf_counter() - t0, 1)}, keep)


def later() -> int:
    """PLAN addendum 14: the model alone on the later cases, against the system and the ranker on the same cases."""
    from uol.final import new_cases_llm as NL, new_cases_report as NR, new_cases_sources as NSrc
    run = A.RUN / "new"
    if not (run / "eval" / SYSTEM / "per_case.parquet").exists():
        raise RuntimeError("the system has not been scored on the later cases")
    NL.P6 = {**NL.P6, "base": FT.RANKER, "system": SYSTEM}
    rows_a, M, _nr = NL._base(run)
    rows = [int(r) for r in rows_a]
    from uol.embedding_push import common as ep
    from uol.eval_lenses import llm_alone as LA
    from uol.round3 import common as C
    luna = pin_model()
    luna.CACHE = run / "llm" / "luna" / "cache.sqlite"
    luna.LEDGER = run / "llm" / "luna" / "LUNA_LEDGER.jsonl"
    luna.CACHE.parent.mkdir(parents=True, exist_ok=True)
    closed = np.asarray([r for r in C.lockbox_rows() if int(r) < NSrc.N_V11], dtype=np.int64)
    C.lockbox_rows = ep.lockbox_rows = lambda: closed           # the guard keeps refusing the closed test set; the new rows pass
    LA.TAG = "llm_alone_later"
    t0 = time.perf_counter()
    answers = LA.run(rows)
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    ranked = {r: resolved(answers[r]["articulos"], generic) for r in rows}
    frame = score_ranked(rows, ranked)
    out = OUT / "later"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "per_case.parquet")
    (out / "answers.json").write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    frames = {"alone": frame, SYSTEM: pd.read_parquet(run / "eval" / SYSTEM / "per_case.parquet"), FT.RANKER: pd.read_parquet(run / "eval" / FT.RANKER / "per_case.parquet")}
    pos = {r: i for i, r in enumerate(rows)}
    check = score_ranked(rows, {r: [int(c) for c in np.argsort(-M[pos[r]], kind="stable")[:200]] for r in rows}).set_index("query_id")["ndcg_at_10"]
    stored = frames[FT.RANKER].set_index("query_id")["ndcg_at_10"].reindex(check.index)
    rule = json.loads((run / "HEADLINE_RULE.json").read_text(encoding="utf-8"))
    dates = NR.decision_dates(run / "NEW" / "release")
    headline = set(dates) - {c["query_id"] for c in rule["overlap_cases"]}
    cutoff = FT.CUTOFFS[KIND]
    after = {q for q, x in dates.items() if x["earliest"] and x["earliest"] > cutoff} & headline
    scored = set(frame["query_id"])
    res = {"cutoff": cutoff, "cases": len(rows), "headline_scored": len(headline & scored), "decided_after_cutoff_scored": len(after & scored),
           "scorer_vs_stored_ranker": float((check - stored).abs().max()), "seconds": round(time.perf_counter() - t0, 1), "means": {}, "contrasts": {}}
    for name, keepset in (("headline", headline), ("decided_after_cutoff", after)):
        res["means"][name] = {k: float(f[f["query_id"].isin(keepset)]["ndcg_at_10"].mean()) for k, f in frames.items()}
        for b in (SYSTEM, FT.RANKER):
            res["contrasts"][f"{name}: alone - {b}"] = NR.pair(frames, "alone", b, "ndcg_at_10", 2, keepset)
        res["contrasts"][f"{name}: {SYSTEM} - {FT.RANKER}"] = NR.pair(frames, SYSTEM, FT.RANKER, "ndcg_at_10", 2, keepset)
    A.write_json(out / "contrasts.json", res)
    A.receipt("agents_later", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return 0


def later_more() -> int:
    """PLAN addendum 14, the author "Yes, run them": the list agent with word search and the combination on the later cases,
    each as on the development sample, nothing tuned, against the model alone, the system and the ranker on the same cases."""
    from types import SimpleNamespace
    from uol.final import external as X, new_cases_llm as NL, new_cases_report as NR, new_cases_sources as NSrc
    run = A.RUN / "new"
    NL.P6 = {**NL.P6, "base": FT.RANKER, "system": SYSTEM}
    rows_a, M, _nr = NL._base(run)
    rows = [int(r) for r in rows_a]
    from uol.embedding_push import common as ep
    from uol.eval_lenses import e6_agent as E
    from uol.round3 import common as C
    luna = pin_model()
    luna.CACHE = run / "llm" / "luna" / "cache.sqlite"
    luna.LEDGER = run / "llm" / "luna" / "LUNA_LEDGER.jsonl"
    closed = np.asarray([r for r in C.lockbox_rows() if int(r) < NSrc.N_V11], dtype=np.int64)
    C.lockbox_rows = ep.lockbox_rows = lambda: closed           # the guard keeps refusing the closed test set; the new rows pass
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    answers = json.loads((OUT / "later" / "answers.json").read_text(encoding="utf-8"))
    pos = {r: i for i, r in enumerate(rows)}
    rk = {r: [int(c) for c in np.argsort(-M[pos[r]], kind="stable")[:200]] for r in rows}
    model = {r: resolved(answers[str(r)]["articulos"], generic) for r in rows}

    t0 = time.perf_counter()
    E.OUT = OUT / "later_list_agent_words"
    E.OUT.mkdir(parents=True, exist_ok=True)
    E.PROMPT, E.PER_CALL = LIST_PROMPT, LIST_PER_CALL
    mask = np.zeros(len(d.candidate_ids), dtype=bool)
    mask[list(generic)] = True
    E.core.meta = lambda: SimpleNamespace(rows=np.asarray([], dtype=np.int64), generic=mask)   # no frozen baseline row exists for a later case
    start = {r: rk[r][:LIST_DEPTH] for r in rows}
    res = E.run(rows, "words", "agents_later_list_words", start=start)
    (E.OUT / "result.json").write_text(json.dumps(res, default=str, ensure_ascii=False), encoding="utf-8")
    agent_ranked = {}
    for c in res["cases"]:
        r = int(c["row"])
        ids = [E.parse_aid(x) for x in c["returned"]]
        agent_ranked[r] = [i for i in ids if i is not None] + start[r]
    agent_s = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    merged = {r: rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    w = float(json.loads((run / "scores" / SYSTEM / "manifest.json").read_text(encoding="utf-8"))["w_all"])
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[KIND]}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    o = X.rerank_orders(rows, shown, lambda items: X.ask_claude(items, X.codex_api(cfg, "agents_later_combo", usage), stats, cfg))
    tot, usd = X.codex_usage(usage, cfg)
    (OUT / "later_combo").mkdir(parents=True, exist_ok=True)
    (OUT / "later_combo" / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    combo_ranked = {r: fuse_order(shown[r], o["orders"], o["passes"], r, w) + merged[r][X.TOP:] for r in rows}

    frames = {"alone": pd.read_parquet(OUT / "later" / "per_case.parquet"), SYSTEM: pd.read_parquet(run / "eval" / SYSTEM / "per_case.parquet"),
              FT.RANKER: pd.read_parquet(run / "eval" / FT.RANKER / "per_case.parquet"), "list_agent_words": score_ranked(rows, agent_ranked), "combo": score_ranked(rows, combo_ranked)}
    frames["list_agent_words"].to_parquet(E.OUT / "per_case.parquet")
    frames["combo"].to_parquet(OUT / "later_combo" / "per_case.parquet")
    rule = json.loads((run / "HEADLINE_RULE.json").read_text(encoding="utf-8"))
    dates = NR.decision_dates(run / "NEW" / "release")
    headline = set(dates) - {c["query_id"] for c in rule["overlap_cases"]}
    after = {q for q, x in dates.items() if x["earliest"] and x["earliest"] > FT.CUTOFFS[KIND]} & headline
    out = {"agent": {"usage": res["usage"], "seconds": agent_s, "steps_mean": float(np.mean([c["steps"] for c in res["cases"]])),
                     "searches_mean": float(np.mean([c["searches"] for c in res["cases"]]))},
           "combo": {"usage": tot, "usd_list_equivalent": usd, "stats": stats, "w_all": w, "seconds": round(time.perf_counter() - t0, 1),
                     "reached_in_30": float(np.mean([bool({int(g) for g in d.specific_idx[r]} & set(shown[r])) for r in rows]))},
           "means": {}, "success_at_10": {k: float(f[f["query_id"].isin(headline)]["success_at_10"].mean()) for k, f in frames.items()}, "contrasts": {}}
    for name, keep in (("headline", headline), ("decided_after_cutoff", after)):
        out["means"][name] = {k: float(f[f["query_id"].isin(keep)]["ndcg_at_10"].mean()) for k, f in frames.items()}
        for a in ("list_agent_words", "combo"):
            for b in ("alone", SYSTEM, FT.RANKER):
                out["contrasts"][f"{name}: {a} - {b}"] = NR.pair(frames, a, b, "ndcg_at_10", 3, keep)
    A.write_json(OUT / "later_more.json", out)
    A.receipt("agents_later_more", out)
    print(json.dumps({k: out[k] for k in ("means", "success_at_10", "agent", "combo")}, indent=1, default=float), flush=True)
    return 0


def model_answers(rows: list[int], prompts: list[tuple[str, list[str], list[int]]], tag: str, sample: int = 0) -> dict[int, list[dict]]:
    """The model's 20 articles per case for ready prompts (prompt, ids in it, their rows), matched by id."""
    from uol.eval_lenses import llm_alone as LA
    from uol.round3 import luna
    schema = luna.batch_schema(LA.ITEM)
    res = luna.run_many([(pr, schema) for pr, _, _ in prompts], effort=LA.EFFORT, tag=tag, workers=4, sample=sample)
    out: dict[int, list[dict]] = {}
    for (prompt, ids, part), r in zip(prompts, res):
        got = luna.checked(r, prompt, schema, ids, effort=LA.EFFORT, tag=tag)
        for i, row in zip(ids, part):
            out[row] = got[i]["articulos"]
    return out


def repeat(smoke: int | None) -> int:
    """PLAN addendum 15: the same prompt asked two more times; the three lists merged by reciprocal rank fusion."""
    from uol.embedding_push import common as ep
    from uol.eval_lenses import llm_alone as LA
    pin_model()
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = sample_rows(smoke)
    first = json.loads((OUT / "alone" / "answers.json").read_text(encoding="utf-8"))
    built = []
    for s in range(0, len(rows), LA.BATCH):
        part = rows[s:s + LA.BATCH]
        prompt, ids, _sha = LA.build(part)
        built.append((prompt, ids, part))
    t0 = time.perf_counter()
    samples = [{r: first[str(r)]["articulos"] for r in rows}] + [model_answers(rows, built, LA.TAG, sample=s) for s in REPEATS]
    name = "repeat" + (f"_smoke{smoke}" if smoke else "")
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "answers.json").write_text(json.dumps([{str(k): v for k, v in a.items()} for a in samples], ensure_ascii=False), encoding="utf-8")
    lists = [{r: resolved(a[r], generic) for r in rows} for a in samples]
    ranked = {r: rrf([x[r] for x in lists]) for r in rows}
    same = float(np.mean([len(set(lists[0][r]) & set(lists[1][r])) / max(len(lists[0][r]), 1) for r in rows]))
    return finish(name, rows, ranked, {"samples": 1 + len(REPEATS), "overlap_of_two_samples": same, "union_size": float(np.mean([len(ranked[r]) for r in rows])),
                                       "seconds": round(time.perf_counter() - t0, 1)}, alone_frame(rows))


def alone_frame(rows: list[int]) -> dict[str, pd.DataFrame]:
    f = pd.read_parquet(OUT / "alone" / "per_case.parquet")
    return {"alone": f[f["row"].isin(rows)]}


def past_cases(rows: list[int], memory: list[int]) -> dict[int, list[int]]:
    """The ``EXAMPLES`` most similar memory cases by the encoder's stored query vectors: another fold, one per group."""
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    d = ep.release()
    Q = qe.query_vectors("c")
    mem = np.asarray(memory, dtype=np.int64)
    S = Q[np.asarray(rows)] @ Q[mem].T
    out: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        q, seen, got = d.queries[r], set(), []
        for j in np.argsort(-S[i], kind="stable"):
            m = int(mem[j])
            c = d.queries[m]
            if c.fold == q.fold or c.component_id == q.component_id or c.component_id in seen:
                continue
            seen.add(c.component_id)
            got.append(m)
            if len(got) == EXAMPLES:
                break
        out[r] = got
    return out


def examples(smoke: int | None) -> int:
    """PLAN addendum 15: the model with similar past cases from the other folds and the articles they cited."""
    import re
    from uol.embedding_push import common as ep
    from uol.eval_lenses import e6_agent as E, llm_alone as LA
    from uol.round3 import prompt_guard
    pin_model()
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = sample_rows(smoke)
    memory = [int(r) for r in ep.dev_rows() if len(d.specific_idx[int(r)])]
    past = past_cases(rows, memory)
    tpl = EXAMPLES_PROMPT.read_text(encoding="utf-8")
    clean = lambda r: re.sub(r"\s+", " ", d.queries[r].text).strip()
    built = []
    for s in range(0, len(rows), LA.BATCH):
        part = rows[s:s + LA.BATCH]
        ids = [f"c{i + 1:02d}" for i in range(len(part))]
        items, sanctioned = [], []
        for i, r in zip(ids, part):
            lines = []
            for m in past[r]:
                facts = clean(m)[:EXAMPLE_CHARS]
                cited = "; ".join(E.label(int(a)) for a in d.specific_idx[m])
                lines.append(f"- Hechos: {facts}\n  Artículos citados: {cited}")
                sanctioned += [facts, cited]
            items.append(f"[id={i}] {clean(r)}\nCasos anteriores parecidos:\n" + "\n".join(lines))
            sanctioned.append(clean(r))
        prompt = tpl.replace("{items}", "\n\n".join(items))
        prompt_guard.check(prompt, part, sanctioned=sanctioned, template=tpl)
        built.append((prompt, ids, part))
    t0 = time.perf_counter()
    answers = model_answers(rows, built, "llm_examples_dev")
    name = "examples" + (f"_smoke{smoke}" if smoke else "")
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "answers.json").write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    ranked = {r: resolved(answers[r], generic) for r in rows}
    copied = float(np.mean([len(set(ranked[r]) & {int(a) for m in past[r] for a in d.specific_idx[m]}) / max(len(ranked[r]), 1) for r in rows]))
    return finish(name, rows, ranked, {"examples": EXAMPLES, "share_of_answers_from_examples": copied, "seconds": round(time.perf_counter() - t0, 1)}, alone_frame(rows))


def save_system(sid: str, rows: list[int], ranked: dict[int, list[int]], meta: dict) -> dict:
    """A ranked list per case as a stored system: the same matrix, evaluator and measures as every other system."""
    from uol.embedding_push import common as ep
    from uol.final import harness, llm_jobs
    n = len(ep.release().candidate_ids)
    M = np.full((len(rows), n), -1e9, dtype=np.float64) - np.arange(n)[None, :] * 1e-6
    for i, r in enumerate(rows):
        for k, c in enumerate(ranked[r]):
            M[i, int(c)] = 1e6 - k
    llm_jobs.save_matrix(sid, np.asarray(rows, dtype=np.int64), M, meta)
    return harness.run(sid, "DEV", store=FT.STORE, force=True)


def all_rows() -> list[int]:
    return [int(r) for r in FT.spec_rows()]


def alone_full() -> int:
    """The model alone on every development case: the sample's answers are kept, the other cases are asked as they were."""
    from uol.eval_lenses import llm_alone as LA
    from uol.embedding_push import common as ep
    pin_model()
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = all_rows()
    first = json.loads((OUT / "alone" / "answers.json").read_text(encoding="utf-8"))
    rest = [r for r in rows if str(r) not in first]
    t0 = time.perf_counter()
    got = LA.run(rest)
    answers = {str(r): (first[str(r)] if str(r) in first else got[r]) for r in rows}
    (OUT / "alone_full").mkdir(parents=True, exist_ok=True)
    (OUT / "alone_full" / "answers.json").write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    res = save_system(ALONE_SYSTEM, rows, {r: resolved(answers[str(r)]["articulos"], generic) for r in rows},
                      {"model": MODEL, "recipe": "the facts only (eval_lenses.llm_alone), generic articles out, PLAN addendum 14"})
    A.receipt("agents_alone_full", {"cases": len(rows), "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10")})
    print(json.dumps({"cases": len(rows), "ndcg_at_10": res.get("ndcg_at_10")}), flush=True)
    return 0


def alone_models() -> int:
    """PLAN addendum 16: the model alone on the sample with every other model on the subscription route."""
    from uol.eval_lenses import llm_alone as LA
    from uol.embedding_push import common as ep
    from uol.round3 import luna
    from uol.final import codex
    luna.MODEL = "unset"
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = sample_rows(None)
    frames = system_frames(rows)
    out = {}
    for kind in OTHER_MODELS:
        cfg = FT.MODELS[kind]
        # each model is its own process (run_alone_models.sh): the codex route pins one model per process
        raise_if = os.environ.get("UOL_ALONE_KIND")
        if raise_if and raise_if != kind:
            continue
        pin = pin_model(cfg["model"])
        LA.TAG = f"llm_alone_dev_{kind}"
        t0 = time.perf_counter()
        ans = LA.run(rows)
        (OUT / f"alone_{kind}").mkdir(parents=True, exist_ok=True)
        (OUT / f"alone_{kind}" / "answers.json").write_text(json.dumps({str(k): v for k, v in ans.items()}, ensure_ascii=False), encoding="utf-8")
        f = score_ranked(rows, {r: resolved(ans[r]["articulos"], generic) for r in rows})
        f.to_parquet(OUT / f"alone_{kind}" / "per_case.parquet")
        out[kind] = {"model": cfg["model"], "mean_per_case": float(f["ndcg_at_10"].mean()), "success_at_10": float(f["success_at_10"].mean()),
                     "reranked_over_ranker_same_cases": float(A.per_case(FT.STORE, FT.effort_system(kind)).set_index("row").reindex(rows)["ndcg_at_10"].mean()),
                     "seconds": round(time.perf_counter() - t0, 1)}
        A.receipt(f"agents_alone_{kind}", out[kind])
    print(json.dumps(out, indent=1), flush=True)
    return 0


def combo_full(source: str = "alone_full") -> int:
    """The combination on every development case, from a stored list of the model's answers."""
    from uol.embedding_push import common as ep
    from uol.final import external as X
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = all_rows()
    answers = json.loads((OUT / source / "answers.json").read_text(encoding="utf-8"))
    rk = ranker_lists(rows)
    model = {r: resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    merged = {r: rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[KIND]}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    o = X.rerank_orders(rows, shown, lambda items: X.ask_claude(items, X.codex_api(cfg, "agents_combo_full", usage), stats, cfg))
    tot, usd = X.codex_usage(usage, cfg)
    out = OUT / "combo_full"
    out.mkdir(parents=True, exist_ok=True)
    (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    ranked = {r: fuse_order(shown[r], o["orders"], o["passes"], r, w) + merged[r][X.TOP:] for r in rows}
    res = save_system(COMBO_SYSTEM, rows, ranked, {"model": MODEL, "base": FT.RANKER, "w_all": w, "usage": tot, "usd_list_equivalent": usd, "source": source, **stats,
                                                   "recipe": "the model's list and the ranker's merged by RRF k 60, first 30 reranked, fused as the system, PLAN addendum 14"})
    A.receipt("agents_combo_full", {"cases": len(rows), "source": source, "usage": tot, "usd_list_equivalent": usd, "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10"), **stats})
    print(json.dumps({"cases": len(rows), "ndcg_at_10": res.get("ndcg_at_10"), "usd_list_equivalent": usd}), flush=True)
    return 0


VARIANTS = {"merged": "the merged list, no second reading", "fwd": "the combination's forward order alone, from its held answers",
            "self": "the model's own list read again, no ranker", "s0": "merged with the ranker without the encoder",
            "d60": "the first 60 of the merged list read", "low": "the second reading at low effort",
            "on": "merged with the list of the ranker plus an addition"}
VARIANT_SYSTEMS = {v: f"{KIND}_combo_{v}" for v in VARIANTS}
ON = "on"                           # rebuild PLAN step 3: an addition read in the system, its ranker in place of the ranker


def on_system(base: str) -> str:
    return f"{KIND}_combo_on_{base}"


def ask_some(ask):
    """``ask`` for the items that have candidates: a model's short list can leave a window empty."""
    def inner(items):
        got = ask([x for x in items if x[2]])
        return {**got, **{x[0]: [] for x in items if not x[2]}}
    return inner


def combo_variant(name: str, base: str | None = None) -> int:
    """One change to the combination on every development case, same list source and weight (PLAN addendum 20). ``on``:
    another ranker's list in place of the ranker's, everything else as the system (an addition read in the system)."""
    from uol.embedding_push import common as ep
    from uol.final import external as X
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = all_rows()
    answers = json.loads((OUT / "alone_full" / "answers.json").read_text(encoding="utf-8"))
    model = {r: resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    if name == "s0":
        want = {d.queries[r].query_id: r for r in rows}
        top = pd.read_parquet(FT.base_top200("S0"))
        rk = {want[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in top[top["query_id"].isin(want)].groupby("query_id")}
    else:
        rk = ranker_lists(rows, base if name == ON else FT.RANKER)
    merged = {r: rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    if name == "self":
        merged = {r: list(model[r]) for r in rows}
    depth = 60 if name == "d60" else X.TOP
    shown = {r: merged[r][:depth] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[KIND]}
    if name == "low":
        cfg = {**cfg, "effort": "low"}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    tag = f"on_{base}" if name == ON else name
    sid = on_system(base) if name == ON else VARIANT_SYSTEMS[name]
    out = OUT / f"combo_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    if name == "merged":
        ranked, tot, usd = merged, {}, 0.0
    else:
        if name == "fwd":
            o = {**json.loads((OUT / "combo_full" / "orders.json").read_text(encoding="utf-8")), "passes": ["fwd"]}
            tot, usd = {}, 0.0
        else:
            ask = ask_some(lambda items: X.ask_claude(items, X.codex_api(cfg, f"agents_combo_{tag}", usage), stats, cfg))
            o = FT.rerank_depth(rows, shown, ask, depth)
            tot, usd = X.codex_usage(usage, cfg)
            (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
        ranked = {r: fuse_order(shown[r], o["orders"], o["passes"], r, w) + merged[r][depth:] for r in rows}
    res = save_system(sid, rows, ranked, {"model": MODEL, "base": base if name == ON else "S0" if name == "s0" else FT.RANKER, "w_all": w, "usage": tot, "usd_list_equivalent": usd, **stats,
                                          "recipe": f"{VARIANTS[name]}{' ' + base if name == ON else ''}, PLAN addendum 20"})
    A.receipt(f"agents_combo_{tag}", {"cases": len(rows), "usage": tot, "usd_list_equivalent": usd, "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10"), **stats})
    print(json.dumps({"variant": tag, "cases": len(rows), "ndcg_at_10": res.get("ndcg_at_10"), "usd_list_equivalent": usd}), flush=True)
    return 0


FINAL_BASE = "ft_lean"                                    # the ranker (addendum 22)
FINAL_SYSTEM = f"{KIND}_combo_fwd_on_{FINAL_BASE}"
FINAL_CHECKS = {"merged": "the merged list, no second reading", "self": "the model's own list read again, forward order of the held answers",
                "both": "both orders, addendum 21's run", "noenc": "merged with the ranker without the encoder",
                "low": "the second reading at low effort", "d60": "the first 60 of the merged list read"}


def final_system(check: str | None = None) -> str:
    return FINAL_SYSTEM if check is None else f"{KIND}_final_{check}"


def final_variant(check: str | None = None) -> int:
    """PLAN addendum 22: the final system (``check`` None), derived from addendum 21's held answers with the forward order
    alone, or one change to it. Same model answers and weight; the forward order alone is asked where a reading is new."""
    from uol.embedding_push import common as ep
    from uol.final import external as X
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = all_rows()
    answers = json.loads((OUT / "alone_full" / "answers.json").read_text(encoding="utf-8"))
    model = {r: resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    rk = ranker_lists(rows, "ft_lean_noenc" if check == "noenc" else FINAL_BASE)
    merged = {r: rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    if check == "self":
        merged = {r: list(model[r]) for r in rows}
    depth = 60 if check == "d60" else X.TOP
    shown = {r: merged[r][:depth] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[KIND], **({"effort": "low"} if check == "low" else {})}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    sid = final_system(check)
    out = OUT / sid
    out.mkdir(parents=True, exist_ok=True)
    tot, usd, passes = {}, 0.0, ["fwd"]
    if check == "merged":
        ranked = merged
    else:
        if check in (None, "both"):
            o = json.loads((OUT / f"combo_on_{FINAL_BASE}" / "orders.json").read_text(encoding="utf-8"))
            passes = o["passes"] if check == "both" else ["fwd"]
        elif check == "self":
            o = json.loads((OUT / "combo_self" / "orders.json").read_text(encoding="utf-8"))
        else:
            ask = ask_some(lambda items: X.ask_claude(items, X.codex_api(cfg, f"agents_final_{check}", usage), stats, cfg))
            o = FT.rerank_depth(rows, shown, ask, depth, passes=("fwd",))
            tot, usd = X.codex_usage(usage, cfg)
            (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
        ranked = {r: fuse_order(shown[r], o["orders"], passes, r, w) + merged[r][depth:] for r in rows}
    recipe = "the final system, forward order of addendum 21's answers" if check is None else FINAL_CHECKS[check]
    res = save_system(sid, rows, ranked, {"model": MODEL, "base": "ft_lean_noenc" if check == "noenc" else FINAL_BASE, "w_all": w, "passes": passes,
                                          "depth": depth, "effort": cfg["effort"], "usage": tot, "usd_list_equivalent": usd, **stats,
                                          "recipe": f"{recipe}, PLAN addendum 22"})
    A.receipt(f"agents_{sid}", {"cases": len(rows), "usage": tot, "usd_list_equivalent": usd, "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10"), **stats})
    print(json.dumps({"system": sid, "cases": len(rows), "ndcg_at_10": res.get("ndcg_at_10"), "usd_list_equivalent": usd}), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["alone", "agent", "list_agent", "combo", "later", "later_more", "repeat", "examples", "alone_full", "combo_full", "alone_models", "combo_variant", "final"])
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--base", default=None, help="with --variant on: the ranker whose list replaces the ranker's")
    ap.add_argument("--check", choices=list(FINAL_CHECKS), default=None, help="with final: one change to the final system")
    ap.add_argument("--tool", choices=["words", "embedding"], default="embedding")
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--source", default="alone_full")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] would run {a.run} ({a.tool}) with {MODEL} on {a.smoke or 300} development cases into {OUT}")
        return 0
    os.environ.setdefault("UOL_LUNA_CACHE", str(A.ROOT / ".cache" / "footing" / "codex_noweb.sqlite"))
    if a.run == "later":
        return later()
    if a.run == "later_more":
        return later_more()
    if a.run in ("alone_full", "combo_full"):
        A.install_systems([A.spec_llm(FT.RANKER, s, "claude") for s in (ALONE_SYSTEM, COMBO_SYSTEM)], dry=False)
    if a.run == "final":
        A.install_systems([A.spec_llm(FINAL_BASE, final_system(c), "claude") for c in (None, *FINAL_CHECKS)], dry=False)
        D.activate(FT.STORE)
        return final_variant(a.check)
    if a.run == "combo_variant":
        if a.variant == ON and not a.base:
            raise SystemExit("--variant on needs --base")
        D.activate(FT.STORE)
        return combo_variant(a.variant, a.base)
    D.activate(FT.STORE)
    if a.run == "alone_full":
        return alone_full()
    if a.run == "combo_full":
        return combo_full(a.source)
    if a.run == "alone_models":
        return alone_models()
    if a.run == "combo":
        return combo(a.smoke)
    if a.run == "repeat":
        return repeat(a.smoke)
    if a.run == "examples":
        return examples(a.smoke)
    if a.run == "alone":
        return alone(a.smoke)
    return agent(a.tool, a.run == "list_agent", a.smoke)


if __name__ == "__main__":
    sys.exit(main())
