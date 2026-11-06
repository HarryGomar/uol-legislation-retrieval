"""The same model judging each candidate alone (examiner point, 2 Nov 2026).

Chapter 6 compares GPT-6.1 Sol reading the ranker's first 30 articles side by side (listwise, ``l1_luna``'s windows)
with cross-encoders and a pointwise judge that differ from it in size and training, so that comparison cannot separate
the way of reading from the strength of the model. Here GPT-6.1 Sol, at the same effort, through the same route, on the
same ranker's same first 30 articles and the same sample of 300 development cases (DEV-A300), gives for each candidate
alone the probability that the court cited it. The prompt (``prompts/P1_pointwise.txt``) keeps the listwise prompt's
framing and its candidate lines (``l1_luna.candidate_line``: law, number, 600-character excerpt). A call holds
``PER_CALL`` items of different cases at the same rank, so no two candidates of one case are ever read together. The 30
are reordered by probability, ties in the ranker's order, and the ranker's list follows from rank 31.

Compared in one family (record arithmetic, ``audit_d181_common.dev_family``) on the 300 cases: pointwise minus the
ranker, listwise minus pointwise (the listwise run of the record, both orders, ``gpt61sol_rerank_S0+qwen3_8b``).

    py -3.12 -m uol.final.pointwise run [--smoke N] --go      ask (cached, resumable), then score
    py -3.12 -m uol.final.pointwise score                     score the cached answers only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import time
from pathlib import Path

import pandas as pd

from uol.final import audit_d181_common as A, footing as F

ROOT = Path(__file__).resolve().parents[3]
PROMPT = Path(__file__).resolve().parent / "prompts" / "P1_pointwise.txt"
OUT = ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "pointwise"
WORK = A.WORK / "pointwise"
KIND, TOP, PER_CALL = "gpt61sol", 30, 10
LISTWISE = "gpt61sol_rerank_S0+qwen3_8b"
ITEM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probabilidad"],
               "properties": {"probabilidad": {"type": "integer"}}}


def setup():
    from uol.final import llm_agents as LG
    F.D.activate(F.STORE)
    luna = LG.pin_model(F.MODELS[KIND]["model"])
    WORK.mkdir(parents=True, exist_ok=True)
    luna.CACHE, luna.LEDGER = WORK / "cache.sqlite", WORK / "LUNA_LEDGER.jsonl"
    return luna


def rows_and_lists(smoke: int | None) -> tuple[list[int], dict[int, list[int]], dict[int, list[int]]]:
    from uol.final import external as X, llm_agents as LG
    rows = LG.sample_rows(smoke)
    t = pd.read_parquet(F.base_top200(F.RANKER))
    from uol.embedding_push import common as ep
    d = ep.release()
    want = {d.queries[r].query_id: r for r in rows}
    full = {want[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in t[t["query_id"].isin(want)].groupby("query_id")}
    top = X.top30(F.base_top200(F.RANKER), rows)
    assert all(full[r][:TOP] == top[r] for r in rows)
    return rows, top, full


def build(part: list[tuple[str, int, int]], tpl: str) -> str:
    from uol.embedding_push import common as ep
    from uol.round3 import l1_luna, prompt_guard
    import re
    d = ep.release()
    texts, lines = [], []
    for iid, row, cand in part:
        ln = l1_luna.candidate_line(cand, False)
        lines.append(ln)
        texts.append(f"[id={iid}]\nHechos: {re.sub(r'[ \t\r\n]+', ' ', d.queries[row].text).strip()}\nCandidato: {ln}")
    prompt = tpl.replace("{items}", (chr(10) * 2).join(texts))
    prompt_guard.check(prompt, [r for _, r, _ in part], sanctioned=lines, template=tpl)
    return prompt


def ask(rows: list[int], top: dict[int, list[int]]) -> tuple[dict[str, int], dict]:
    from uol.final import external as X
    from uol.round3 import luna as L
    tpl = PROMPT.read_text(encoding="utf-8")
    schema = L.batch_schema(ITEM_SCHEMA)
    cfg = {**X.SWEEP["gpt6sol"], **F.MODELS[KIND], "workers": int(os.environ.get("UOL_POINTWISE_WORKERS", "6"))}
    usage: dict[str, dict] = {}
    api = X.codex_api(cfg, "pointwise_dev", usage)
    items = [(f"r{r}k{k}", r, top[r][k]) for k in range(TOP) for r in rows]   # same rank, different cases, side by side
    got: dict[str, int] = {}

    def run(parts, sample):
        reqs = [{"custom_id": f"p{sample}_{i}", "prompt": build(p, tpl), "schema": schema, "model": cfg["model"],
                 "effort": cfg["effort"], **({"sample": sample} if sample else {})} for i, p in enumerate(parts)]
        res = api(reqs)
        for p, q in zip(parts, reqs):
            r = res.get(q["custom_id"]) or {}
            try:
                out = json.loads(r.get("text") or "{}")
            except ValueError:
                out = {}
            ids = {iid for iid, _, _ in p}
            for x in out.get("results", []):
                if isinstance(x, dict) and x.get("id") in ids and isinstance(x.get("probabilidad"), int):
                    got[x["id"]] = max(0, min(100, x["probabilidad"]))

    parts = [items[s:s + PER_CALL] for s in range(0, len(items), PER_CALL)]
    for s in range(0, len(parts), 60):                       # progress in steps of 60 calls (answered calls are cached)
        run(parts[s:s + 60], 0)
        print(f"{min(s + 60, len(parts))}/{len(parts)} calls, {len(got)} answers", flush=True)
    for attempt in range(1, 4):
        missing = [it for it in items if it[0] not in got]
        if not missing:
            break
        run([[it] for it in missing], attempt)
    tot, usd = X.codex_usage(usage, cfg)
    return got, {"usage": tot, "usd_list_equivalent": usd, "unanswered": sum(it[0] not in got for it in items)}


def score(rows: list[int], top: dict[int, list[int]], full: dict[int, list[int]], got: dict[str, int]) -> dict:
    from uol.final import llm_agents as LG
    ranked = {}
    for r in rows:
        p = [got.get(f"r{r}k{k}", 0) for k in range(TOP)]
        order = sorted(range(TOP), key=lambda k: (-p[k], k))   # ties keep the ranker's order
        ranked[r] = [top[r][k] for k in order] + full[r][TOP:]
    frames = {"pointwise": LG.score_ranked(rows, ranked), "ranker": LG.score_ranked(rows, {r: full[r] for r in rows})}
    lw = pd.read_parquet(ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store" / "runs" / LISTWISE / "DEV" / "per_case.parquet")
    frames["listwise"] = lw[lw["row"].isin(rows)][["query_id", "row", "component_id", "ndcg_at_10"]].reset_index(drop=True)
    assert len(frames["listwise"]) == len(rows)
    gm = {k: float(f.groupby("component_id")["ndcg_at_10"].mean().mean()) for k, f in frames.items()}
    case = {k: float(f["ndcg_at_10"].mean()) for k, f in frames.items()}   # per case, as the sample table prints its scores
    fam = A.dev_family("pointwise", [("pointwise", frames["pointwise"], "ranker", frames["ranker"]),
                                     ("listwise", frames["listwise"], "pointwise", frames["pointwise"])], population="DEV-A300")
    ties = sum(len(set(got.get(f"r{r}k{k}", 0) for k in range(TOP))) < TOP for r in rows)
    return {"scores": gm, "scores_case": case, "family": fam, "cases_with_ties": ties, "ranked": {str(r): v[:TOP] for r, v in ranked.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--smoke", type=int)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "run" and not a.go:
        print(f"[dry] would ask {F.MODELS[KIND]['model']} for {TOP} pointwise probabilities on {a.smoke or 300} cases, {PER_CALL} items per call")
        print(f"[dry] codex sign-in: {A.codex_signed_in()}")
        return 0
    setup()
    rows, top, full = rows_and_lists(a.smoke)
    t0 = time.perf_counter()
    if a.step == "run":
        s = A.codex_signed_in()
        if not s["ok"]:
            raise RuntimeError(f"Codex CLI not signed in: {s}")
        got, meta = ask(rows, top)
    else:
        got, meta = ask_cached(rows, top)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("answers.json" if not a.smoke else f"answers_smoke{a.smoke}.json")).write_text(json.dumps(got), encoding="utf-8")
    res = score(rows, top, full, got)
    res.update({"step": "pointwise", "rows": len(rows), "model": F.MODELS[KIND]["model"], "effort": F.MODELS[KIND]["effort"],
                "per_call": PER_CALL, "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
                "seconds": round(time.perf_counter() - t0, 1), **meta,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    A.write_json(OUT / ("result.json" if not a.smoke else f"result_smoke{a.smoke}.json"), res)
    if not a.smoke:
        A.receipt("pointwise", {k: v for k, v in res.items() if k not in ("ranked", "family")})
    c = res["family"].get("contrasts", res["family"])
    print(json.dumps({"scores": res["scores"], "unanswered": meta.get("unanswered"), "usd": meta.get("usd_list_equivalent"),
                      "contrasts": {k: {x: v.get(x) for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in c.items() if isinstance(v, dict) and "delta" in v}},
                     indent=1, default=float))
    return 0


def ask_cached(rows, top):
    """The same calls, answered from the cache only (``score``): a call not in the cache stops the step."""
    from uol.round3 import luna as L
    orig = L.call

    def cached_only(*a, **k):
        res = orig(*a, **k)
        if not res.get("cached"):
            raise RuntimeError("score found a call that is not cached; run first")
        return res
    L.call = cached_only
    try:
        return ask(rows, top)
    finally:
        L.call = orig


if __name__ == "__main__":
    raise SystemExit(main())
