"""One test run of the ranker, the ranker without the encoder and the system on the ranker (common_footing PLAN addendum 22).

``test_set`` holds the frozen record of D198 in ``footing_20261101/test``; this module writes ``test_final`` beside it and
reads ``test`` only for its rows and the model's stored answers. Each step is its own process. Only ``score`` opens the
labels, and it runs once, after FINAL_SYSTEM.json names the systems.

    fit      ``ft_lean`` and ``ft_lean_noenc`` on all development cases, as ``S0+qwen3_8b`` was (``new_cases_fit.fit``)
    rank     each saved model against its batch output (``new_cases_fit.dryrun``), the later-case control of
             ``test_set.rank``, then both rankers on the 352 test cases and the 46 later cases, no labels
    freeze   FINAL_SYSTEM.json: the systems scored and the frozen settings of D198
    rerank   the model's stored list and the ranker's merged, the first 30 read once in forward order, no labels
    score    the systems of FINAL_SYSTEM.json, once, on the 352 cases and the 46 together

    py -3.12 -m uol.final.test_final fit --go
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A, test_set as TS

ROOT = TS.ROOT
OUT = ROOT / "experiments" / "runs" / "footing_20261101" / "test_final"
T0 = TS.OUT                                          # the frozen record: rows and the model's answers are read, nothing written
MODELS = OUT / "models"
WORK = ROOT / ".cache" / "footing" / "test_final_fit"
RANKERS = ("ft_lean", "ft_lean_noenc")
NAMES = {"ranker": "ft_lean", "ranker_noenc": "ft_lean_noenc"}
FINAL = OUT / "FINAL_SYSTEM.json"
SCORED = OUT / "scored.json"
CONTRASTS = (("combo", "alone"), ("combo", "ranker"), ("ranker", "ranker_noenc"))


def top_file(sid: str) -> Path:
    return OUT / f"ranker_top200_{sid}.json"


def fit() -> int:
    from uol.final import new_cases_fit as NF, new_cases_steps as NS
    A.need_threads()
    A.install_systems([s for s in FT.specs() if s["id"] in RANKERS], dry=False)
    links = NS.link_fit_features(WORK)
    topics = A.seed_topic_cache(TS.AUDIT_NEW / "models" / "topic", WORK)
    t0 = time.perf_counter()
    res = NF.fit(MODELS, WORK, RANKERS)
    old = json.loads((TS.AUDIT_NEW / "models" / A.NEW_SYS / "info.json").read_text(encoding="utf-8"))
    for sid in RANKERS:
        info = res["rankers"][sid]
        want = {s for s in old["feature_ids"]} - set(FT.LEAN_DROP) - (set(FT.ENCODER) if sid == "ft_lean_noenc" else set())
        if set(info["feature_ids"]) != want:
            raise RuntimeError(f"{sid}: sources {info['feature_ids']}, expected {sorted(want)}")
        if {k: v["sha256"] for k, v in info["topics"].items()} != {k: v["sha256"] for k, v in old["topics"].items()}:
            raise RuntimeError(f"{sid}: topic models differ from the ranker of the record")
    A.receipt("test_final_fit", {"d58_links": links, "topic_seed": topics, "seconds": round(time.perf_counter() - t0, 1),
                                 "rankers": {s: {k: r[k] for k in ("feature_ids", "train_pairs", "feature_columns", "fit_seconds", "files")}
                                             for s, r in res["rankers"].items()}})
    return 0


def _later_sources():
    from uol.final import new_cases_sources as NSrc, new_cases_steps as NS
    sm, blocks, _ = NS._sources(TS.AUDIT_NEW)
    nr = NSrc.new_rows(TS.AUDIT_NEW / "release_ext")
    if [int(x["row"]) for x in nr] != [int(r) for r in sm["rows"]]:
        raise RuntimeError("the later cases' sources are not in the order of their rows")
    texts = {q["query_id"]: q["text"] for q in NSrc._load(TS.AUDIT_NEW / "release_ext" / "queries.jsonl")[NSrc.N_V11:]}
    later = [int(r) for r in pd.read_parquet(TS.FOOTING_NEW / "eval" / A.NEW_SYS / "per_case.parquet")["row"]]
    return nr, blocks, texts, later


def rank() -> int:
    from uol.embedding_push import common as ep
    from uol.facts_to_law import metrics
    from uol.final import new_cases_fit as NF, timing
    A.need_threads()
    dry = NF.dryrun(MODELS, WORK, OUT / "checks", RANKERS)
    if not dry["pass"]:
        raise RuntimeError(f"dry run: {dry['fails']}")
    d = ep.release()
    rf = TS.rows_file()
    exclude, key = timing.eval_view()

    def top(S: np.ndarray) -> list[int]:
        return [int(c) for c in metrics.evaluate_matrix(np.asarray(S, dtype=np.float64)[None], [[]], exclude_idx=exclude, top_k=200, tie_order=key)[0]["topk"]]

    pop = NF.dev_population()
    nr, blocks, texts, later = _later_sources()
    at = {int(x["row"]): i for i, x in enumerate(nr)}

    def ranker(sid: str, models: Path):
        info = json.loads((models / sid / "info.json").read_text(encoding="utf-8"))
        for n, h in info["files"].items():
            if A.sha256(models / sid / n) != h:
                raise RuntimeError(f"{sid}/{n}: sha256 differs from info.json")
        with (models / sid / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        rk = NF.NewCaseRanker(NF.prepared(sid, pop), model, NF.load_topic_models(models, info), None)
        if not np.array_equal(np.asarray(rk.table[NF.LOCKBOX], dtype=np.float32), np.load(models / "instrument_prior_lockbox.npy")):
            raise RuntimeError("the all-development instrument prior differs from the fit's")
        return rk, info

    def later_row(rk, r: int) -> np.ndarray:
        i = at[r]
        return rk.rank(texts[nr[i]["query_id"]], {s: np.asarray(blocks[s][i]) for s in rk.sources}, NF.LOCKBOX, m1_cols=None)["row"]

    # control: the record's ranker on every later case through this path must rank as the stored run did
    rk, _ = ranker(A.NEW_SYS, TS.AUDIT_NEW / "models")
    stored = pd.read_parquet(TS.FOOTING_NEW / "eval" / A.NEW_SYS / "top200.parquet")
    qid = {int(x["row"]): x["query_id"] for x in nr}
    ctl = []
    for r in later:
        want = [int(c) for c in stored[stored["query_id"] == qid[r]].sort_values("rank")["candidate_idx"]]
        ctl.append({"row": r, "top30_equal": top(later_row(rk, r))[:30] == want[:30]})
    A.write_json(OUT / "control_rank.json", ctl)
    if not all(c["top30_equal"] for c in ctl):
        raise RuntimeError(f"the path does not reproduce the later cases: {sum(not c['top30_equal'] for c in ctl)} differ")
    del rk

    out = {"control_later_cases": len(ctl), "dry_run": {s: {k: v for k, v in r.items() if k != "cases"} for s, r in dry["rankers"].items()}}
    for sid in RANKERS:
        rk, info = ranker(sid, MODELS)
        files = {s: np.load(TS.SOURCE_SCORES / s / "scores.npy", mmap_mode="r") for s in rk.sources}
        t0 = time.perf_counter()
        lists = {}
        for i, (r, text) in enumerate(zip(rf["rows"], rf["texts"])):
            lists[int(r)] = top(rk.rank(text, {s: np.asarray(m[r]) for s, m in files.items()}, NF.LOCKBOX, m1_cols=None)["row"])
            if i % 50 == 0:
                print(f"{sid} {i}/{len(rf['rows'])} {time.perf_counter() - t0:.0f} s", flush=True)
        for r in later:
            lists[r] = top(later_row(rk, r))
        A.write_json(top_file(sid), {str(r): v for r, v in lists.items()})
        out[sid] = {"cases": len(lists), "seconds": round(time.perf_counter() - t0, 1), "sources": rk.sources, "model_sha256": info["files"]["model.pkl"]}
        del rk, files
    A.receipt("test_final_rank", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def freeze() -> int:
    """Written after the development runs of addendum 22 and before any reading of the test cases."""
    from uol.final import llm_agents as LG
    if FINAL.exists():
        raise FileExistsError(f"{FINAL} exists")
    old = json.loads(TS.FINAL.read_text(encoding="utf-8"))
    rec = {"systems": ["combo", "alone", "ranker", "ranker_noenc"], "dev_system": LG.FINAL_SYSTEM, "rankers": NAMES,
           **{k: old[k] for k in ("passes", "effort", "depth", "w_all", "rrf_k")},
           "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "basis": "PLAN addendum 22: the D198 settings over the 8-source ranker; one score, reported whatever it shows"}
    A.write_json(FINAL, rec)
    print(json.dumps(rec, indent=1), flush=True)
    return 0


def rerank() -> int:
    """The final system on the test and later cases: the forward order alone is asked, as the frozen settings read."""
    if not FINAL.exists():
        raise RuntimeError("no FINAL_SYSTEM.json")
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    if final["passes"] != ["fwd"] or final["depth"] != 30:
        raise RuntimeError(f"unexpected settings {final}")
    luna_out = OUT / "llm"
    LG = TS._llm()
    from uol.round3 import luna
    luna.CACHE, luna.LEDGER = luna_out / "cache.sqlite", luna_out / "LUNA_LEDGER.jsonl"
    luna.CACHE.parent.mkdir(parents=True, exist_ok=True)
    from uol.embedding_push import common as ep
    from uol.final import external as X
    from uol.final import new_cases_sources as NSrc
    NSrc.activate_ext(TS.FOOTING_NEW / "release_ext", OUT / "logs")   # the later rows' facts; logs here, not in test/
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    test = TS.rows_file()["rows"]
    _, _, _, later = _later_sources()
    rows = [*test, *later]
    answers = {**json.loads((T0 / "alone_answers.json").read_text(encoding="utf-8")),
               **{k: v for k, v in json.loads((LG.OUT / "later" / "answers.json").read_text(encoding="utf-8")).items() if int(k) in set(later)}}
    rk = {int(k): v for k, v in json.loads(top_file(NAMES["ranker"]).read_text(encoding="utf-8")).items()}
    model = {r: LG.resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / LG.SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    if w != final["w_all"]:
        raise RuntimeError(f"weight {w}, frozen {final['w_all']}")
    merged = {r: LG.rrf([model[r], [c for c in rk[r] if c not in generic]], k=final["rrf_k"]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[LG.KIND]}
    if cfg["effort"] != final["effort"]:
        raise RuntimeError(f"effort {cfg['effort']}, frozen {final['effort']}")
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    ask = LG.ask_some(lambda items: X.ask_claude(items, X.codex_api(cfg, "test_final_combo", usage), stats, cfg))
    o = FT.rerank_depth(rows, shown, ask, X.TOP, passes=("fwd",))
    tot, usd = X.codex_usage(usage, cfg)
    (OUT / "combo_orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    ranked = {r: LG.fuse_order(shown[r], o["orders"], final["passes"], r, w) + merged[r][X.TOP:] for r in rows}
    A.write_json(OUT / "combo_ranked.json", {str(r): ranked[r] for r in rows})
    A.receipt("test_final_combo", {"cases": len(rows), "w_all": w, "passes": final["passes"], "usage": tot, "usd_list_equivalent": usd,
                                   "seconds": round(time.perf_counter() - t0, 1), **stats})
    return 0


def score() -> int:
    """The labels, once. The result files are written before anything is printed."""
    from uol.final import llm_agents as LG, measures, new_cases_report as NR
    from uol.round3 import common as C
    if SCORED.exists():
        raise RuntimeError("scored once")
    if not FINAL.exists():
        raise RuntimeError("no FINAL_SYSTEM.json")
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    from uol.final import context, new_cases_sources as NSrc
    NSrc.activate_ext(TS.FOOTING_NEW / "release_ext", OUT / "logs")
    context.apply_label_corrections()
    context.install_labelled_evaluate()
    d = ep_release()
    C.lockbox_read("final_score", f"addendum 22: one score of {final['systems']}", Path(__file__))
    generic = {int(i) for i in d.generic_idx}
    rf = TS.rows_file()
    test = rf["rows"]
    if [d.queries[r].query_id for r in test] != rf["query_ids"]:
        raise RuntimeError("the test rows moved in the extended release")
    later, ll, dates = TS.later_lists(d, final["passes"])
    dates = {**TS.test_dates(rf["query_ids"]), **dates}
    rows = test + later
    load = lambda p: {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    answers = json.loads((T0 / "alone_answers.json").read_text(encoding="utf-8"))
    lists = {"combo": load(OUT / "combo_ranked.json"), **{k: load(top_file(s)) for k, s in NAMES.items()},
             "alone": {**{r: LG.resolved(answers[str(r)]["articulos"], generic) for r in test}, **{r: ll["alone"][r] for r in later}}}
    n = len(d.candidate_ids)
    frames = {}
    for name in final["systems"]:
        frames[name], _ = TS.measure(rows, TS.matrix(rows, {r: lists[name][r] for r in rows}, n), True)
        frames[name]["publication_date"] = frames[name]["query_id"].map(dates)
    want = pd.read_parquet(T0 / "per_case_alone.parquet").set_index("query_id")["ndcg_at_10"]
    got = frames["alone"].set_index("query_id")["ndcg_at_10"].reindex(want.index)
    control = {"alone_reproduces_test": float(np.abs(got.to_numpy() - want.to_numpy()).max())}
    first = frames["combo"]
    cut = FT.CUTOFFS[LG.KIND]
    parts = {"all": set(first["query_id"]), "without_later_46": set(rf["query_ids"]) & set(first["query_id"]),
             "after_model_cutoff": set(first[first["publication_date"] > cut]["query_id"]),
             "before_model_cutoff": set(first[first["publication_date"] <= cut]["query_id"])}
    res = {"final_system": final, "cutoff": cut, "control": control, "cases": {k: len(v) for k, v in parts.items()}, "summaries": {}, "contrasts": {},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for part, keep in parts.items():
        res["summaries"][part] = {k: measures.summary(f[f["query_id"].isin(keep)]) for k, f in frames.items()}
        for a, b in CONTRASTS:
            res["contrasts"][f"{part}: {a} - {b}"] = NR.pair(frames, a, b, "ndcg_at_10", len(CONTRASTS), keep)
    for name, f in frames.items():
        f.to_parquet(OUT / f"per_case_{name}.parquet", index=False)
    A.write_json(SCORED, res)
    print(json.dumps({"control": control, "cases": res["cases"],
                      "summaries": {p: {k: {m: v.get(m) for m in ("ndcg_at_10", "success_at_10", "n_cases", "n_groups")} for k, v in s.items()} for p, s in res["summaries"].items()},
                      "contrasts": {k: {x: v.get(x) for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in res["contrasts"].items() if v}}, indent=1, default=float), flush=True)
    return 0


def ep_release():
    from uol.embedding_push import common as ep
    d = ep.release()
    if not getattr(d, "_uol_final_corrected", False):
        raise RuntimeError("the corrected labels are not active")
    return d


STEPS = {"fit": fit, "rank": rank, "freeze": freeze, "rerank": rerank, "score": score}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    return STEPS[a.step]()


if __name__ == "__main__":
    sys.exit(main())
