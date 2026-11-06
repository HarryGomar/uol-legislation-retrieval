"""The final system once on the closed test set (D196; common_footing PLAN addendum 17).

Each step is its own process. Only ``score`` opens the labels, and it runs once, after FINAL_SYSTEM.json is written.

    prepare   the 352 cases' rows and texts, from the full release (local profile)
    control   the scoring path on stored development rows must reproduce the stored summary
    rank      the all-development ranker on these rows, no labels (bundle profile, as ``audit_d181_new.score``)
    alone     the model's list on these rows, no labels
    rerank    the model's and the ranker's lists merged, the first 30 reranked, fused as on development, no labels
    control_later  the same path on the 46 cases published after 11 Sep 2026, whose labels are open
    score     the systems named in FINAL_SYSTEM.json, once, on the 352 cases and the 46 together

    py -3.12 -m uol.final.test_set prepare --go
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "runs" / "footing_20261101" / "test"
SOURCE_SCORES = ROOT / "experiments" / "runs" / "embedding_push_20261004" / "scores"
AUDIT_NEW = ROOT / "experiments" / "runs" / "audit_d181_20261028" / "new"     # the ranker's saved models and the later cases' sources
AUDIT_WORK = ROOT / ".cache" / "audit_d181" / "new_score"
FOOTING_NEW = ROOT / "experiments" / "runs" / "footing_20261101" / "new"
SOURCES = ROOT / "data" / "derived" / "facts_to_law_v2" / "sjf_structured_sources_v1" / "sources.jsonl"
FINAL = OUT / "FINAL_SYSTEM.json"
SCORED = OUT / "scored.json"
CONTROL_CASES = 5
COUNT_LIMIT = (340, 360)


def rows_file() -> dict:
    return json.loads((OUT / "rows.json").read_text(encoding="utf-8"))


def prepare() -> int:
    from uol.embedding_push import common as ep
    from uol.final import context
    from uol.round3 import common as C
    context.active()
    if not ep._lockbox_allowed():
        raise RuntimeError("the release's FREEZE.json does not allow the closed test set")
    C.lockbox_read("final_score", "rows and texts of the closed test set, no labels", Path(__file__))
    rows = [int(r) for r in ep.population_rows("P_spec", ep.lockbox_rows())]
    if not COUNT_LIMIT[0] <= len(rows) <= COUNT_LIMIT[1]:
        raise RuntimeError(f"{len(rows)} test cases")
    d = ep.release()
    OUT.mkdir(parents=True, exist_ok=True)
    A.write_json(OUT / "rows.json", {"rows": rows, "query_ids": [d.queries[r].query_id for r in rows], "texts": [d.queries[r].text for r in rows],
                                     "candidate_ids_sha256": hashlib.sha256("\n".join(d.candidate_ids).encode()).hexdigest()})
    print(f"{len(rows)} cases, {len({d.queries[r].component_id for r in rows})} groups", flush=True)
    return 0


def measure(rows: list[int], M: np.ndarray, allow: bool) -> tuple[pd.DataFrame, dict]:
    """``new_cases_steps.evaluate``'s reading: the project evaluator, P_spec, specific view, top 200."""
    from uol.embedding_push import common as ep
    from uol.final import measures
    d = ep.release()
    unseen = measures.unseen_mask(ep.dev_rows())
    res = ep.evaluate(M, np.asarray(rows, dtype=np.int64), "P_spec", top_k=measures.TOP_K, allow_lockbox=allow)
    pos = {r: j for j, r in enumerate(rows)}
    sub = M[[pos[int(r)] for r in res["rows"]]]
    per = measures.per_case(res["per_query"], res["rows"], sub, specific=True, unseen=unseen)
    frame = pd.DataFrame([{"query_id": d.queries[int(r)].query_id, "row": int(r), "component_id": d.queries[int(r)].component_id, **rec}
                          for r, rec in zip(res["rows"], per)])
    return frame, measures.summary(frame)


def matrix(rows: list[int], ranked: dict[int, list[int]], n: int) -> np.ndarray:
    """As ``llm_agents.save_system``: a ranked list per case as a full score row."""
    M = np.full((len(rows), n), -1e9, dtype=np.float64) - np.arange(n)[None, :] * 1e-6
    for i, r in enumerate(rows):
        for k, c in enumerate(ranked[r]):
            M[i, int(c)] = 1e6 - k
    return M


def control() -> int:
    """The scoring path on the stored system's development lists must give its stored summary."""
    from uol.embedding_push import common as ep
    from uol.final import context
    context.active()
    d = ep.release()
    sid = FT.effort_system("gpt61sol")
    run = FT.STORE / "runs" / sid / "DEV"
    top = pd.read_parquet(run / "top200.parquet")
    index = {d.queries[int(r)].query_id: int(r) for r in ep.dev_rows()}
    ranked = {index[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in top.groupby("query_id")}
    rows = sorted(ranked)
    _, got = measure(rows, matrix(rows, ranked, len(d.candidate_ids)), False)
    want = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    res = {k: {"here": got.get(k), "stored": want.get(k)} for k in ("ndcg_at_10", "success_at_10", "success_at_1", "law_at_1")}
    A.write_json(OUT / "control_scoring.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if abs(got["ndcg_at_10"] - want["ndcg_at_10"]) > 1e-4:
        raise RuntimeError("the scoring path does not reproduce the stored summary")
    return 0


def rank() -> int:
    """``audit_d181_new.score``'s ranker branch on the test rows: their sources from the score files, no labels."""
    from uol.embedding_push import common as ep
    from uol.facts_to_law import metrics
    from uol.final import new_cases_fit as NF, new_cases_sources as NSrc, new_cases_steps as NS, timing
    A.need_threads()
    A.install_systems(A.new_specs(), dry=False)
    NS.link_fit_features(AUDIT_WORK)
    NF.activate(AUDIT_WORK)
    d = ep.release()
    rf = rows_file()
    if rf["candidate_ids_sha256"] != hashlib.sha256("\n".join(d.candidate_ids).encode()).hexdigest():
        raise RuntimeError("the catalogue differs from the one the rows were read with")
    sid, models = A.NEW_SYS, AUDIT_NEW / "models"
    info = json.loads((models / sid / "info.json").read_text(encoding="utf-8"))
    for n, h in info["files"].items():
        if A.sha256(models / sid / n) != h:
            raise RuntimeError(f"{sid}/{n}: sha256 differs from info.json")
    with (models / sid / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    sc = NF.prepared(sid, NF.dev_population())
    rk = NF.NewCaseRanker(sc, model, NF.load_topic_models(models, info), None)
    if not np.array_equal(np.asarray(rk.table[NF.LOCKBOX], dtype=np.float32), np.load(models / "instrument_prior_lockbox.npy")):
        raise RuntimeError("the all-development instrument prior differs from the fit's")
    exclude, key = timing.eval_view()

    def top(S: np.ndarray) -> list[int]:
        return [int(c) for c in metrics.evaluate_matrix(np.asarray(S, dtype=np.float64)[None], [[]], exclude_idx=exclude, top_k=200, tie_order=key)[0]["topk"]]

    # control: the same path on later cases must rank as the stored run did
    sm, blocks, pairs = NS._sources(AUDIT_NEW)
    nr = NSrc.new_rows(AUDIT_NEW / "release_ext")
    texts = {q["query_id"]: q["text"] for q in NSrc._load(AUDIT_NEW / "release_ext" / "queries.jsonl")[NSrc.N_V11:]}
    stored = np.load(AUDIT_NEW / "scores" / sid / "scores.npy")
    ctl = []
    for i in range(CONTROL_CASES):
        src = {s: (pairs[nr[i]["row"]] if s == "colbert_bge" else np.asarray(blocks[s][i])) for s in rk.sources}
        got = rk.rank(texts[nr[i]["query_id"]], src, NF.LOCKBOX, m1_cols=None)["row"]
        ctl.append({"case": nr[i]["query_id"], "top30_equal": top(got)[:30] == top(stored[i])[:30], "max_abs": float(np.nanmax(np.abs(got - stored[i])))})
    A.write_json(OUT / "control_rank.json", ctl)
    print(json.dumps(ctl), flush=True)
    if not all(c["top30_equal"] for c in ctl):
        raise RuntimeError("the ranker does not reproduce the later cases")

    files = {s: np.load(SOURCE_SCORES / s / "scores.npy", mmap_mode="r") for s in rk.sources}
    rows = rf["rows"]
    S = np.full((len(rows), len(d.candidate_ids)), np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    for i, (r, text) in enumerate(zip(rows, rf["texts"])):
        src = {s: np.asarray(m[r]) for s, m in files.items()}
        S[i] = rk.rank(text, src, NF.LOCKBOX, m1_cols=None)["row"]
        if i % 50 == 0:
            print(f"{i}/{len(rows)} {time.perf_counter() - t0:.0f} s", flush=True)
    np.save(OUT / "ranker_scores.npy", S)
    A.write_json(OUT / "ranker_top200.json", {str(r): top(S[i]) for i, r in enumerate(rows)})
    A.receipt("test_rank", {"cases": len(rows), "seconds": round(time.perf_counter() - t0, 1), "scores_sha256": A.sha256(OUT / "ranker_scores.npy"),
                            "model_sha256": info["files"]["model.pkl"], "control": ctl})
    return 0


def _llm():
    from uol.embedding_push import common as ep
    from uol.final import context, llm_agents as LG
    from uol.round3 import common as C
    context.active()
    luna = LG.pin_model()
    luna.CACHE = OUT / "llm" / "cache.sqlite"
    luna.LEDGER = OUT / "llm" / "LUNA_LEDGER.jsonl"
    luna.CACHE.parent.mkdir(parents=True, exist_ok=True)
    none = np.empty(0, dtype=np.int64)
    C.lockbox_rows = ep.lockbox_rows = lambda: none              # the runners refuse these rows; no label is read in this process
    return LG


def alone() -> int:
    LG = _llm()
    from uol.eval_lenses import llm_alone as LA
    LA.TAG = "llm_alone_test"
    rows = rows_file()["rows"]
    t0 = time.perf_counter()
    answers = LA.run(rows)
    (OUT / "alone_answers.json").write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    A.receipt("test_alone", {"cases": len(rows), "model": LG.MODEL, "seconds": round(time.perf_counter() - t0, 1)})
    return 0


def rerank() -> int:
    """``llm_agents.combo_full`` on the test rows: nothing tuned here. Both orders are asked, as on the development cases,
    and the frozen settings (``FINAL_SYSTEM.json`` ``passes``) decide which are fused."""
    if not FINAL.exists():
        raise RuntimeError("no FINAL_SYSTEM.json: freeze first, its settings decide the reading")
    passes = json.loads(FINAL.read_text(encoding="utf-8"))["passes"]
    LG = _llm()
    from uol.embedding_push import common as ep
    from uol.final import external as X
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = rows_file()["rows"]
    answers = json.loads((OUT / "alone_answers.json").read_text(encoding="utf-8"))
    rk = {int(k): v for k, v in json.loads((OUT / "ranker_top200.json").read_text(encoding="utf-8")).items()}
    model = {r: LG.resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / LG.SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    merged = {r: LG.rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[LG.KIND]}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    o = X.rerank_orders(rows, shown, lambda items: X.ask_claude(items, X.codex_api(cfg, "test_combo", usage), stats, cfg))
    tot, usd = X.codex_usage(usage, cfg)
    (OUT / "combo_orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    ranked = {r: LG.fuse_order(shown[r], o["orders"], passes, r, w) + merged[r][X.TOP:] for r in rows}
    A.write_json(OUT / "combo_ranked.json", {str(r): ranked[r] for r in rows})
    A.receipt("test_combo", {"cases": len(rows), "w_all": w, "passes": passes, "usage": tot, "usd_list_equivalent": usd, "seconds": round(time.perf_counter() - t0, 1), **stats})
    return 0


def extended():
    """The release with the later rows appended, on the labels of every other reading."""
    from uol.embedding_push import common as ep
    from uol.final import context, new_cases_sources as NSrc
    NSrc.activate_ext(FOOTING_NEW / "release_ext", OUT / "logs")
    context.apply_label_corrections()
    context.install_labelled_evaluate()
    d = ep.release()
    if not getattr(d, "_uol_final_corrected", False):
        raise RuntimeError("the corrected labels are not active")
    return d


def later_lists(d, passes: list[str] | None = None) -> tuple[list[int], dict[str, dict[int, list[int]]], dict[str, str]]:
    """The cases published after 11 Sep 2026, already through the same frozen pipeline (``llm_agents.later``, ``later_more``): their lists rebuilt from what was stored.
    ``passes``: the orders fused (the frozen settings); None keeps the stored ones, as ``control_later`` checks."""
    from uol.final import external as X, llm_agents as LG
    generic = {int(i) for i in d.generic_idx}
    base = pd.read_parquet(FOOTING_NEW / "eval" / A.NEW_SYS / "per_case.parquet")
    rows = [int(r) for r in base["row"]]
    dates = dict(zip(base["query_id"], base["publication_date"]))
    index = {d.queries[r].query_id: r for r in rows}
    top = pd.read_parquet(FOOTING_NEW / "eval" / A.NEW_SYS / "top200.parquet")
    rk = {index[q]: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in top[top["query_id"].isin(index)].groupby("query_id")}
    answers = json.loads((LG.OUT / "later" / "answers.json").read_text(encoding="utf-8"))
    model = {r: LG.resolved(answers[str(r)]["articulos"], generic) for r in rows}
    o = json.loads((LG.OUT / "later_combo" / "orders.json").read_text(encoding="utf-8"))
    w = float(json.loads((A.WORK / "dev" / "llm" / LG.SYSTEM / "meta.json").read_text(encoding="utf-8"))["w_all"])
    merged = {r: LG.rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    combo = {r: LG.fuse_order(merged[r][:X.TOP], o["orders"], o["passes"] if passes is None else passes, r, w) + merged[r][X.TOP:] for r in rows}
    return rows, {"ranker": rk, "alone": model, "combo": combo}, dates


def control_later() -> int:
    """The labels of the later cases are already open: this path must give their stored scores."""
    from uol.final import llm_agents as LG
    d = extended()
    rows, lists, _ = later_lists(d)
    stored = {"ranker": FOOTING_NEW / "eval" / A.NEW_SYS / "per_case.parquet", "alone": LG.OUT / "later" / "per_case.parquet", "combo": LG.OUT / "later_combo" / "per_case.parquet"}
    res = {}
    for name, ranked in lists.items():
        f, _ = measure(rows, matrix(rows, ranked, len(d.candidate_ids)), True)
        want = pd.read_parquet(stored[name]).set_index("query_id")["ndcg_at_10"].reindex(f["query_id"])
        res[name] = {"here": float(f["ndcg_at_10"].mean()), "stored": float(want.mean()), "largest_case_difference": float(np.abs(f["ndcg_at_10"].to_numpy() - want.to_numpy()).max())}
    A.write_json(OUT / "control_later.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if max(v["largest_case_difference"] for v in res.values()) > 1e-6:
        raise RuntimeError("the path does not reproduce the later cases")
    return 0


def test_dates(ids: list[str]) -> dict[str, str]:
    want, out = set(ids), {}
    with SOURCES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("query_id") in want:
                out[r["query_id"]] = r["publication_date"]
    return out


def score() -> int:
    """The labels, once. ``FINAL_SYSTEM.json`` names the systems; the result files are written before anything is printed."""
    from uol.final import llm_agents as LG, measures, new_cases_report as NR
    from uol.round3 import common as C
    if SCORED.exists():
        raise RuntimeError("the test set has been scored; it is scored once")
    if not FINAL.exists():
        raise RuntimeError("no FINAL_SYSTEM.json: the final system is not frozen")
    if not (OUT / "control_later.json").exists():
        raise RuntimeError("run `control_later` first")
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    d = extended()
    C.lockbox_read("final_score", f"one score of {final['systems']}", Path(__file__))
    generic = {int(i) for i in d.generic_idx}
    rf = rows_file()
    test = rf["rows"]
    if [d.queries[r].query_id for r in test] != rf["query_ids"]:
        raise RuntimeError("the test rows moved in the extended release")
    later, ll, dates = later_lists(d, final["passes"])
    dates = {**test_dates(rf["query_ids"]), **dates}
    here = {"combo": {int(k): v for k, v in json.loads((OUT / "combo_ranked.json").read_text(encoding="utf-8")).items()},
            "ranker": {int(k): v for k, v in json.loads((OUT / "ranker_top200.json").read_text(encoding="utf-8")).items()}}
    answers = json.loads((OUT / "alone_answers.json").read_text(encoding="utf-8"))
    here["alone"] = {r: LG.resolved(answers[str(r)]["articulos"], generic) for r in test}
    rows = test + later
    n = len(d.candidate_ids)
    frames = {}
    for name in final["systems"]:
        frames[name], _ = measure(rows, matrix(rows, {**here[name], **ll[name]}, n), True)
        frames[name]["publication_date"] = frames[name]["query_id"].map(dates)
    first = frames[final["systems"][0]]
    cut = FT.CUTOFFS[LG.KIND]
    parts = {"all": set(first["query_id"]), "without_later_46": set(rf["query_ids"]) & set(first["query_id"]),
             "after_model_cutoff": set(first[first["publication_date"] > cut]["query_id"]),
             "before_model_cutoff": set(first[first["publication_date"] <= cut]["query_id"])}
    res = {"final_system": final, "cutoff": cut, "cases": {k: len(v) for k, v in parts.items()}, "summaries": {}, "contrasts": {}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    names = list(frames)
    for part, keep in parts.items():
        res["summaries"][part] = {k: measures.summary(f[f["query_id"].isin(keep)]) for k, f in frames.items()}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                res["contrasts"][f"{part}: {a} - {b}"] = NR.pair(frames, a, b, "ndcg_at_10", len(names) * (len(names) - 1) // 2, keep)
    for name, f in frames.items():
        f.to_parquet(OUT / f"per_case_{name}.parquet", index=False)
    A.write_json(SCORED, res)
    print(json.dumps({"cases": res["cases"], "summaries": {p: {k: {m: v.get(m) for m in ("ndcg_at_10", "success_at_10", "success_at_1", "n_cases", "n_groups")} for k, v in s.items()} for p, s in res["summaries"].items()},
                      "contrasts": res["contrasts"]}, indent=1, default=float), flush=True)
    return 0


STEPS = {"prepare": prepare, "control": control, "control_later": control_later, "rank": rank, "alone": alone, "rerank": rerank, "score": score}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step}")
        return 0
    return STEPS[a.step]()


if __name__ == "__main__":
    sys.exit(main())
