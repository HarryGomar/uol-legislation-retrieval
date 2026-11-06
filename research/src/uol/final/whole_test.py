"""The test cases on the whole run's path (FRAMING decision 1; overnight_20261105/PLAN.md, "what landing a system means").

The models are fitted on all development cases with the choices made as in each fold, on development rows only. ``fit`` and
``control`` open no test case. ``rank`` answers the test cases from their facts, with fresh encoding and fresh model answers
and no labels. ``score`` opens the labels once. Every step that loads a test row (``load``, ``rank``, ``score``, and building the pack for those rows) stops unless ``TEST_RELEASE.json`` holds the author's word,
and they write to a new folder; ``test/`` and ``test_final/`` are read, never written.

    py -3.12 -m uol.final.whole_test fit --setup ranker --go       all-development model, setting and spec
    py -3.12 -m uol.final.whole_test control --setup ranker --go   the saved model through the live path, development cases
    py -3.12 -m uol.final.whole_test load --setup ranker --go      the first stage of new cases loads; no list is kept
    py -3.12 -m uol.final.whole_test rank --setup system --encoder card --go
    py -3.12 -m uol.final.whole_test score --go
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from uol.final import whole_run as W

ROOT = W.ROOT
RUNS = ROOT / "experiments" / "runs" / "footing_20261101"
OUT = RUNS / "test_whole"
RELEASE = W.PROG / "TEST_RELEASE.json"
CONTROL_CASES = 20
COS_MIN = 0.999            # an encoding of a test case against its stored vector, before anything is answered
VOTES = {"key": "dense", "decay": "hl5", "k": 10}   # new_cases_fit.M1_CFG, best on all development cases in the arm's table
CONTRASTS = (("system", "alone"), ("system", "ranker"), ("ranker", "ranker_noenc"))


def folder_of(setup: str) -> Path:
    return W.WORK / "models" / f"{W.SIDS[setup]}_all"


def fit(setup: str) -> int:
    """As ``ranker_tests.fit_group`` on the all-development group: the setting is chosen on the same inner folds."""
    from uol.embedding_push import fusion_ltr as fl
    from uol.final import lean_rerun as LR, populations, ranker_tests as RT, registry
    from uol.gap_tests import ltr_listwise as LW
    LR.D.activate(W.STORE)
    LR._overlay_stat_glob()
    from uol.round3 import m1_memory                           # after the profile: its cache path is read at import
    sid = W.SIDS[setup]
    pop = populations.load("DEV", None)
    sc = registry.make(sid)
    sc.prepare(pop)
    h = sc.h
    g0 = next(x for x in h.groups if x.role == "dev_fold0")
    key = getattr(sc, "extension_key", None)
    cfg_m1 = {**VOTES, **({"key": key} if key else {})}
    extra = np.asarray(m1_memory.columns_for(cfg_m1, g0, h), dtype=np.float32)   # the same for every development group
    from uol.final import new_cases_fit as NF
    if VOTES != NF.M1_CFG:
        raise RuntimeError("the votes' all-development setting differs from the record's choice")
    dry = NF.dry_rows(pop)
    g = NF.lockbox_group(sc, dry)
    t0 = time.perf_counter()
    res, out_idx, score = RT.fit_group(h.ctx, h.F, h.v, g, extra, "lgbm", "stored", log=lambda m: print(m, flush=True))
    cfg = res["chosen"]
    seed = fl.SEED + 1000 * g.idx
    tr = RT.kept_rows(h.ctx, fl.train_rows_of(h.ctx, g), None)
    idx, y, counts = LW.training_lists(h.ctx, h.F, h.v, tr, np.random.default_rng(seed + 999), n_neg=cfg.get("n_neg", LW.N_NEG))
    model = RT._fit("lgbm", cfg, RT._matrix(h.F, h.v, idx, extra), y, counts, seed)
    if not np.array_equal(RT._predict(model, h.F, h.v, out_idx, extra), score):
        raise RuntimeError("the saved model differs from the chosen fit")
    folder = folder_of(setup)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    table, _ = fl.instrument_tables(h.ctx)
    np.save(folder / "instrument.npy", np.asarray(table[g.idx], dtype=np.float32))
    prev = json.loads((W.WORK / "models" / sid / "spec.json").read_text(encoding="utf-8"))
    spec = {"pool_ids": list(h.v.pool_ids), "feature_ids": list(h.v.feature_ids), "group": int(g.idx), "columns": len(h.v.hgb_cols) + 3, "w": prev.get("w")}
    if spec["columns"] != prev["columns"] or spec["pool_ids"] != prev["pool_ids"] or spec["feature_ids"] != prev["feature_ids"]:
        raise RuntimeError("the all-development columns differ from the fold's")
    (folder / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    meta = {"sid": sid, "group": int(g.idx), "role": g.role, "setting": cfg, "m1": cfg_m1, "grid": res["grid"], "train_cases": int(len(tr)), "train_pairs": res["final_train_pairs"],
            "seconds": round(time.perf_counter() - t0, 1), "sha256": W.sha256(folder / "model.pkl"), "control_top200": control_lists(sc, g, out_idx, score)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("grid", "control_top200")}, indent=1, default=str), flush=True)
    return 0


def control_lists(sc, g, out_idx, score) -> dict[str, list[int]]:
    """The saved model's first 200 on the control cases, by the pair table, written as the harness writes them."""
    from uol.embedding_push import common as ep, fusion_ltr as fl
    from uol.final.adapters.ltr import _RowSink
    d = ep.release()
    rows = np.asarray(g.rows)
    sink = _RowSink(rows, sc.N)
    fl.write_pooled(sc.h.ctx, sc.h.F, sink, rows, out_idx, score, sc.lstar)
    generic = {int(i) for i in d.generic_idx}
    return {d.queries[int(r)].query_id: [int(c) for c in np.argsort(-sink.out[i], kind="stable")[:200 + len(generic)] if int(c) not in generic][:200] for i, r in enumerate(rows)}


def control(setup: str) -> int:
    """The saved all-development model through the live path on development cases, with the fold's instrument prior, so that
    each input equals the pair table's: the lists must match. Stored query vectors; no test case is opened."""
    from uol.final import timing as T, timing_ranker as TR
    W.refuse_stored()
    folder = folder_of(setup)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    live = W.Live(setup if setup != "system" else "ranker", "stored" if setup == "ranker" else "none", "control", models=folder)
    live.rk.instrument = np.load(W.WORK / "models" / W.SIDS[setup] / "instrument.npy")
    rows, _ = TR.case_rows(CONTROL_CASES, 0)
    same10 = 0
    over = []
    for r in rows:
        got, _ = live.ranker_list(r, T.Clock())
        want = meta["control_top200"][live.d.queries[r].query_id]
        same10 += got[:10] == want[:10]
        over.append(len(set(got) & set(want)) / len(want))
    res = {"cases": len(rows), "top10_same": same10, "top200_overlap": float(np.mean(over))}
    (folder / "control.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)
    return 0 if same10 == len(rows) else 1


# ------------------------------------------------------------------------------------------------ the test cases


def need_release() -> dict:
    return W.need_test_release()


def new_cases() -> dict:
    """The test rows and the later cases, in the extended release, as ``test_final`` holds them. Closed until the release."""
    need_release()
    from uol.final import new_cases_fit as NF, new_cases_sources as NSrc, test_final as TF, test_set as TS
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def activate():
        NSrc.activate_ext(TS.FOOTING_NEW / "release_ext", logs)
        NF.point_m1_stage()
        NSrc.stored_qc_v11()

    rf = TS.rows_file()
    _, _, _, later = TF._later_sources()
    rows = [int(r) for r in rf["rows"]] + [int(r) for r in later]
    qc = np.load(TS.AUDIT_NEW / "vectors" / "new_Qc.npy")
    return {"activate": activate, "rows": rows, "append": {"Qc": qc}, "dates": {}, "query_ids": rf["query_ids"], "new_rows": len(qc)}


def blank_vectors(spec: dict) -> None:
    """The query arrays the first stage loads but these sources do not read, extended with empty rows for the later cases."""
    from uol.embedding_push import common as ep
    from uol.final import timing_sources as TS
    n = spec["new_rows"]
    for name, path in (("bge", ep.DERIVED / "bge_m3_chunks_v1" / "query_vectors.npy"), ("e5", ep.DERIVED / "e5_small_queries_v11" / "query_vectors.npy"),
                       ("lsa", TS.staged_path("memory_proto/lsa_dev_v1.npy")), ("clsa", TS.staged_path("memory_proto/clsa_dev_v1.npy"))):
        spec["append"][name] = np.zeros((n, *np.load(path, mmap_mode="r").shape[1:]), dtype=np.float32)


def rows_and_dates(spec: dict) -> dict:
    from uol.embedding_push import common as ep
    from uol.final import test_set as TS
    d = ep.release()
    ids = [d.queries[r].query_id for r in spec["rows"]]
    found = TS.test_dates(ids)
    found.update({x["query_id"]: x["publication_date"] for x in json.loads((TS.FOOTING_NEW / "release_ext" / "new_rows.json").read_text(encoding="utf-8"))["rows"] if x.get("publication_date")})
    return {r: np.datetime64(found[q], "D") for r, q in zip(spec["rows"], ids) if q in found}


def answer_model(live, row: int, ranked: list[int]) -> dict:
    lst, usage = live.model.model_list(row)
    merged = live.model.G.rrf([lst, ranked])
    final, u2 = live.model.rerank(row, merged)
    return {"model_list": lst, "merged": merged, "final": final, "usd": live.model.usd({**usage, **u2})}


def load_only(setup: str) -> int:
    """Loads the first stage for new cases. Its warm-up reads the sources of the first row, a test row, so it waits for the release."""
    need_release()
    spec = new_cases()
    spec["activate"]()
    spec["dates"] = rows_and_dates(spec)
    spec["activate"] = lambda: None
    blank_vectors(spec)
    live = W.Live("ranker" if setup == "system" else setup, "stored" if setup != "noenc" else "none", "load", models=folder_of(setup), new_cases=spec)
    print(json.dumps({"load_seconds": live.load, "rows": len(spec["rows"]), "dates": len(spec["dates"])}), flush=True)
    return 0


def rank(setup: str, encoder: str, workers: int) -> int:
    from uol.final import timing as T
    release = need_release()
    OUT.mkdir(parents=True, exist_ok=True)
    W.refuse_stored()
    out = OUT / f"{setup}_{encoder}"
    out.mkdir(parents=True, exist_ok=False)
    spec = new_cases()
    spec["activate"]()
    spec["dates"] = rows_and_dates(spec)
    spec["activate"] = lambda: None
    blank_vectors(spec)
    missing = [r for r in spec["rows"] if r not in spec["dates"]]
    if missing:
        raise RuntimeError(f"{len(missing)} cases without a date")
    live = W.Live(setup, encoder, "test", models=folder_of(setup), new_cases=spec)
    from uol.embedding_push import common as ep
    from uol.round3 import common as C
    none = np.empty(0, dtype=np.int64)
    C.lockbox_rows = ep.lockbox_rows = lambda: none         # after the load: the prompt builder refuses these rows; no label is read in this process
    t0 = time.perf_counter()
    recs: dict[int, dict] = {}
    for n, r in enumerate(spec["rows"]):
        ranked, info = live.ranker_list(r, T.Clock())
        if live.encoder is not None and n < 5 and info["cos_to_stored"] < COS_MIN:
            raise RuntimeError(f"row {r}: encoding against its stored vector, cosine {info['cos_to_stored']}")
        recs[r] = {"row": r, "query_id": live.d.queries[r].query_id, "info": info, "ranker_list": ranked, "final": ranked}
    (out / "ranker_lists.json").write_text(json.dumps({str(k): v for k, v in recs.items()}, default=float), encoding="utf-8")
    if live.model is not None:
        with ThreadPoolExecutor(workers) as pool:
            done = dict(zip(spec["rows"], pool.map(lambda r: answer_model(live, r, recs[r]["ranker_list"]), spec["rows"])))
        for r, x in done.items():
            recs[r].update(x)
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in spec["rows"]:
            f.write(json.dumps(recs[r], default=float) + "\n")
    res = {"setup": setup, "encoder": encoder, "cases": len(recs), "seconds": round(time.perf_counter() - t0, 1), "release": release, "models": json.loads((folder_of(setup) / "meta.json").read_text(encoding="utf-8"))["sha256"],
           "cos_min": min((x["info"].get("cos_to_stored", 1.0) for x in recs.values()), default=None), "usd_per_case": statistics.mean(x["usd"] for x in recs.values()) if live.model is not None else None}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if live.encoder is not None:
        live.encoder.close()
    print(json.dumps(res, default=float), flush=True)
    return 0


def score() -> int:
    """The labels, once. The files are written before anything is printed. Nothing here selects; what it shows is printed."""
    import pandas as pd
    from uol.final import context, footing as FT, llm_agents as LG, measures, new_cases_report as NR, new_cases_sources as NSrc, test_final as TF, test_set as TS
    from uol.embedding_push import common as ep
    from uol.round3 import common as C
    release = need_release()
    scored = OUT / "scored.json"
    if scored.exists():
        raise RuntimeError("scored once")
    NSrc.activate_ext(TS.FOOTING_NEW / "release_ext", OUT / "logs")
    context.apply_label_corrections()
    context.install_labelled_evaluate()
    d = TF.ep_release()
    C.lockbox_read("whole_score", f"FRAMING decision 1: one score of the system, the ranker and the ranker without the encoder; {release.get('word')}", Path(__file__))
    generic = {int(i) for i in d.generic_idx}
    rf = TS.rows_file()
    test = rf["rows"]
    if [d.queries[r].query_id for r in test] != rf["query_ids"]:
        raise RuntimeError("the test rows moved in the extended release")
    later, ll, dates = TS.later_lists(d, ["fwd"])
    dates = {**TS.test_dates(rf["query_ids"]), **dates}
    rows = test + later

    def load(name: str, key: str) -> dict[int, list[int]]:
        out = {}
        with (OUT / name / "cases.jsonl").open(encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                out[int(x["row"])] = x[key]
        return out

    # the model alone: its list of 20 asked fresh, one case per call, the same answer the system merges (the author, 5 Nov: "Yes, ask fresh")
    sysf = next(f"system_{e}" for e in ("card", "stored", "cpu") if (OUT / f"system_{e}" / "cases.jsonl").exists())
    lists = {"system": load(sysf, "final"), "ranker": load(sysf, "ranker_list"), "ranker_noenc": load("noenc_none", "final"),
             "alone": load(sysf, "model_list")}
    n = len(d.candidate_ids)
    frames = {}
    for name, li in lists.items():
        frames[name], _ = TS.measure(rows, TS.matrix(rows, {r: li[r] for r in rows}, n), True)
        frames[name]["publication_date"] = frames[name]["query_id"].map(dates)
    first = frames["system"]
    cut = FT.CUTOFFS[LG.KIND]
    parts = {"all": set(first["query_id"]), "without_later_46": set(rf["query_ids"]) & set(first["query_id"]),
             "after_model_cutoff": set(first[first["publication_date"] > cut]["query_id"]),
             "before_model_cutoff": set(first[first["publication_date"] <= cut]["query_id"])}
    res = {"release": release, "system_folder": sysf, "cutoff": cut, "cases": {k: len(v) for k, v in parts.items()},
           "summaries": {}, "contrasts": {}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for part, keep in parts.items():
        res["summaries"][part] = {k: measures.summary(f[f["query_id"].isin(keep)]) for k, f in frames.items()}
        for a, b in CONTRASTS:
            res["contrasts"][f"{part}: {a} - {b}"] = NR.pair(frames, a, b, "ndcg_at_10", len(CONTRASTS), keep)
    for name, f in frames.items():
        f.to_parquet(OUT / f"per_case_{name}.parquet", index=False)
    (scored).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"cases": res["cases"], "summaries": {p: {k: {m: v.get(m) for m in ("ndcg_at_10", "success_at_10", "n_cases", "n_groups")} for k, v in s.items()} for p, s in res["summaries"].items()},
                      "contrasts": {k: {x: v.get(x) for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in res["contrasts"].items() if v}}, indent=1, default=float), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit", "control", "load", "rank", "score"])
    ap.add_argument("--setup", choices=list(W.SIDS), default="ranker")
    ap.add_argument("--encoder", choices=["none", "stored", "card", "cpu"], default="card")   # stored: the vectors the same encoder computed earlier
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step} {a.setup}")
        return 0
    if a.step == "fit":
        return fit(a.setup)
    if a.step == "control":
        return control(a.setup)
    if a.step == "load":
        return load_only(a.setup)
    if a.step == "rank":
        return rank(a.setup, a.encoder, a.workers)
    return score()


if __name__ == "__main__":
    sys.exit(main())
