"""D181 re-runs on the development cases (prepared 28 Oct 2026; NOT launched). R1 and the DEV part of R3.

R1. ``S_full_minus_colbert``: the record's ``S_full`` entry with ``drop_sources: [colbert_bge]`` (the mechanism of
``S_full_ext_minus_colbert``, D79), refitted on the five development folds by ``harness.run`` exactly as S_full
(same features less the ColBERT columns, same per-group hyper-parameters and seeds, same topic grid, bundle profile,
corrected labels, 4 threads). Steps, in order:

    control    S_full fold 0 on this PC, with the d58 table and the fold-0 topic models of the 26 Oct spread run;
               must equal that run's fold-0 per-case results and top 200 exactly (the same-machine control)
    retrain    S_full on all five folds (the paired control), then S_full_minus_colbert on the same topic models;
               S_full against the record: reported, gated by --tolerance (RUN-B fitted other topic models)
               [--with-s0: S0 on this PC too, token R1-S0; --with-f8: S_full_minus_D and
               S_full_minus_colbert_minus_M, the two F8 rows whose base changes]
    contrasts  100,000-draw paired group bootstrap (D138 arithmetic) in the record's file format
    overlay    R4 input: a copy of the record's store with the audit runs and contrast files added

R3 (DEV):
    llm        the LLM stage over S_full_minus_colbert's top 30 (llm_jobs.llm_rerank: l1_luna main prompt, window 20,
               stride 10, two passes, 3 samples, gpt-5.6-luna via Codex CLI; fusion_route W_EXT per fold); --smoke N
    llm-eval   harness.run of llm_rerank_S_full_minus_colbert and its contrasts
    claude     Claude Opus 5.5 on DEV-A300 (external.drive_claude, one sample), fused per fold, evaluated

Every compute step is a dry run unless ``--go`` and ``UOL_AUDIT_D181_APPROVE`` holds its token (R1, R1-S0,
R3-gpt-dev, R3-claude-dev). Heavy steps go through the heavy-job wrappers, e.g.

    $env:UOL_FINAL_THREADS=4; $env:OMP_NUM_THREADS=4; $env:UOL_AUDIT_D181_APPROVE="R1"
        py -3.12 -m uol.round3.heavy uol.final.audit_d181_dev retrain --go

Outputs: ``<RUN>/dev/`` (store, contrasts_100k, checks), ``<RUN>/receipts``, ``<RUN>/logs``; work in ``<WORK>/dev``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from uol.final import audit_d181_common as A

DEV = A.RUN / "dev"
STORE = DEV / "store"
CONTROL_STORE = DEV / "control_fold0"
WORKD = A.WORK / "dev"
TOLERANCE = 0.003            # |S_full here - S_full record| on DEV nDCG@10 (fold-0 spread on this PC was .0016)


def inputs() -> dict[str, bool]:
    print("R1/R3-DEV inputs on this machine:")
    return A.check_paths({
        "record S_full DEV run": A.STORE / "runs" / "S_full" / "DEV" / "per_case.parquet",
        "record S0 DEV run": A.STORE / "runs" / "S0" / "DEV" / "per_case.parquet",
        "record S_full receipt (topic grid, folds)": A.STORE / "receipts" / "S_full__DEV.json",
        "record llm_rerank_S_full_ext DEV run": A.STORE / "runs" / "llm_rerank_S_full_ext" / "DEV" / "per_case.parquet",
        "record claude_rerank_S_full_ext DEV-A300 run": A.STORE / "runs" / "claude_rerank_S_full_ext" / "DEV-A300" / "per_case.parquet",
        "100k contrast files (F1, F6, F7)": A.BOOT100K / "F7.json",
        "bundle on Drive (BUNDLE.json)": A.BUNDLE_DATA / "BUNDLE.json",
        "bundle manifest": A.BUNDLE_DATA.parent / "MANIFEST.json",
        "bundle overlay (verified 26 Oct)": A.NEW_RUN / "bundle_overlay" / "OVERLAY.json",
        "bundle LLM cache seed": A.BUNDLE_DATA / "luna" / "cache_dev.sqlite",
        "frozen pipelines (LTR_emb_hgb)": A.ROOT / "experiments/runs/embedding_push_20261004/fusion/pipelines/LTR_emb_hgb/manifest.json",
        "M1 stage (activate() checks it)": A.ROOT / ".cache" / "timing_stage_20261026c" / "MANIFEST.json",
        "d58 development pair table (to hard-link)": A.NEW_WORK / "fit" / "d58" / "X_control.npy",
        "fold-0 topic models of the spread run": A.SPREAD / "topic",
        "spread fold-0 S_full per-case (exact control)": A.SPREAD / "store" / "runs" / "S_full" / "DEV" / "per_case.parquet",
        "spread fold-0 S_full top 200": A.SPREAD / "store" / "runs" / "S_full" / "DEV" / "top200.parquet",
        "Anthropic key file (.env, ANTHROPIC_KEY)": A.ENV_FILE,
        "Claude venv": A.CLAUDE_VENV,
    })


def activate(store: Path):
    """``new_cases_fit.activate`` with its store in the audit folder: bundle profile on the Drive bundle (overlay of
    verified local arrays), corrected labels, 4 threads, the d58 table hard-linked read-only, topic cache seeded."""
    A.need_threads()
    A.refuse_record_paths(store, WORKD)
    from uol.final import new_cases_fit as NF, new_cases_steps as NS
    os.environ["UOL_FINAL_LABELS"] = "corrected"
    WORKD.mkdir(parents=True, exist_ok=True)
    links = NS.link_fit_features(WORKD)                     # read-only hard links, no bytes written
    topics = A.seed_topic_cache(A.SPREAD / "topic", WORKD)  # fold-0 models: the control equals the spread run
    from uol.final import bundle_overlay, context, cost
    bundle_overlay.install(A.BUNDLE_DATA)
    prof = context.activate_bundle(A.BUNDLE_DATA, store=Path(store), work=WORKD)
    cost.set_threads(A.THREADS)
    NF.point_m1_stage()
    return prof, {"d58_links": links, "topic_seed": topics}


def _frames_equal(a, b, keys=("query_id",)) -> bool:
    a = a.sort_values(list(keys)).reset_index(drop=True)
    b = b.sort_values(list(keys)).reset_index(drop=True)
    return list(a.columns) == list(b.columns) and a.equals(b)


def control(go: bool) -> int:
    if not A.approved("R1", go):
        inputs()
        print(f"[dry] would run S_full DEV fold 0 into {CONTROL_STORE} and require per_case/top200 == {A.SPREAD}/store")
        return 0
    import pandas as pd
    from uol.final import harness
    prof, seeded = activate(CONTROL_STORE)
    t0 = time.perf_counter()
    harness.run("S_full", "DEV", store=CONTROL_STORE, folds=[0])
    mine = pd.read_parquet(CONTROL_STORE / "runs" / "S_full" / "DEV" / "per_case.parquet")
    ref = pd.read_parquet(A.SPREAD / "store" / "runs" / "S_full" / "DEV" / "per_case.parquet")
    tm = pd.read_parquet(CONTROL_STORE / "runs" / "S_full" / "DEV" / "top200.parquet")
    tr = pd.read_parquet(A.SPREAD / "store" / "runs" / "S_full" / "DEV" / "top200.parquet")
    res = {"per_case_identical": _frames_equal(mine, ref), "top200_identical": _frames_equal(tm, tr, ("query_id", "rank")),
           "ndcg_mine": float(mine["ndcg_at_10"].mean()), "ndcg_spread": float(ref["ndcg_at_10"].mean()),
           "seconds": round(time.perf_counter() - t0, 1), **seeded}
    res["pass"] = bool(res["per_case_identical"] and res["top200_identical"])
    A.write_json(DEV / "checks" / "control_fold0.json", res)
    A.receipt("dev_control", res)
    A.log(f"control fold 0: pass={res['pass']} ({res['seconds']}s)", "dev")
    return 0 if res["pass"] else 2


def retrain(go: bool, with_s0: bool, tolerance: float, cont: bool, with_f8: bool = False) -> int:
    ok = A.approved("R1", go)
    ok_s0 = A.approved("R1-S0", go) if with_s0 else False
    specs = [A.spec_minus_colbert()] + ([A.spec_minus_colbert_minus_m()] if with_f8 else [])
    if not ok:
        inputs()
        A.install_systems(specs, dry=True)
        extra = (" and S0" if with_s0 else "") + (f" and F8 rows S_full_minus_D, {A.NEW_SYS_MINUS_M}" if with_f8 else "")
        print(f"[dry] would run S_full (5 folds) then {A.NEW_SYS} (5 folds){extra} into {STORE}")
        for s in specs:
            print(f"[dry] spec {s['id']}: {json.dumps(s['adapter_args'])}")
        return 0
    chk = DEV / "checks" / "control_fold0.json"
    if not chk.exists() or not json.loads(chk.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("run and pass `control` first")
    import pandas as pd
    from uol.final import harness, measures
    A.install_systems(specs, dry=False)
    prof, seeded = activate(STORE)
    out = {"seeded": seeded}
    t0 = time.perf_counter()
    harness.run("S_full", "DEV", store=STORE)
    out["S_full_seconds"] = round(time.perf_counter() - t0, 1)
    mine = pd.read_parquet(STORE / "runs" / "S_full" / "DEV" / "per_case.parquet")
    rec = pd.read_parquet(A.STORE / "runs" / "S_full" / "DEV" / "per_case.parquet")
    a, b = measures.summary(mine), measures.summary(rec)
    out["S_full_here"], out["S_full_record"] = a["ndcg_at_10"], b["ndcg_at_10"]
    out["difference"] = a["ndcg_at_10"] - b["ndcg_at_10"]
    out["per_fold"] = {int(k): measures.summary(mine[mine["fold"] == k])["ndcg_at_10"] -
                       measures.summary(rec[rec["fold"] == k])["ndcg_at_10"] for k in sorted(rec["fold"].unique())}
    out["within_tolerance"] = abs(out["difference"]) <= tolerance
    out["tolerance"] = tolerance
    A.write_json(DEV / "checks" / "S_full_vs_record.json", out)
    A.log(f"S_full here {a['ndcg_at_10']:.6f} vs record {b['ndcg_at_10']:.6f} (diff {out['difference']:+.6f})", "dev")
    if not out["within_tolerance"] and not cont:
        A.log("S_full outside the tolerance: stopped before the leave-one-out (rerun with --continue to override)", "dev")
        return 3
    t0 = time.perf_counter()
    harness.run(A.NEW_SYS, "DEV", store=STORE)
    out[f"{A.NEW_SYS}_seconds"] = round(time.perf_counter() - t0, 1)
    for sid in (["S0"] if ok_s0 else []) + (["S_full_minus_D", A.NEW_SYS_MINUS_M] if with_f8 else []):
        t0 = time.perf_counter()
        harness.run(sid, "DEV", store=STORE)
        out[f"{sid}_seconds"] = round(time.perf_counter() - t0, 1)
    A.write_json(DEV / "checks" / "S_full_vs_record.json", out)
    A.receipt("dev_retrain", out)
    return 0


def overlay(go: bool) -> int:
    """R4 input: ``<RUN>/overlay_store`` = a verified copy of the record's store plus the audit runs, receipts and
    status files, and the D181 contrast files under ``contrasts_d181/`` (the record itself is never touched)."""
    over = A.RUN / "overlay_store"
    if not A.approved("R1", go):
        print(f"[dry] would copy {A.STORE} (713 MB) to {over} and add the audit runs of {STORE} and {DEV / 'contrasts_100k'}")
        return 0
    A.refuse_record_paths(over)
    A.copy_tree_verified(A.STORE, over)
    added = []
    for sub in ("runs", "receipts", "status"):
        for p in sorted((STORE / sub).rglob("*")):
            rel = p.relative_to(STORE)
            if p.is_file() and not (over / rel).exists():
                A.copy_verified(p, over / rel)
                added.append(str(rel))
    for p in sorted((DEV / "contrasts_100k").glob("*.json")):
        A.copy_verified(p, over / "contrasts_d181" / p.name)
    A.receipt("dev_overlay", {"added": added, "record": str(A.STORE)})
    return 0


def contrasts(go: bool) -> int:
    if not A.approved("R1", go):
        print(f"[dry] would write {DEV / 'contrasts_100k'}: D181a S_full(here) - {A.NEW_SYS}; D181b {A.NEW_SYS} - S0 "
              f"(record; and S0 here if run); D181c S_full(record) - {A.NEW_SYS}; D181d S_full(here) - S_full(record); "
              f"{A.DRAWS_DEV} draws, seed {A.SEED}, margin {A.MARGIN}")
        return 0
    here = lambda s: A.per_case(STORE, s)                  # the statistics need no profile: per-case files only
    rec = lambda s: A.per_case(A.STORE, s)
    fams = {
        "D181a": [("S_full", here("S_full"), A.NEW_SYS, here(A.NEW_SYS))],                 # format of F7 (m = 1)
        "D181b": [(A.NEW_SYS, here(A.NEW_SYS), "S0", rec("S0"))],                          # format of F1 (m = 1)
        "D181c": [("S_full@record", rec("S_full"), A.NEW_SYS, here(A.NEW_SYS))],
        "D181d": [("S_full", here("S_full"), "S_full@record", rec("S_full"))],
    }
    if (STORE / "runs" / "S0" / "DEV" / "per_case.parquet").exists():
        fams["D181b_here"] = [(A.NEW_SYS, here(A.NEW_SYS), "S0", here("S0"))]
    if all((STORE / "runs" / s / "DEV" / "per_case.parquet").exists() for s in ("S_full_minus_D", A.NEW_SYS_MINUS_M)):
        fams["D181_F8"] = [(A.NEW_SYS, here(A.NEW_SYS), "S_full_minus_D", here("S_full_minus_D")),     # F8 (m = 3)
                           (A.NEW_SYS, here(A.NEW_SYS), A.NEW_SYS_MINUS_M, here(A.NEW_SYS_MINUS_M)),
                           ("S_full_ext", rec("S_full_ext"), "S_full_ext_minus_qwen3_8b", rec("S_full_ext_minus_qwen3_8b"))]
    out = DEV / "contrasts_100k"
    t0 = time.perf_counter()
    res = {}
    for name, pairs in fams.items():
        f = A.dev_family(name, pairs)
        A.write_json(out / f"{name}.json", f)
        res[name] = {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class", "p_holm")} for k, v in f["contrasts"].items()}
    A.write_json(out / "SUMMARY.json", {"families": res, "seconds": round(time.perf_counter() - t0, 1),
                                         "rule": "class by the adjusted lower end against the .02 margin (section 4.4)"})
    A.receipt("dev_contrasts", {"families": res})
    print(json.dumps(res, indent=1))
    return 0


def llm(go: bool, smoke: int | None) -> int:
    if not A.approved("R3-gpt-dev", go):
        print(f"[dry] would rerank {A.NEW_SYS}'s DEV top 30 with gpt-5.6-luna (3 samples x 2 passes x 2 windows, 5 "
              f"cases per call: ~3,270 calls, tag final_L1_{A.NEW_SYS}) into {WORKD / 'llm' / A.LLM_SYS}")
        print(f"[dry] codex sign-in: {A.codex_signed_in()}")
        return 0
    s = A.codex_signed_in()
    if not s["ok"]:
        raise RuntimeError(f"Codex CLI not signed in: {s}")
    if not (STORE / "runs" / A.NEW_SYS / "DEV" / "top200.parquet").exists():
        raise RuntimeError("R1 retrain first: the base top 200 is missing")
    from uol.final import llm_jobs
    A.install_systems([A.spec_minus_colbert(), A.spec_llm(A.NEW_SYS, A.LLM_SYS, "gpt")], dry=False)
    activate(STORE)
    t0 = time.perf_counter()
    p = llm_jobs.llm_rerank(A.NEW_SYS, samples=3, store=STORE, limit=smoke, smoke=bool(smoke))
    A.receipt("dev_llm" + ("_smoke" if smoke else ""), {"output": str(p), "seconds": round(time.perf_counter() - t0, 1),
                                                       "codex": s, "usage": usage(f"final_L1_{A.NEW_SYS}")})
    return 0


def usage(tag_prefix: str) -> dict:
    """Calls and tokens of this run from ``<work>/luna/calls.jsonl`` (each cache key once; reused answers flagged),
    USD at the list prices of prices_20261025.json (Table B.3; Codex CLI itself bills the ChatGPT plan)."""
    p = WORKD / "luna" / "calls.jsonl"
    seen, n_cached = {}, 0
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if str(r.get("tag", "")).startswith(tag_prefix):
                n_cached += bool(r.get("cached"))
                seen.setdefault(r["key"], r)
    u = [r.get("usage") or {} for r in seen.values()]
    tin = sum(int(x.get("input_tokens") or 0) for x in u)
    tc = sum(int(x.get("cached_input_tokens") or 0) for x in u)
    tout = sum(int(x.get("output_tokens") or 0) for x in u)
    P = json.loads((A.ROOT / "research/programs/final_benchmark_20261021/prices_20261025.json").read_text(encoding="utf-8"))
    usd = ((tin - tc) * P["llm_input_per_mtok_usd"] + tc * P["llm_cached_input_per_mtok_usd"]
           + tout * P["llm_output_per_mtok_usd"]) / 1e6
    return {"calls": len(seen), "cache_reused_answers": n_cached, "input_tokens": tin, "cached_input_tokens": tc,
            "output_tokens": tout, "usd_list_equivalent": round(usd, 4)}


def llm_eval(go: bool) -> int:
    if not A.approved("R3-gpt-dev", go):
        print(f"[dry] would evaluate {A.LLM_SYS} on DEV and write {DEV / 'contrasts_100k'}/D181e (F6-sized, m = 9, and m = 1)")
        return 0
    from uol.final import harness
    A.install_systems([A.spec_minus_colbert(), A.spec_llm(A.NEW_SYS, A.LLM_SYS, "gpt")], dry=False)
    activate(STORE)
    harness.run(A.LLM_SYS, "DEV", store=STORE)
    here = lambda s: A.per_case(STORE, s)
    pairs = [(A.LLM_SYS, here(A.LLM_SYS), A.NEW_SYS, here(A.NEW_SYS))]
    f1 = A.dev_family("D181e", pairs)
    f9 = A.dev_family("D181e_m9", pairs, size=9)             # sized like F6, where llm_rerank_S_full_ext - S_full_ext sits
    A.write_json(DEV / "contrasts_100k" / "D181e.json", f1)
    A.write_json(DEV / "contrasts_100k" / "D181e_m9.json", f9)
    meta = json.loads((WORKD / "llm" / A.LLM_SYS / "meta.json").read_text(encoding="utf-8"))
    A.receipt("dev_llm_eval", {"w_all": meta.get("w_all"), "w_by_fold": meta.get("w_by_fold"),
                               "usage": usage(f"final_L1_{A.NEW_SYS}")})
    return 0


def claude(go: bool) -> int:
    if not A.approved("R3-claude-dev", go):
        print(f"[dry] would rerank {A.NEW_SYS}'s DEV-A300 top 30 with claude-opus-5-5 (one sample; the record's run: "
              "US$16.10, 300 cases) into <WORK>/dev/external, fuse per fold and evaluate on DEV-A300")
        return 0
    from uol.final import external as X, harness
    A.install_systems([A.spec_minus_colbert(), A.spec_llm(A.NEW_SYS, A.CLAUDE_SYS, "claude")], dry=False)
    prof, _ = activate(STORE)
    X.BASE = A.NEW_SYS                                       # in this process only: the base of fuse()
    X.SYSTEMS["claude"] = A.CLAUDE_SYS
    t0 = time.perf_counter()
    path = X.drive_claude(STORE / "runs" / A.NEW_SYS / "DEV" / "top200.parquet", X.ext_dir(), A.CLAUDE_VENV, A.ENV_FILE)
    X.fuse("claude", STORE)
    harness.run(A.CLAUDE_SYS, "DEV-A300", store=STORE)
    here = lambda s, p="DEV-A300": A.per_case(STORE, s, p)
    f = A.dev_family("D181f", [(A.CLAUDE_SYS, here(A.CLAUDE_SYS), A.NEW_SYS, here(A.NEW_SYS, "DEV"))], population="DEV-A300")
    A.write_json(DEV / "contrasts_100k" / "D181f.json", f)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    A.receipt("dev_claude", {"orders": str(path), "usd": o.get("usd"), "usage": o.get("usage"),
                             "seconds": round(time.perf_counter() - t0, 1)})
    return 0


def claude_full(go: bool) -> int:
    """D188: the reread on every P_spec development case. DEV-A300 keeps its answers; only the other rows are asked
    (their own folder, cache and ledger), the orders are merged, and the merged orders are fused per fold."""
    from uol.embedding_push import common as ep
    from uol.final import external as X, guard, harness, populations
    if not A.approved("R3-claude-dev-full", go):
        print(f"[dry] would rerank {A.NEW_SYS}'s top 30 on the 917 other rows (DEV-A300 reused) into "
              "<WORK>/dev/external/full, merge, fuse per fold and evaluate on DEV")
        return 0
    A.install_systems([A.spec_llm(A.NEW_SYS, A.CLAUDE_SYS, "claude")], dry=False)
    prof, _ = activate(STORE)
    X.BASE = A.NEW_SYS
    X.SYSTEMS["claude"] = A.CLAUDE_SYS
    ext = X.ext_dir()
    orders = ext / X.FILES["claude"]
    keep = ext / "claude_orders_DEV-A300.json"
    if not keep.exists():
        keep.write_bytes(orders.read_bytes())
    dev = ep.dev_rows()                                      # row numbers belong to the release activate() loaded
    spec = [int(r) for r in ep.population_rows("P_spec", dev)]
    sample = {int(r) for r in populations.agent_sample(dev)}
    rest = [int(r) for r in guard.dev_rows_only([r for r in spec if r not in sample])]
    if sample != set(json.loads(keep.read_text(encoding="utf-8"))["rows"]) or len(spec) != 1217 or len(rest) != 917:
        raise RuntimeError(f"rows do not match DEV-A300's answers: {len(spec)} P_spec, {len(rest)} to ask")
    print(f"{len(rest)} rows to ask, {len(sample)} reused, {len(spec)} P_spec", flush=True)
    if os.environ.get("UOL_D188_CHECK_ONLY"):                # the row check, before anything is sent
        return 0
    t0 = time.perf_counter()
    part = X.drive_claude(STORE / "runs" / A.NEW_SYS / "DEV" / "top200.parquet", ext / "full", A.CLAUDE_VENV, A.ENV_FILE,
                          rows=rest)
    if os.environ.get("UOL_D188_ASK_ONLY"):                  # the API wait holds no heavy lock; the rerun reads the cache
        return 0
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (keep, part))
    if set(a["rows"]) & set(b["rows"]) or set(a["rows"]) | set(b["rows"]) != set(spec):
        raise RuntimeError("the merged orders do not cover the P_spec development rows exactly once")
    usd = round(a["usd"] + b["usd"], 4)
    usage = {k: a["usage"][k] + b["usage"][k] for k in a["usage"]}
    merged = {**b, "rows": sorted(a["rows"] + b["rows"]), "population": "DEV", "usd": usd, "usage": usage,
              "unanswered": a["unanswered"] + b["unanswered"], "repairs": a["repairs"] + b["repairs"],
              "orders": {p: {**a["orders"][p], **b["orders"][p]} for p in a["passes"]}}
    orders.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    X.fuse("claude", STORE)
    harness.run(A.CLAUDE_SYS, "DEV", store=STORE)
    here = lambda s: A.per_case(STORE, s, "DEV")
    f = A.dev_family("D188f", [(A.CLAUDE_SYS, here(A.CLAUDE_SYS), A.NEW_SYS, here(A.NEW_SYS))], population="DEV")
    A.write_json(DEV / "contrasts_100k" / "D188f.json", f)
    A.receipt("dev_claude_full", {"orders": str(orders), "usd": usd, "usd_this_step": b["usd"], "usage": usage,
                                  "rows_asked": len(rest), "seconds": round(time.perf_counter() - t0, 1)})
    return 0


LIVE_USD_IN, LIVE_USD_OUT = 4.0, 20.0     # per million tokens, Messages API (scripts/c10_api.py SYNC_IN, SYNC_OUT)
LIVE_CAP_USD, LIVE_CASE_USD = 0.50, 0.12  # D188: the author's US$0.50; a case starts only if its likely price still fits


def reread_timing(go: bool) -> int:
    """D188: wall seconds and price of the reread per case, one case at a time through the Messages API, with the same
    prompt, windows and passes as the batch run. A pass depends only on itself, so the two can also run side by side."""
    if not A.approved("R3-claude-live", go):
        print(f"[dry] would time the reread on DEV-A300 cases, one at a time, up to US${LIVE_CAP_USD:.2f}")
        return 0
    import subprocess
    import numpy as np
    from uol.embedding_push import common as ep
    from uol.final import external as X, populations
    activate(STORE)
    rows = np.random.default_rng(A.SEED).permutation(populations.agent_sample(ep.dev_rows())).tolist()
    work = X.ext_dir() / "live"
    work.mkdir(parents=True, exist_ok=True)
    script = A.ROOT / "scripts" / "reread_live.py"
    calls: list[dict] = []

    def api(requests: list[dict]) -> dict[str, dict]:
        req, res = work / "requests.jsonl", work / "responses.jsonl"
        req.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests), encoding="utf-8")
        r = subprocess.run([str(A.CLAUDE_VENV), str(script), "--requests", str(req), "--responses", str(res), "--env",
                            str(A.ENV_FILE)], capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"reread_live failed ({r.returncode})")
        got = [json.loads(x) for x in res.read_text(encoding="utf-8").splitlines()]
        calls.extend({"row": current, **{k: x[k] for k in ("custom_id", "seconds", "input_tokens", "output_tokens", "stop_reason")}}
                     for x in got)
        return {x["custom_id"]: x for x in got}

    spent = lambda: sum(c["input_tokens"] * LIVE_USD_IN + c["output_tokens"] * LIVE_USD_OUT for c in calls) / 1e6
    cases = []
    for current in rows:
        if spent() + LIVE_CASE_USD > LIVE_CAP_USD:
            break
        n = len(calls)
        stats: dict[str, int] = {}
        top = X.top30(STORE / "runs" / A.NEW_SYS / "DEV" / "top200.parquet", [current])
        X.rerank_orders([current], top, lambda items: X.ask_claude(items, api, stats, X.CLAUDE))
        one = calls[n:]
        by_pass = {p: sum(c["seconds"] for c in one if c["custom_id"].startswith(p)) for p in ("fwd", "rev")}
        cases.append({"row": int(current), "calls": len(one), "seconds_one_after_other": sum(c["seconds"] for c in one),
                      "seconds_side_by_side": max(by_pass.values()),
                      "usd": sum(c["input_tokens"] * LIVE_USD_IN + c["output_tokens"] * LIVE_USD_OUT for c in one) / 1e6, **stats})
    rec = {"model": X.CLAUDE["model"], "effort": X.CLAUDE["effort"], "route": "Messages API, one case per prompt",
           "price_per_mtok": [LIVE_USD_IN, LIVE_USD_OUT], "usd": spent(), "cases": cases, "calls": calls}
    A.write_json(DEV / "reread_timing.json", rec)
    A.receipt("dev_reread_timing", {k: rec[k] for k in ("model", "route", "usd")} | {"cases": len(cases)})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final.audit_d181_dev", description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["inputs", "control", "retrain", "contrasts", "overlay", "llm", "llm-eval", "claude", "claude-full", "reread-timing"])
    ap.add_argument("--go", action="store_true", help="really run (needs UOL_AUDIT_D181_APPROVE); default: dry run")
    ap.add_argument("--with-s0", action="store_true", help="retrain: also S0 on this PC (token R1-S0)")
    ap.add_argument("--with-f8", action="store_true", help=f"retrain: also S_full_minus_D and {A.NEW_SYS_MINUS_M} (F8)")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--continue", dest="cont", action="store_true", help="retrain: go on outside the tolerance")
    ap.add_argument("--smoke", type=int, default=None, help="llm: the first N rows only (still calls the model)")
    a = ap.parse_args(argv)
    if a.step == "inputs":
        res = inputs()
        print(f"codex: {A.codex_signed_in()}")
        return 0 if all(res.values()) else 1
    if a.step == "control":
        return control(a.go)
    if a.step == "retrain":
        return retrain(a.go, a.with_s0, a.tolerance, a.cont, a.with_f8)
    if a.step == "contrasts":
        return contrasts(a.go)
    if a.step == "overlay":
        return overlay(a.go)
    if a.step == "llm":
        return llm(a.go, a.smoke)
    if a.step == "llm-eval":
        return llm_eval(a.go)
    if a.step == "reread-timing":
        return reread_timing(a.go)
    return claude_full(a.go) if a.step == "claude-full" else claude(a.go)


if __name__ == "__main__":
    sys.exit(main())
